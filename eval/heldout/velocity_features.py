"""R6 벨로시티 개선 — 오디오 특징량 후보를 정답 온셋 위치에서 비교한다(개발셋에서만 고른다).

Spearman ρ는 단조 변환에 불변이라, 매핑 곡선이 아니라 "무엇을 재는가"를 바꿔야 순위가 바뀐다.
베이스 후보(분리 스템, 노트 시작 t):
  rms     현행 하이브리드: [t, t+120ms] RMS 최대(dB)
  rise    rms − 직전 [t−60, t−10ms] RMS(dB) — 이전 노트 잔향을 뺀 증가량
  harm    [t, t+80ms] 노트 기본음·2배음 CQT 크기 최대(dB) — 다른 피치 잔향에 덜 민감
  flux    온셋 강도(스펙트럴 플럭스) [t−30, t+30ms] 최대
  rms_pc  rms에서 곡 내 피치에 대한 선형 추세를 뺀 잔차(저음이 원래 크게 녹음되는 편향 제거)
드럼 후보(분리 드럼 스템, 5클래스):
  attack  현행: drum_features.attack_energy(광대역 어택 RMS)
  band    클래스별 대역(킥 30–150 Hz, 스네어 150–4k, 하이햇·심벌 5–16k, 톰 60–500) 50 ms 최대(dB)
  band_z  band를 곡·클래스 안에서 z-점수화(클래스 사이 크기 차이 제거)
곡별 Spearman(정답 벨로시티 고유값 ≥5, 표준편차 ≥5, 노트 ≥20인 곡만).

사용법: .venv-backend/bin/python eval/heldout/velocity_features.py --kind bass --set dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
import run_heldout as R  # noqa: E402
import score_drums  # noqa: E402

SR = 22050
HOP = 128
BANDS = {"kick": (30, 150), "snare": (150, 4000), "hihat": (5000, 16000), "cymbal": (5000, 16000), "tom": (60, 500)}


def sets(kind: str, name: str):
    """[(song, audio, gt_mid)]"""
    stem = "bass" if kind == "bass" else "drums"
    if name == "dev":
        songs = [i["song"] for i in json.loads((HERE / "data" / "slakh_train_manifest.json").read_text())]
        return [(s, HERE / "runs" / "_sep" / "slakh_train" / s / f"{stem}.wav",
                 HERE / "data" / "slakh_train" / s / f"{stem}.mid") for s in songs
                if (HERE / "runs" / "_sep" / "slakh_train" / s / "stems.json").exists()]
    man = json.loads((HERE / "data" / "manifest.json").read_text())
    if name == "slakh":
        return [(i["song"], HERE / "runs" / "_sep" / "slakh" / i["song"] / f"{stem}.wav",
                 HERE / "data" / "slakh" / i["song"] / f"{stem}.mid") for i in man["slakh"]]
    return [(i["song"], HERE / "runs" / "_sep" / name / i["song"] / f"{stem}.wav",
             HERE / "data" / name / i["song"] / f"{stem}.mid") for i in man.get(name, [])]


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-7))


def bass_features(y: np.ndarray, notes) -> dict[str, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=HOP)[0]
    tt = np.arange(len(rms)) * HOP / SR
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(21), n_bins=72))

    def win(arr, a, b, fn=np.max):
        m = (tt[: len(arr)] >= a) & (tt[: len(arr)] < b)
        return fn(arr[: len(tt)][m]) if m.any() else 1e-7

    out = {k: [] for k in ("rms", "rise", "harm", "flux")}
    for s, _e, p in notes:
        r = win(rms, s, s + 0.12)
        pre = win(rms, s - 0.06, s - 0.01, np.mean)
        out["rms"].append(_db(r))
        out["rise"].append(_db(r) - _db(pre))
        k0, k1 = p - 21, p - 21 + 12
        hs = [win(C[k], s, s + 0.08) for k in (k0, k1) if 0 <= k < C.shape[0]]
        out["harm"].append(_db(max(hs)) if hs else -140.0)
        out["flux"].append(win(flux, s - 0.03, s + 0.03))
    res = {k: np.array(v, float) for k, v in out.items()}
    pitch = np.array([p for *_, p in notes], float)
    if np.std(pitch) > 0:
        coef = np.polyfit(pitch, res["rms"], 1)
        res["rms_pc"] = res["rms"] - np.polyval(coef, pitch)
    else:
        res["rms_pc"] = res["rms"]
    return res


def drum_features(y: np.ndarray, hits) -> dict[str, np.ndarray]:
    import drum_features as df

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    tt = np.arange(S.shape[1]) * HOP / SR
    att = df.attack_energy(librosa.resample(y, orig_sr=SR, target_sr=df.SR), np.array([t for t, _ in hits]), df.SR)
    band = []
    for t, c in hits:
        lo, hi = BANDS[c]
        fm = (freqs >= lo) & (freqs < hi)
        m = (tt >= t) & (tt < t + 0.05)
        band.append(_db(S[fm][:, m].sum(axis=0).max()) if m.any() else -140.0)
    band = np.array(band)
    z = band.copy()
    cls = np.array([c for _, c in hits])
    for c in set(cls):
        k = cls == c
        if k.sum() > 1 and band[k].std() > 0:
            z[k] = (band[k] - band[k].mean()) / band[k].std()
        else:
            z[k] = 0.0
    return {"attack": _db(np.asarray(att, float)), "band": band, "band_z": z}


def gt_bass(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    seen, out = set(), []
    for n in sorted((n for i in pm.instruments for n in i.notes), key=lambda n: n.start):
        key = (n.pitch, round(n.start, 2))
        if key in seen or n.start < R.IGNORE_HEAD:
            continue
        seen.add(key)
        out.append(((n.start, n.end, n.pitch), n.velocity))
    return out


def gt_drums(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    out = []
    for i in pm.instruments:
        for n in i.notes:
            c9 = score_drums.CLASS_MAP.get(n.pitch)
            c5 = R.COLLAPSE5.get(c9) if c9 else None
            if c5 and n.start >= R.IGNORE_HEAD:
                out.append(((n.start, c5), n.velocity))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("bass", "drums"), required=True)
    ap.add_argument("--set", default="dev")
    args = ap.parse_args()
    per = {}
    for song, audio, gt in sets(args.kind, args.set):
        if not audio.exists() or not gt.exists():
            continue
        rows = gt_bass(gt) if args.kind == "bass" else gt_drums(gt)
        v = np.array([r[1] for r in rows], float)
        if len(rows) < 20 or len(np.unique(v)) < 5 or v.std() < 5:
            continue
        y, _ = librosa.load(str(audio), sr=SR, mono=True)
        feats = bass_features(y, [r[0] for r in rows]) if args.kind == "bass" else drum_features(y, [r[0] for r in rows])
        per[song] = {k: float(spearmanr(v, f).statistic) if np.std(f) > 0 else 0.0 for k, f in feats.items()}
    names = list(next(iter(per.values())).keys()) if per else []
    print(f"{args.kind} {args.set}: n={len(per)}곡")
    for k in names:
        r = [per[s][k] for s in per]
        lo, hi = R.bootstrap_ci(r)
        d = [per[s][k] - per[s][names[0]] for s in per]
        dlo, dhi = R.bootstrap_ci(d)
        print(f"  {k:8s} mean ρ={np.mean(r):.3f} [{lo:.3f}, {hi:.3f}]  vs {names[0]}: {np.mean(d):+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    out = HERE / "runs" / "_velocity"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"features_{args.kind}_{args.set}.json").write_text(json.dumps(per, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
