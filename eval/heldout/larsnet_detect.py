"""Round 2 N2 — LarsNet 피스 스템 → 클래스별 검출 → 8클래스 드럼 MIDI (drums_r2.py가 부른다, 메인 venv).

사전 등록(결과를 보기 전, 2026-10-25 08:10 고정)
  시스템
    larsnet_onset  각 LarsNet 스템에서 단순 온셋(librosa onset_strength + peak_pick). 스템별 delta는
                   개발 학습 분할(Slakh train songs[:40])에서 그 스템 클래스의 F1 최대로 고른다.
    larsnet_adtof  각 LarsNet 스템에 ADTOF를 돌리고, 스템에 해당하는 클래스만 남긴다.
    adtof_split    현행 출력(htdemucs 스템 → ADTOF, 5클래스)을 그대로 두고, 하이햇 노트만 오픈/클로즈,
                   심벌 노트만 크래시/라이드로 나눈다(LarsNet hihat·cymbals 스템의 특징). 5클래스 결과는 현행과 같다.
  하이햇·심벌 세분(세 시스템 공통): LarsNet 해당 스템에서 온셋 주변 특징(고역 감쇠 3구간, 중심 주파수,
    다음 타격까지 간격)을 뽑아 로지스틱 회귀(open vs closed, ride vs crash)를 Slakh train songs[:40]의
    정답 온셋으로 학습. 문턱 0.5 고정.
  선택: 개발 검증 분할(songs[40:], 17곡)에서 8클래스 F1이 가장 높은 시스템 하나만 MDB·Slakh test에 1회.
  판정: MDB 8클래스 F1이 현행(stem_adtof) 대비 paired CI 하한 > 0, 그리고 MDB 5클래스 F1 유의 하락 없음.
  렌더링(Slakh train)으로만 학습·튜닝하므로 실제 녹음(MDB)이 판정 세트다.

LarsNet 스템은 runs/_sep_larsnet/<세트>/<곡>/에 캐시한다(채점 뒤 삭제).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs"
DATA = HERE / "data"
SEP = RUNS / "_sep_larsnet"
MODEL_DIR = RUNS / "_r2" / "larsnet_models"
STEMS = ["kick", "snare", "toms", "hihat", "cymbals"]
STEM_GM = {"kick": 36, "snare": 38, "toms": 45, "hihat": 42, "cymbals": 49}
STEM_CLASS = {"kick": "kick", "snare": "snare", "toms": "tom", "hihat": "hihat", "cymbals": "cymbal"}
SR, HOP = 22050, 110  # 5 ms
DELTAS = [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.5]
OPEN, CLOSED, CRASH, RIDE = 46, 42, 49, 51


# ---------------------------------------------------------------- 분리 캐시

def separate_many(pairs: list[tuple[Path, Path]]) -> None:
    todo = [(str(i), str(o)) for i, o in pairs if not (Path(o) / "cymbals.wav").exists()]
    if not todo:
        return
    jf = SEP / "_jobs.json"
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(todo))
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore")
    subprocess.run(["nice", "-n", "15", str(ROOT / ".venv-adtstr/bin/python"), str(HERE / "larsnet_separate.py"),
                    "--jobs", str(jf)], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stem_dir(set_name: str, song: str) -> Path:
    return SEP / set_name / song


def load(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


# ---------------------------------------------------------------- 온셋

def onset_env(y: np.ndarray) -> np.ndarray:
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    m = np.percentile(env, 99.5) if len(env) else 1.0
    return env / (m if m > 0 else 1.0)


def onsets(env: np.ndarray, delta: float) -> np.ndarray:
    import librosa

    pk = librosa.util.peak_pick(env, pre_max=6, post_max=6, pre_avg=20, post_avg=20, delta=delta, wait=6)  # 30 ms
    return pk * HOP / SR


# ---------------------------------------------------------------- 세분 특징

def _band_env(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    return 10 * np.log10(S[(f >= lo) & (f < hi)].sum(axis=0) + 1e-10), S, f


def features(y: np.ndarray, times: np.ndarray) -> np.ndarray:
    """온셋마다 [감쇠 80-150ms, 150-300ms, 300-600ms (dB, 어택 대비), 중심주파수(kHz), 저고역비(dB), 다음 타격 간격(s)]."""
    hi_db, S, f = _band_env(y, 3000, 11000)
    mid_db = 10 * np.log10(S[(f >= 500) & (f < 3000)].sum(axis=0) + 1e-10)
    cen = (S * f[:, None]).sum(axis=0) / (S.sum(axis=0) + 1e-10) / 1000.0
    n = len(hi_db)
    fr = lambda t: int(np.clip(round(t * SR / HOP), 0, n - 1))  # noqa: E731
    seg = lambda a, b: hi_db[fr(a):max(fr(a) + 1, fr(b))]  # noqa: E731
    out = []
    ts = np.sort(times)
    for t in times:
        k = np.searchsorted(ts, t, side="right")
        ioi = min(1.0, ts[k] - t) if k < len(ts) else 1.0
        att = seg(t - 0.01, t + 0.03).max()
        d1, d2, d3 = (seg(t + a, t + b).mean() - att for a, b in ((0.08, 0.15), (0.15, 0.3), (0.3, 0.6)))
        a0 = fr(t)
        a1 = max(a0 + 1, fr(t + 0.05))
        out.append([d1, d2, d3, float(cen[a0:a1].mean()), float(mid_db[a0:a1].mean() - hi_db[a0:a1].mean()), ioi])
    return np.asarray(out, float).reshape(-1, 6)


def fit_split(train_songs: list[str]) -> dict:
    """Slakh train 정답 온셋으로 open/closed, crash/ride 로지스틱 회귀를 학습해 저장."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sets = {"hihat": ([], []), "cymbals": ([], [])}
    for s in train_songs:
        pm = pretty_midi.PrettyMIDI(str(DATA / "slakh_train" / s / "drums.mid"))
        notes = [(n.start, n.pitch) for i in pm.instruments for n in i.notes]
        for stem, pos, neg in (("hihat", {46}, {42, 44}), ("cymbals", {51, 53, 59}, {49, 52, 55, 57})):
            ts = np.array(sorted({round(t, 3) for t, p in notes if p in pos | neg}))
            if len(ts) == 0:
                continue
            lab = {round(t, 3): int(p in pos) for t, p in notes if p in pos | neg}
            y = load(stem_dir("slakh_train", s) / f"{stem}.wav")
            sets[stem][0].append(features(y, ts))
            sets[stem][1].append(np.array([lab[t] for t in ts]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    info = {}
    for stem, (X, Y) in sets.items():
        X, Y = np.vstack(X), np.concatenate(Y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X, Y)
        joblib.dump(clf, MODEL_DIR / f"split_{stem}.joblib")
        info[stem] = {"n": int(len(Y)), "pos": int(Y.sum()), "train_acc": float(clf.score(X, Y))}
    (MODEL_DIR / "split_info.json").write_text(json.dumps(info, indent=2))
    return info


def split_times(set_name: str, song: str, stem: str, times: np.ndarray) -> np.ndarray:
    """True = 오픈 하이햇 / 라이드."""
    import joblib

    if len(times) == 0:
        return np.zeros(0, bool)
    clf = joblib.load(MODEL_DIR / f"split_{stem}.joblib")
    y = load(stem_dir(set_name, song) / f"{stem}.wav")
    return clf.predict_proba(features(y, np.asarray(times)))[:, 1] >= 0.5


# ---------------------------------------------------------------- 시스템

def _write(out: Path, hits: list[tuple[float, int]]) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for t, p in sorted(hits):
        inst.notes.append(pretty_midi.Note(100, int(p), float(t), float(t) + 0.05))
    pm.instruments.append(inst)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))
    return out


def _deltas() -> dict:
    p = MODEL_DIR / "onset_deltas.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _adtof_on(audio: Path, out_mid: Path) -> dict[int, np.ndarray]:
    if not out_mid.exists():
        out_mid.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore", TF_CPP_MIN_LOG_LEVEL="3")
        subprocess.run(["nice", "-n", "15", str(ROOT / ".venv-adtof/bin/python"), str(ROOT / "backend/scripts/adtof_transcribe.py"),
                        str(audio), str(out_mid)], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pm = pretty_midi.PrettyMIDI(str(out_mid))
    by = {}
    for i in pm.instruments:
        for n in i.notes:
            by.setdefault(n.pitch, []).append(n.start)
    return {k: np.array(sorted(v)) for k, v in by.items()}


def transcribe(system: str, set_name: str, song: str, drums_stem: Path) -> Path:
    out = RUNS / system / "drums" / set_name / song / "drums.mid"
    if out.exists():
        return out
    d = stem_dir(set_name, song)
    separate_many([(drums_stem, d)])
    per: dict[str, np.ndarray] = {}
    if system == "larsnet_onset":
        deltas = _deltas()
        for stem in STEMS:
            per[stem] = onsets(onset_env(load(d / f"{stem}.wav")), deltas[stem])
    elif system == "larsnet_adtof":
        adtof_gm = {"kick": 36, "snare": 38, "toms": 45, "hihat": 42, "cymbals": 49}  # ADTOF 출력 번호(adtof_transcribe.py)
        for stem in STEMS:
            by = _adtof_on(d / f"{stem}.wav", RUNS / system / "drums" / set_name / song / f"adtof_{stem}.mid")
            per[stem] = by.get(adtof_gm[stem], np.zeros(0))
    elif system == "adtof_split":
        src = json.loads((RUNS / "stem_adtof" / "drums.json").read_text())[f"{set_name}/{song}"]["midi"]
        pm = pretty_midi.PrettyMIDI(str(ROOT / src))
        hits = [(n.start, n.pitch) for i in pm.instruments for n in i.notes]
        hh = np.array([t for t, p in hits if p == 42])
        cy = np.array([t for t, p in hits if p == 49])
        o, r = split_times(set_name, song, "hihat", hh), split_times(set_name, song, "cymbals", cy)
        rest = [(t, p) for t, p in hits if p not in (42, 49)]
        return _write(out, rest + [(t, OPEN if x else CLOSED) for t, x in zip(hh, o)]
                      + [(t, RIDE if x else CRASH) for t, x in zip(cy, r)])
    else:
        raise SystemExit(f"unknown system {system}")
    hits = [(t, STEM_GM[s]) for s in ("kick", "snare", "toms") for t in per[s]]
    o = split_times(set_name, song, "hihat", per["hihat"])
    r = split_times(set_name, song, "cymbals", per["cymbals"])
    hits += [(t, OPEN if x else CLOSED) for t, x in zip(per["hihat"], o)]
    hits += [(t, RIDE if x else CRASH) for t, x in zip(per["cymbals"], r)]
    return _write(out, hits)


def tune_deltas(train_songs: list[str]) -> dict:
    """스템별 온셋 delta: 그 스템 클래스(5클래스)의 정답 온셋 대비 F1 합계 최대."""
    import sys

    sys.path.insert(0, str(ROOT / "eval"))
    import score_drums
    from run_heldout import COLLAPSE5

    best = {}
    for stem in STEMS:
        tot = {dl: [0, 0, 0] for dl in DELTAS}
        for s in train_songs:
            pm = pretty_midi.PrettyMIDI(str(DATA / "slakh_train" / s / "drums.mid"))
            ref = np.array(sorted(n.start for i in pm.instruments for n in i.notes
                                  if COLLAPSE5.get(score_drums.CLASS_MAP.get(n.pitch)) == STEM_CLASS[stem]))
            env = onset_env(load(stem_dir("slakh_train", s) / f"{stem}.wav"))
            for dl in DELTAS:
                m, eo, ro = score_drums.match_onsets(onsets(env, dl), ref, 0.05)
                tot[dl] = [tot[dl][0] + m, tot[dl][1] + eo, tot[dl][2] + ro]
        f1 = {dl: score_drums.prf(*v)[2] for dl, v in tot.items()}
        best[stem] = max(f1, key=f1.get)
        print(stem, {k: round(v, 3) for k, v in f1.items()}, "→", best[stem], flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "onset_deltas.json").write_text(json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(HERE))
    import drums_r2

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["separate", "fit"])
    ap.add_argument("--sets", nargs="+", default=["slakh_train"])
    a = ap.parse_args()
    if a.cmd == "separate":
        for st in a.sets:
            songs = drums_r2.songs_of(st) if st != "slakh_train_all" else None
            if st == "slakh_train_all":
                st = "slakh_train"
                songs = sorted(p.name for p in (RUNS / "_sep" / st).iterdir() if (p / "stems.json").exists()
                               and (DATA / st / p.name / "drums.mid").exists())
            separate_many([(drums_r2.drum_stem(st, s), stem_dir(st, s)) for s in songs])
            print(st, len(songs), "songs separated", flush=True)
    else:
        songs = sorted(p.name for p in (RUNS / "_sep" / "slakh_train").iterdir() if (p / "stems.json").exists()
                       and (DATA / "slakh_train" / p.name / "drums.mid").exists())[:40]
        print(json.dumps(fit_split(songs)))
        tune_deltas(songs)
