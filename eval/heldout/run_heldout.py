"""홀드아웃 실제 믹스 벤치마크 — 분리 → 전사 경로로 세 시스템을 같은 조건에서 잰다.

fetch_heldout.py가 만든 eval/heldout/data/ 를 입력으로 쓴다. 지표 정의는 eval/score.py,
eval/score_drums.py의 함수를 그대로 재사용한다.

시스템(이름 = 입력_엔진):
  bass  (Slakh)        stem_basicpitch   현행 기본값: htdemucs bass 스템 → Basic Pitch + 하이브리드 후처리
                       stem_yourmt3      htdemucs bass 스템 → YourMT3+ (단성 선택)
                       stem_yourmt3_hyb  htdemucs bass 스템 → YourMT3+ → 하이브리드 후처리
                       mix_yourmt3       믹스 그대로 → YourMT3+ 전체 악기 → 베이스 프로그램(32~39)만
  drums (Slakh + MDB)  stem_egmd         현행 기본값: htdemucs drums 스템 → E-GMD 분류기
                       stem_adtof        htdemucs drums 스템 → ADTOF
                       mix_egmd / mix_adtof / mix_yourmt3   믹스 그대로

사용법:
    .venv-backend/bin/python eval/heldout/run_heldout.py run [--systems ...] [--sets slakh mdb] [--songs ...]
    .venv-backend/bin/python eval/heldout/run_heldout.py report      # RESULTS_HELDOUT.md 표 출력
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / "data"
RUNS = HERE / "runs"
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

import score  # noqa: E402
import score_drums  # noqa: E402

BASS_SYSTEMS = ["stem_basicpitch", "stem_yourmt3", "stem_yourmt3_hyb", "mix_yourmt3"]
DRUM_SYSTEMS = ["stem_egmd", "stem_adtof", "mix_egmd", "mix_adtof", "mix_yourmt3"]
BASELINE = {"bass": "stem_basicpitch", "drums": "stem_egmd"}
BASS_PROGRAMS = range(32, 40)

# 구간을 잘라 만든 Slakh 정답은 창 시작 전에 시작된 노트를 버리므로, 양쪽 모두 첫 0.1초 온셋은 채점에서 뺀다.
IGNORE_HEAD = 0.1
OFFSET_RATIO = 0.2

# 세 시스템이 모두 표현할 수 있는 공통 5클래스(ADTOF 체계). 9클래스 CLASS_MAP을 접는다.
COLLAPSE5 = {
    "kick": "kick", "snare": "snare", "tom": "tom",
    "hihat_closed": "hihat", "hihat_open": "hihat",
    "crash": "cymbal", "ride": "cymbal",
}
CLASSES5 = ["kick", "snare", "hihat", "tom", "cymbal"]


# ---------------------------------------------------------------- 실행

def isolated_py(name: str) -> str:
    py = ROOT / name / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"{name} venv가 없다: backend/scripts/setup_isolated_models.sh 먼저 실행")
    return str(py)


def separate(mix: Path, set_name: str, song: str) -> dict[str, Path]:
    out = RUNS / "_sep" / set_name / song
    marker = out / "stems.json"
    if marker.exists():
        return {k: Path(v) for k, v in json.loads(marker.read_text()).items()}
    out.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("DEMUCS_DEVICE", "mps")
    cp = subprocess.run(
        [sys.executable, str(ROOT / "backend/scripts/demucs_separate.py"), str(mix), str(out),
         "--model", "htdemucs", "--device", device],
        check=True, capture_output=True, text=True,
    )
    info = json.loads([ln for ln in cp.stdout.splitlines() if ln.strip().startswith("{")][-1])
    stems = {k: str(out / v) for k, v in info["stems"].items()}
    marker.write_text(json.dumps(stems))
    return {k: Path(v) for k, v in stems.items()}


def yourmt3_all(audio: Path, out_dir: Path) -> Path:
    """YourMT3+ 전체 악기 전사(믹스 입력). 결과를 캐시해 베이스·드럼 양쪽에서 재사용한다."""
    out = out_dir / "yourmt3_all.mid"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [isolated_py(".venv-yourmt3"), str(ROOT / "backend/scripts/yourmt3_transcribe.py"),
             str(audio), str(out), "--instrument", "all"],
            check=True, capture_output=True, text=True,
        )
    return out


def filter_midi(src: Path, dest: Path, keep) -> Path:
    pm = pretty_midi.PrettyMIDI(str(src))
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        if keep(inst):
            out.instruments.append(inst)
    out.write(str(dest))
    return dest


def run_system(kind: str, system: str, set_name: str, song: str) -> Path:
    from backend import pipeline

    src = DATA / set_name / song
    mix = src / "mix.wav"
    out_dir = RUNS / system / kind / set_name / song
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = "bass" if kind == "bass" else "drums"

    if system.startswith("stem_"):
        audio = separate(mix, set_name, song)[stem_name]
    else:
        audio = mix

    if system == "mix_yourmt3":
        allm = yourmt3_all(mix, RUNS / "_yourmt3_mix" / set_name / song)
        if kind == "bass":
            return filter_midi(allm, out_dir / "bass.mid", lambda i: not i.is_drum and i.program in BASS_PROGRAMS)
        return filter_midi(allm, out_dir / "drums.mid", lambda i: i.is_drum)

    if kind == "bass":
        backend = {"stem_basicpitch": "basicpitch", "stem_yourmt3": "yourmt3", "stem_yourmt3_hyb": "yourmt3_hybrid"}[system]
        res = pipeline.run_bass_mode(audio, out_dir, backend=backend)
        if res.get("backend") != backend:
            raise RuntimeError(f"{system}: {backend} 대신 {res.get('backend')}로 폴백됨")
        return Path(res["midiPath"])

    backend = system.split("_", 1)[1]  # egmd | adtof
    res = pipeline.run_drums_mode(audio, out_dir, separate=False, backend=backend)
    if backend == "adtof" and res.get("backend") != "adtof":
        raise RuntimeError(f"{system}: ADTOF 대신 E-GMD로 폴백됨")
    return Path(res["midiPath"])


# ---------------------------------------------------------------- 채점

def _notes(path: Path, drums: bool | None = None):
    pm = pretty_midi.PrettyMIDI(str(path))
    out = []
    for inst in pm.instruments:
        if drums is not None and inst.is_drum != drums:
            continue
        out += [(n.start, n.end, n.pitch) for n in inst.notes if n.start >= IGNORE_HEAD]
    return sorted(out)


def _f1_with_offset(ref, est):
    import mir_eval

    ri = np.array([[a, b] for a, b, _ in ref]) if ref else np.zeros((0, 2))
    ei = np.array([[a, b] for a, b, _ in est]) if est else np.zeros((0, 2))
    rp = np.array([score.midi_to_hz(p) for *_, p in ref])
    ep = np.array([score.midi_to_hz(p) for *_, p in est])
    return mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=score.ONSET_TOLERANCE_SEC,
        pitch_tolerance=score.PITCH_TOLERANCE_CENTS, offset_ratio=OFFSET_RATIO)[2]


def _chroma(notes):
    return [(a, b, 60 + p % 12) for a, b, p in notes]


def score_bass(est_path: Path, gt_path: Path) -> dict:
    """대표값 f1은 옥타브 무관(pitch class) onset F1이다.

    Slakh 베이스 패치는 음역 아래 노트를 한 옥타브 올려 렌더링한다: Track01876 실측에서 정답
    pitch<35(B1)인 노트를 Basic Pitch·YourMT3+ 모두 정확히 +12로 듣고, 35 이상은 일치했다.
    즉 Slakh 정답의 옥타브는 오디오와 어긋날 수 있어 옥타브 정확도는 여기서 잴 수 없다
    (옥타브는 eval/ground_truth 개발셋에서 잰다). 엄격 F1(f1_strict)도 함께 남긴다.
    """
    ref = _notes(gt_path)
    est = _notes(est_path, drums=False)
    p, r, f1_strict, _ = score.mir_eval_metrics(ref, est)
    pc, rc, f1_pc, _ = score.mir_eval_metrics(_chroma(ref), _chroma(est))
    pairs = _onset_pairs(ref, est)
    n = len(pairs)
    return {
        "f1": f1_pc, "precision": pc, "recall": rc,
        "f1_offset": _f1_with_offset(_chroma(ref), _chroma(est)),
        "f1_strict": f1_strict, "precision_strict": p, "recall_strict": r,
        # 피치 무시 온셋 매칭 쌍 기준. 옥타브 차이는 Slakh 렌더링 접힘을 포함하므로 진단용.
        "octave_diff_rate": sum(1 for a, b in pairs if a != b and (a - b) % 12 == 0) / n if n else 0.0,
        "pitch_class_err_rate": sum(1 for a, b in pairs if (a - b) % 12 != 0) / n if n else 0.0,
        "n_ref": len(ref), "n_est": len(est),
    }


def _onset_pairs(ref, est, tol=score.ONSET_TOLERANCE_SEC):
    est_t = np.array([e[0] for e in est])
    used = np.zeros(len(est), dtype=bool)
    pairs = []
    for on, _off, pitch in ref:
        lo, hi = np.searchsorted(est_t, on - tol), np.searchsorted(est_t, on + tol)
        cand = [i for i in range(lo, hi) if not used[i]]
        if not cand:
            continue
        i = min(cand, key=lambda k: abs(est_t[k] - on))
        used[i] = True
        pairs.append((pitch, est[i][2]))
    return pairs


def _drum_by5(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    by = {c: [] for c in CLASSES5}
    unmapped = {}
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            if n.start < IGNORE_HEAD:
                continue
            c9 = score_drums.CLASS_MAP.get(n.pitch)
            c5 = COLLAPSE5.get(c9) if c9 else None
            if c5 is None:
                unmapped[n.pitch] = unmapped.get(n.pitch, 0) + 1
                continue
            by[c5].append(n.start)
    return {c: np.array(sorted(v)) for c, v in by.items()}, unmapped


def score_drum(est_path: Path, gt_path: Path, tol: float = 0.05) -> dict:
    est, est_un = _drum_by5(est_path)
    ref, ref_un = _drum_by5(gt_path)
    tm = te = tr = 0
    per = {}
    for c in CLASSES5:
        m, eo, ro = score_drums.match_onsets(est[c], ref[c], tol)
        tm, te, tr = tm + m, te + eo, tr + ro
        if len(ref[c]) or len(est[c]):
            per[c] = {"f1": score_drums.prf(m, eo, ro)[2], "n_ref": int(len(ref[c])), "n_est": int(len(est[c]))}
    all_e = np.sort(np.concatenate([est[c] for c in CLASSES5]))
    all_r = np.sort(np.concatenate([ref[c] for c in CLASSES5]))
    return {
        "f1": score_drums.prf(tm, te, tr)[2],
        "f1_onset_only": score_drums.prf(*score_drums.match_onsets(all_e, all_r, tol))[2],
        "per_class": per, "n_ref": int(tr + tm), "n_est": int(te + tm),
        "unmapped_est": est_un, "unmapped_ref": ref_un,
    }


# ---------------------------------------------------------------- 집계

def bootstrap_ci(values, n=10000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_summary(kind: str, system: str) -> dict:
    p = RUNS / system / f"{kind}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cmd_run(args) -> int:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for kind, systems in (("bass", BASS_SYSTEMS), ("drums", DRUM_SYSTEMS)):
        if args.kind and kind != args.kind:
            continue
        sets = ["slakh"] if kind == "bass" else ["slakh", "mdb"]
        for system in systems:
            if args.systems and system not in args.systems:
                continue
            summary_path = RUNS / system / f"{kind}.json"
            summary = load_summary(kind, system)
            for set_name in sets:
                if args.sets and set_name not in args.sets:
                    continue
                for item in manifest[set_name]:
                    song = item["song"]
                    if args.songs and song not in args.songs:
                        continue
                    key = f"{set_name}/{song}"
                    if key in summary and not args.force:
                        continue
                    gt = DATA / set_name / song / ("bass.mid" if kind == "bass" else "drums.mid")
                    t0 = time.time()
                    try:
                        midi = run_system(kind, system, set_name, song)
                        res = score_bass(midi, gt) if kind == "bass" else score_drum(midi, gt)
                    except Exception as e:  # noqa: BLE001 — 한 곡 실패가 전체를 멈추지 않게 기록만
                        res = {"error": f"{type(e).__name__}: {e}"[:500]}
                    res["seconds"] = round(time.time() - t0, 1)
                    summary[key] = res
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
                    f1 = res.get("f1")
                    print(f"{kind:5s} {system:17s} {key:28s} F1={f1:.4f} ({res['seconds']}s)" if f1 is not None
                          else f"{kind:5s} {system:17s} {key:28s} ERROR {res['error']}", flush=True)
    return 0


def cmd_report(args) -> int:
    manifest = json.loads((DATA / "manifest.json").read_text())
    lines = []
    for kind, systems, sets in (("bass", BASS_SYSTEMS, ["slakh"]), ("drums", DRUM_SYSTEMS, ["slakh", "mdb"])):
        for set_name in sets:
            songs = [f"{set_name}/{i['song']}" for i in manifest[set_name]]
            base = load_summary(kind, BASELINE[kind])
            lines.append(f"\n### {kind} — {set_name} ({len(songs)}곡)\n")
            extra = ["f1_offset", "f1_strict", "pitch_class_err_rate"] if kind == "bass" else ["f1_onset_only"]
            lines.append("| 시스템 | 평균 F1 [95% CI] | 기준선 대비 차이 [95% CI] | " + " | ".join(extra) + " | 실패 |")
            lines.append("|---|---|---|" + "---|" * len(extra) + "---|")
            for system in systems:
                s = load_summary(kind, system)
                ok = [k for k in songs if k in s and "f1" in s[k]]
                fails = [k for k in songs if k in s and "error" in s[k]]
                if not ok:
                    continue
                f = [s[k]["f1"] for k in ok]
                lo, hi = bootstrap_ci(f)
                paired = [s[k]["f1"] - base[k]["f1"] for k in ok if k in base and "f1" in base[k]]
                dlo, dhi = bootstrap_ci(paired)
                diff = "—" if system == BASELINE[kind] else f"{np.mean(paired):+.4f} [{dlo:+.3f}, {dhi:+.3f}]"
                ex = " | ".join(f"{np.mean([s[k][e] for k in ok]):.4f}" for e in extra)
                name = f"**{system}** (비교 기준: 2026-10-24 이전 기본값)" if system == BASELINE[kind] else system
                lines.append(f"| {name} | {np.mean(f):.4f} [{lo:.3f}, {hi:.3f}] (n={len(ok)}) | {diff} | {ex} | {len(fails)} |")
            if kind == "drums":
                lines.append("\n클래스별 평균 F1 (곡별 F1 평균, 해당 클래스가 정답에 있는 곡만):\n")
                lines.append("| 시스템 | " + " | ".join(CLASSES5) + " |")
                lines.append("|---|" + "---|" * len(CLASSES5))
                for system in systems:
                    s = load_summary(kind, system)
                    ok = [k for k in songs if k in s and "f1" in s[k]]
                    if not ok:
                        continue
                    cells = []
                    for c in CLASSES5:
                        v = [s[k]["per_class"][c]["f1"] for k in ok if c in s[k]["per_class"] and s[k]["per_class"][c]["n_ref"] > 0]
                        cells.append(f"{np.mean(v):.3f}" if v else "—")
                    lines.append(f"| {system} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=("bass", "drums"))
    r.add_argument("--systems", nargs="*")
    r.add_argument("--sets", nargs="*")
    r.add_argument("--songs", nargs="*")
    r.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
