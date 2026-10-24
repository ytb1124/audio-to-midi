"""베이스 6곡 / 드럼 8곡 일괄 전사 + 채점 하니스.

채점은 `eval/score.py`, `eval/score_drums.py`의 함수를 그대로 재사용한다(지표 정의를
이 파일에서 새로 만들지 않기 위함). 각 곡의 원본 CLI 출력도 로그로 남긴다.

사용법:
    python eval/run_bench.py bass  --backend basicpitch --tag baseline
    python eval/run_bench.py drums --backend egmd       --tag baseline
    python eval/run_bench.py bass  --backend yourmt3    --tag yourmt3

결과: eval/runs/<tag>/<kind>/<song>/ (MIDI·로그), eval/runs/<tag>/<kind>.json (요약)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "eval" / "ground_truth"
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

import score  # noqa: E402
import score_drums  # noqa: E402

BASS_SONGS = ["bass_sample", "bass_sample2", "bass_sample3", "bass_sample4", "bass_sample5", "bass_test"]
DRUM_SONGS = [f"drums_sample{i}" for i in (4, 5, 6, 7, 8, 9, 10, 11)]


def octave_error_breakdown(ref_notes, est_notes, tol=score.ONSET_TOLERANCE_SEC):
    """onset 매칭(pitch 무시) 쌍의 pitch 차이 분포. ±12 편향 진단용."""
    unused = list(range(len(est_notes)))
    diffs = []
    for r_on, _r_off, r_pitch in ref_notes:
        best, best_d = None, None
        for i in unused:
            d = abs(est_notes[i][0] - r_on)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            continue
        unused.remove(best)
        diffs.append(est_notes[best][2] - r_pitch)
    n = len(diffs)
    c = Counter(diffs)
    frac = lambda k: (c.get(k, 0) / n) if n else 0.0  # noqa: E731
    return {
        "matched": n,
        "same": frac(0),
        "plus12": frac(12),
        "minus12": frac(-12),
        "other_octave": (sum(v for k, v in c.items() if k % 12 == 0 and k not in (0, 12, -12)) / n) if n else 0.0,
        "non_octave": (sum(v for k, v in c.items() if k % 12 != 0) / n) if n else 0.0,
    }


def score_bass(est_path: Path, gt_path: Path) -> dict:
    ref = score.load_notes(gt_path)
    est = score.load_notes(est_path)
    _p, _r, f1_raw, _ = score.mir_eval_metrics(ref, est)
    shift, shift_frac, _n = score.detect_global_octave_shift(ref, est)
    if shift is not None:
        _p2, _r2, f1_oct, _ = score.mir_eval_metrics(score.shift_notes(ref, shift), est)
    else:
        f1_oct = f1_raw
    # score.py와 동일: 전역 시간 오프셋(|lag|>15ms)이 감지되면 시간+옥타브 보정 재채점.
    lag = score.detect_global_time_offset(ref, est)
    f1_time = None
    if abs(lag) > 0.015:
        # 음수 lag로 첫 정답 노트가 0초 앞으로 밀리면 mir_eval이 거부한다(score.py CLI도
        # 같은 지점에서 예외). 시작/끝을 0으로 클램프해 채점만 가능하게 한다.
        ref_t = [(max(0.0, a), max(1e-3, b), p) for a, b, p in score.time_shift_notes(ref, lag)]
        if shift is not None:
            ref_t = score.shift_notes(ref_t, shift)
        f1_time = score.mir_eval_metrics(ref_t, est)[2]
    out = {
        "f1_raw": f1_raw,
        "global_shift": shift,
        "global_shift_frac": shift_frac,
        "f1_octave": f1_oct,  # score.py가 "실질적 채점 기준"으로 내는 옥타브 보정 F1
        "time_lag": lag,
        "f1_time_octave": f1_time,
        # 대표값: score.py가 출력하는 마지막 보정 점수(시간 보정이 있으면 그것, 없으면 옥타브 보정)
        "f1": f1_time if f1_time is not None else f1_oct,
        "n_ref": len(ref),
        "n_est": len(est),
        "octave_vs_gt": octave_error_breakdown(ref, est),
    }
    oct_gt = GT_DIR / f"{gt_path.stem}_octcorrected.mid"
    if oct_gt.exists():
        ref_c = score.load_notes(oct_gt)
        _p3, _r3, f1_c, _ = score.mir_eval_metrics(ref_c, est)
        out["f1_vs_octcorrected_gt"] = f1_c
        out["octave_vs_octcorrected_gt"] = octave_error_breakdown(ref_c, est)
    return out


def score_drum(est_path: Path, gt_path: Path, tol: float = 0.05) -> dict:
    est_by, _ = score_drums.load_by_class(est_path)
    ref_by, _ = score_drums.load_by_class(gt_path)
    per_class = {}
    tm = te = tr = 0
    for cls in score_drums.CLASS_ORDER:
        e = est_by.get(cls, [])
        r = ref_by.get(cls, [])
        if len(e) == 0 and len(r) == 0:
            continue
        import numpy as np

        m, eo, ro = score_drums.match_onsets(np.asarray(e), np.asarray(r), tol)
        tm += m; te += eo; tr += ro
        per_class[cls] = score_drums.prf(m, eo, ro)[2]
    import numpy as np

    _p, _r, f_micro = score_drums.prf(tm, te, tr)
    est_all = np.array(sorted(t for v in est_by.values() for t in v))
    ref_all = np.array(sorted(t for v in ref_by.values() for t in v))
    f_onset = score_drums.prf(*score_drums.match_onsets(est_all, ref_all, tol))[2]
    return {"f1": f_micro, "f1_onset_only": f_onset, "per_class": per_class}


def transcribe(kind: str, backend: str, wav: Path, out_dir: Path) -> Path:
    if backend == "egmd_loso":
        # 공정 비교용 참조선: 배포 분류기는 sample10 외 7곡으로 학습돼 in-sample이므로,
        # 해당 곡을 뺀 LOSO 폴드 모델(eval/_loso_models)로 같은 스크립트를 돌린다.
        loso = ROOT / "eval" / "_loso_models" / f"without_{wav.stem}.joblib"
        out = out_dir / "drums.mid"
        cmd = [sys.executable, str(ROOT / "backend" / "scripts" / "drum_audio_to_midi.py"), str(wav), str(out)]
        if loso.exists():  # sample10은 학습에서 제외된 곡이라 배포 모델이 곧 out-of-sample
            cmd += ["--model", str(loso)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out

    if backend == "yourmt3_hybrid" and (ROOT / "eval/runs/yourmt3/bass" / wav.stem / "bass_yourmt3.mid").exists():
        # 파이프라인의 yourmt3_hybrid와 같은 호출(YourMT3+ 출력 → 베이스 후처리 --use-given-raw)이되,
        # 이미 만든 yourmt3 런의 MIDI를 재사용해 재추론(곡당 1~4분)을 생략한다.
        raw = ROOT / "eval/runs/yourmt3/bass" / wav.stem / "bass_yourmt3.mid"
        out = out_dir / "bass_hybrid.mid"
        subprocess.run(
            [sys.executable, str(ROOT / "backend/scripts/bass_basicpitch_audio_hybrid.py"),
             str(wav), str(raw), str(out), "--use-given-raw"],
            check=True, capture_output=True, text=True, cwd=str(out_dir),
        )
        return out

    from backend import pipeline

    mode = "bass" if kind == "bass" else "drums"
    result = pipeline.transcribe(mode=mode, input_audio=wav, output_dir=out_dir, backend=backend)
    return Path(result["midiPath"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("bass", "drums"))
    ap.add_argument("--backend", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--songs", nargs="*")
    ap.add_argument("--rescore", action="store_true", help="기존 MIDI로 채점만 다시")
    args = ap.parse_args()

    songs = args.songs or (BASS_SONGS if args.kind == "bass" else DRUM_SONGS)
    run_root = ROOT / "eval" / "runs" / args.tag / args.kind
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root.parent / f"{args.kind}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    scorer_cli = "score.py" if args.kind == "bass" else "score_drums.py"
    for song in songs:
        wav = GT_DIR / f"{song}.wav"
        gt = GT_DIR / f"{song}.mid"
        out_dir = run_root / song
        midi_marker = out_dir / "_midi_path.txt"
        t0 = time.time()
        if args.rescore and midi_marker.exists():
            midi = Path(midi_marker.read_text().strip())
            elapsed = summary.get(song, {}).get("seconds")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            midi = transcribe(args.kind, args.backend, wav, out_dir)
            midi_marker.write_text(str(midi))
            elapsed = round(time.time() - t0, 1)
        res = score_bass(midi, gt) if args.kind == "bass" else score_drum(midi, gt)
        res["seconds"] = elapsed
        res["backend"] = args.backend
        summary[song] = res
        log = subprocess.run(
            [sys.executable, str(ROOT / "eval" / scorer_cli), str(midi), str(gt)],
            capture_output=True, text=True,
        )
        (out_dir / f"{scorer_cli}.log").write_text(log.stdout + log.stderr)
        print(f"{song:16s} F1={res['f1']:.4f}  ({elapsed}s)", flush=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    f1s = [summary[s]["f1"] for s in songs if s in summary]
    print(f"{'MEAN':16s} F1={sum(f1s)/len(f1s):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
