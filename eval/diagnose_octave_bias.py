"""베이스 옥타브 오류가 일정한 +12반음 편향인지 곡별로 진단한다.

onset 매칭(pitch 무시, 50ms) 쌍의 pitch 차이(생성−정답) 분포를 낸다:
same / +12 / −12 / 기타 옥타브(±24…) / 비옥타브. 정답은 두 가지로 본다.
  - 원본 GT(`<곡>.mid`) — 익스포트 시 한 옥타브 낮게 저장됐다는 기록이 있는 라벨
  - 옥타브 보정 GT(`<곡>_octcorrected.mid`, 있는 곡만)

사용법:
    python eval/diagnose_octave_bias.py --tag baseline [--tag yourmt3] [--raw-basicpitch]
`--raw-basicpitch`는 baseline 런의 Basic Pitch 원출력(후처리 전)도 함께 진단한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_bench import BASS_SONGS, GT_DIR, ROOT, octave_error_breakdown
import score

COLS = ("same", "plus12", "minus12", "other_octave", "non_octave")


def row(est_mid: Path, gt_mid: Path) -> dict:
    return octave_error_breakdown(score.load_notes(gt_mid), score.load_notes(est_mid))


def fmt(d: dict) -> str:
    return " ".join(f"{d[c]*100:6.1f}%" for c in COLS) + f"  (n={d['matched']})"


def sources(tags: list[str], raw_bp: bool):
    for tag in tags:
        root = ROOT / "eval" / "runs" / tag / "bass"
        yield tag, lambda song, root=root: Path((root / song / "_midi_path.txt").read_text().strip())
    if raw_bp:
        root = ROOT / "eval" / "runs" / "baseline" / "bass"
        yield "basicpitch_raw", lambda song: next((root / song / "basic_pitch_bass_raw").glob("*.mid"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--raw-basicpitch", action="store_true")
    ap.add_argument("--json", type=Path, help="결과 저장 경로")
    args = ap.parse_args()

    out: dict = {}
    header = f"{'song':14s} {'GT':6s} " + " ".join(f"{c:>7s}" for c in ("same", "+12", "-12", "±24+", "non-oct"))
    for name, midi_of in sources(args.tag, args.raw_basicpitch):
        print(f"\n=== {name} ===")
        print(header)
        out[name] = {}
        for song in BASS_SONGS:
            est = midi_of(song)
            r_raw = row(est, GT_DIR / f"{song}.mid")
            out[name][song] = {"gt": r_raw}
            print(f"{song:14s} {'orig':6s} {fmt(r_raw)}")
            oct_gt = GT_DIR / f"{song}_octcorrected.mid"
            if oct_gt.exists():
                r_oct = row(est, oct_gt)
                out[name][song]["octcorrected_gt"] = r_oct
                print(f"{'':14s} {'octfix':6s} {fmt(r_oct)}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
