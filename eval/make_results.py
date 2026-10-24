"""eval/runs/<tag>/{bass,drums}.json → 곡별·평균 비교표(markdown)를 출력한다.

사용법:
    python eval/make_results.py --bass baseline yourmt3 yourmt3_hybrid --drums baseline egmd_loso adtof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_bench import BASS_SONGS, DRUM_SONGS, ROOT


def load(tag: str, kind: str) -> dict:
    path = ROOT / "eval" / "runs" / tag / f"{kind}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def table(kind: str, tags: list[str], songs: list[str], key: str = "f1") -> str:
    data = {t: load(t, kind) for t in tags}
    lines = ["| 곡 | " + " | ".join(tags) + " |", "|---|" + "---:|" * len(tags)]
    sums = {t: [] for t in tags}
    for s in songs:
        cells = []
        for t in tags:
            v = data[t].get(s, {}).get(key)
            if v is None:
                cells.append("—")
            else:
                sums[t].append(v)
                cells.append(f"{v:.4f}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    means = []
    for t in tags:
        vals = sums[t]
        means.append(f"**{sum(vals)/len(vals):.4f}**" if len(vals) == len(songs) else "—")
    lines.append("| **평균** | " + " | ".join(means) + " |")
    return "\n".join(lines)


def mean(tag: str, kind: str, songs: list[str]) -> float | None:
    d = load(tag, kind)
    vals = [d[s]["f1"] for s in songs if s in d]
    return sum(vals) / len(vals) if len(vals) == len(songs) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bass", nargs="+", default=[])
    ap.add_argument("--drums", nargs="+", default=[])
    ap.add_argument("--bass-key", default="f1")
    args = ap.parse_args()
    if args.bass:
        print(table("bass", args.bass, BASS_SONGS, args.bass_key))
        print()
    if args.drums:
        print(table("drums", args.drums, DRUM_SONGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
