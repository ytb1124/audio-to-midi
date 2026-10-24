"""비트·다운비트 추적(격리 venv .venv-beat에서 실행). beat_grid.py가 subprocess로 부른다.

기본은 Beat This!(CPJKU, final0, DBN 없음) — eval/heldout/beats_eval.py 홀드아웃 결과로 선택.
출력: 마지막 줄 JSON {"beats": [...], "downbeats": [...]}

사용법: .venv-beat/bin/python beat_track.py <audio> [--tracker beat_this|madmom]
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")


def beat_this(path: str):
    from beat_this.inference import File2Beats

    beats, downbeats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(path)
    return [float(x) for x in beats], [float(x) for x in downbeats]


def madmom(path: str):
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    arr = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(RNNDownBeatProcessor()(path))
    beats = [float(x) for x in DBNBeatTrackingProcessor(fps=100)(RNNBeatProcessor()(path))]
    return beats, [float(t) for t, k in arr if int(k) == 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--tracker", default="beat_this", choices=("beat_this", "madmom"))
    a = ap.parse_args()
    beats, downbeats = {"beat_this": beat_this, "madmom": madmom}[a.tracker](a.audio)
    print(json.dumps({"beats": beats, "downbeats": downbeats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
