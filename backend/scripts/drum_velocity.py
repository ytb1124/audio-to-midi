"""드럼 MIDI의 velocity를 오디오 어택 에너지로 다시 매긴다.

ADTOF는 타격 시각과 class만 내고 velocity는 고정값(100)이다. E-GMD 분류기 경로
(`drum_audio_to_midi.py`)와 같은 방식 — 온셋 어택 RMS(`drum_features.attack_energy`)를
파일 내부 dB 다이나믹 레인지로 매핑(`rms_to_velocity`) — 을 그대로 적용해 두 엔진의
velocity 동작을 맞춘다. 노트 시각·pitch·길이는 건드리지 않는다.

사용법:
    python drum_velocity.py input.wav drums.mid [output.mid]   # output 생략 시 덮어쓰기
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drum_features as df  # noqa: E402
from drum_audio_to_midi import rms_to_velocity  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("midi", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    args = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(str(args.midi))
    notes = [n for inst in pm.instruments if inst.is_drum for n in inst.notes]
    if not notes:
        print(json.dumps({"notes": 0}))
        return 0
    y, sr = librosa.load(str(args.audio), sr=df.SR, mono=True)
    vels = rms_to_velocity(df.attack_energy(y, np.array([n.start for n in notes]), sr))
    for n, v in zip(notes, vels):
        n.velocity = int(v)
    pm.write(str(args.output or args.midi))
    print(json.dumps({"notes": len(notes), "velocity_std": round(float(np.std(vels)), 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
