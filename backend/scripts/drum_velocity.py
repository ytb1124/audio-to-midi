"""드럼 MIDI의 velocity를 오디오 어택 에너지로 다시 매긴다.

ADTOF는 타격 시각과 class만 내고 velocity는 고정값(100)이다. E-GMD 분류기 경로
(`drum_audio_to_midi.py`)와 같은 방식 — 온셋 어택 RMS(`drum_features.attack_energy`)를
파일 내부 dB 다이나믹 레인지로 매핑(`rms_to_velocity`) — 을 그대로 적용해 두 엔진의
velocity 동작을 맞춘다. 노트 시각·pitch·길이는 건드리지 않는다.

모드(2026-10-25, R6):
- attack: 위 광대역 어택 RMS(이전 방식)
- band: 노트 클래스(GM 번호)별 대역(킥 30–150 Hz, 스네어 150–4 kHz, 하이햇·심벌 5–16 kHz,
  톰 60–500 Hz)의 STFT 에너지 50 ms 최대. 광대역 어택은 킥과 동시에 친 하이햇이 킥 에너지를
  그대로 받아 버린다. Slakh train 개발셋 곡별 Spearman ρ 0.461 → 0.701(+0.240 [+0.185, +0.293]).
  홀드아웃 판정은 eval/heldout/velocity_eval.py.

사용법:
    python drum_velocity.py input.wav drums.mid [output.mid] [--mode band]   # output 생략 시 덮어쓰기
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


BANDS = {"kick": (30, 150), "snare": (150, 4000), "hihat": (5000, 16000), "cymbal": (5000, 16000), "tom": (60, 500)}
GM_CLASS = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare",
            42: "hihat", 44: "hihat", 46: "hihat", 41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
            49: "cymbal", 51: "cymbal", 52: "cymbal", 53: "cymbal", 55: "cymbal", 57: "cymbal", 59: "cymbal"}
DEFAULT_MODE = "band"  # R6 홀드아웃(Slakh test 16곡) ρ 0.330 → 0.620, paired +0.290 [+0.152, +0.429]


def band_energy(y: np.ndarray, sr: int, notes) -> np.ndarray:
    """노트마다 자기 클래스 대역의 STFT 에너지 [t, t+50ms] 최대(선형). 매핑에 없는 pitch는 광대역."""
    hop = 128
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    tt = np.arange(S.shape[1]) * hop / sr
    out = []
    for n in notes:
        lo, hi = BANDS.get(GM_CLASS.get(n.pitch, ""), (30, 16000))
        fm = (freqs >= lo) & (freqs < hi)
        m = (tt >= n.start) & (tt < n.start + 0.05)
        out.append(float(S[fm][:, m].sum(axis=0).max()) if m.any() else 0.0)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("midi", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--mode", choices=("attack", "band"), default=DEFAULT_MODE)
    args = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(str(args.midi))
    notes = [n for inst in pm.instruments if inst.is_drum for n in inst.notes]
    if not notes:
        print(json.dumps({"notes": 0}))
        return 0
    if args.mode == "band":
        y, sr = librosa.load(str(args.audio), sr=22050, mono=True)
        vels = rms_to_velocity(band_energy(y, sr, notes))
    else:
        y, sr = librosa.load(str(args.audio), sr=df.SR, mono=True)
        vels = rms_to_velocity(df.attack_energy(y, np.array([n.start for n in notes]), sr))
    for n, v in zip(notes, vels):
        n.velocity = int(v)
    pm.write(str(args.output or args.midi))
    print(json.dumps({"notes": len(notes), "velocity_std": round(float(np.std(vels)), 1), "mode": args.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
