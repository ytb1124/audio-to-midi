"""피치 노트 MIDI의 벨로시티를 오디오 어택 에너지로 다시 매긴다 (2026-10-25, R6).

YourMT3+는 벨로시티를 상수로 낸다(홀드아웃 Slakh 정답 대비 Spearman ρ = 0). 베이스 하이브리드
(`bass_fragment_merge_v2.py`)와 같은 방식 — 노트 시작 [t, t+120 ms] RMS 최대를 파일 내 dB 범위
(p10~p95, 최소 6 dB)로 52~104에 매핑 — 을 적용한다. Slakh train 개발셋에서 이 특징량이 대안
(증가량·피치 대역·플럭스·피치 보정)보다 같거나 나았다(eval/heldout/velocity_features.py).
노트 시각·피치·길이는 바꾸지 않는다.

사용법: python note_velocity.py input.wav notes.mid [output.mid]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

HOP = 256


def map_velocities(audio: str, notes) -> np.ndarray:
    y, sr = librosa.load(audio, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    t = np.arange(len(rms)) * HOP / sr
    peak = []
    for n in notes:
        m = (t >= n.start) & (t < n.start + 0.12)
        peak.append(float(rms[m].max()) if m.any() else 1e-6)
    db = 20 * np.log10(np.maximum(np.array(peak), 1e-6))
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    norm = np.clip((db - lo) / max(hi - lo, 6.0), 0.0, 1.0)
    return np.round(52 + 52 * norm).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi")
    ap.add_argument("output", nargs="?")
    a = ap.parse_args()
    pm = pretty_midi.PrettyMIDI(a.midi)
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if len(notes) >= 8:
        for n, v in zip(notes, map_velocities(a.audio, notes)):
            n.velocity = int(v)
    pm.write(a.output or a.midi)
    print(json.dumps({"notes": len(notes), "velocity_std": float(np.std([n.velocity for n in notes])) if notes else 0.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
