#!/usr/bin/env python3
"""정답 MIDI가 없는 실전 오디오 파일용 진단 도구.

생성된 MIDI의 각 노트를 오디오 자체의 pYIN 피치 트랙과 대조해서:
  - pyin_agree      : pYIN 중앙값이 노트 pitch와 0.6반음 이내
  - octave_off      : pYIN과의 거리가 옥타브 배수(11.4~12.6, 23.4~24.6)
  - pitch_off       : 그 외 1반음 이상 어긋남 (진짜 피치 오류 후보)
  - unvoiced        : 노트 구간에서 pYIN voiced 프레임이 30% 미만 (유령 노트 후보)
로 분류하고, 문제 노트 목록과 시간대별 요약을 출력한다.

Usage:
    python eval/diagnose_audio_vs_midi.py <audio.wav> <generated.mid> [--csv out.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import librosa
    import pretty_midi

    print(f"Loading audio: {args.audio}", file=sys.stderr)
    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    print("Running pYIN (C1~C5)...", file=sys.stderr)
    hop = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C1"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        hop_length=hop,
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi_track = librosa.hz_to_midi(f0)

    pm = pretty_midi.PrettyMIDI(args.midi)
    notes = sorted(
        [n for i in pm.instruments if not i.is_drum for n in i.notes],
        key=lambda n: n.start,
    )
    print(f"Generated notes: {len(notes)}", file=sys.stderr)

    rows = []
    for n in notes:
        # 어택 직후 안정 구간을 보기 위해 노트 앞 20%(최대 30ms)는 건너뜀
        skip = min(0.03, (n.end - n.start) * 0.2)
        mask = (times >= n.start + skip) & (times <= n.end)
        seg = midi_track[mask]
        vprob = voiced_prob[mask]
        voiced = seg[~np.isnan(seg)]

        voiced_frac = len(voiced) / max(len(seg), 1)

        if voiced_frac < 0.3 or len(voiced) == 0:
            cat = "unvoiced"
            pyin_med = np.nan
            dist = np.nan
        else:
            pyin_med = float(np.median(voiced))
            dist = pyin_med - n.pitch
            adist = abs(dist)
            if adist <= 0.6:
                cat = "pyin_agree"
            elif 11.4 <= adist <= 12.6 or 23.4 <= adist <= 24.6:
                cat = "octave_off"
            elif adist >= 1.0:
                cat = "pitch_off"
            else:
                cat = "pyin_agree"  # 0.6~1.0 사이는 벤딩/비브라토 여지로 관대하게

        rows.append(
            {
                "start": n.start,
                "end": n.end,
                "dur": n.end - n.start,
                "pitch": n.pitch,
                "pyin_median": pyin_med,
                "pyin_dist": dist,
                "voiced_frac": voiced_frac,
                "category": cat,
            }
        )

    import collections

    counts = collections.Counter(r["category"] for r in rows)
    total = len(rows)
    print()
    print(f"=== 노트별 오디오 대조 결과 ({total} notes) ===")
    for cat in ["pyin_agree", "octave_off", "pitch_off", "unvoiced"]:
        c = counts.get(cat, 0)
        print(f"  {cat:12s}: {c:4d}  ({c/total:.1%})")

    print()
    print("=== octave_off 노트 (pYIN 기준 옥타브가 다름) ===")
    for r in rows:
        if r["category"] == "octave_off":
            direction = "MIDI가 높음" if r["pyin_dist"] < 0 else "MIDI가 낮음"
            print(
                f"  {r['start']:7.2f}s dur={r['dur']:.3f}s midi={r['pitch']:3d} "
                f"pyin={r['pyin_median']:6.2f} ({direction})"
            )

    print()
    print("=== pitch_off 노트 (옥타브도 아닌 피치 불일치) ===")
    for r in rows:
        if r["category"] == "pitch_off":
            print(
                f"  {r['start']:7.2f}s dur={r['dur']:.3f}s midi={r['pitch']:3d} "
                f"pyin={r['pyin_median']:6.2f} dist={r['pyin_dist']:+.2f}"
            )

    print()
    print("=== unvoiced 노트 (유령/노이즈 후보, voiced<30%) ===")
    for r in rows:
        if r["category"] == "unvoiced":
            print(f"  {r['start']:7.2f}s dur={r['dur']:.3f}s midi={r['pitch']:3d} voiced={r['voiced_frac']:.0%}")

    if args.csv:
        import csv as csvmod

        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved CSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
