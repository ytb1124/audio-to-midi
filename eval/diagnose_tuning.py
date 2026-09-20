#!/usr/bin/env python3
"""
파일의 실제 튜닝/최저음 범위 진단 (shadow — 자동 적용 안 함, audit만).
(사용자 명세 1단계, 2026-07-13)

신뢰도 높은 안정 구간의 최저 pitch를 찾아 Drop D / 5현(low B) 가능성을 진단.
단발성 최저값(subharmonic 등)은 제외하고 다음을 모두 만족하는 프레임만 사용:
  - 충분한 voiced probability
  - 100~200ms 이상 지속되는 안정 구간(±0.5 semitone)
  - 동일 pitch class가 파일 내 반복
  - harmonic continuity(해당 pitch bin CQT 에너지 지속) 존재

출력: 추정 튜닝, 최저 허용 MIDI 후보, confidence. 자동 적용 없음.

사용법: python eval/diagnose_tuning.py <audio.wav> [--csv out.csv]
"""
import argparse
import sys
from collections import Counter

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
STD_TUNING = {"E1": 28, "A1": 33, "D2": 38, "G2": 43}  # 4현 표준
DROP_D = 26   # Drop D 최저
LOW_B = 23    # 5현 low B


def nm(p):
    p = int(round(p))
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--min-seg-ms", type=float, default=120.0)
    ap.add_argument("--voiced-thresh", type=float, default=0.5)
    args = ap.parse_args()

    import librosa

    print(f"Loading {args.audio}", file=sys.stderr)
    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    hop = 256

    # pYIN: 아주 낮은 음까지 보려고 fmin=B0(약 31Hz)
    f0, voiced_flag, vprob = librosa.pyin(
        y, fmin=librosa.note_to_hz("B0"), fmax=librosa.note_to_hz("C5"),
        sr=sr, frame_length=2048, hop_length=hop, fill_na=np.nan,
    )
    midi = np.full(len(f0), np.nan)
    valid = np.isfinite(f0)
    midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)
    vprob = np.nan_to_num(vprob, nan=0.0)

    # CQT (harmonic continuity 확인용)
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
                           n_bins=72, bins_per_octave=12))
    C = np.log1p(10.0 * C)
    base = 24
    frame_dur = hop / sr
    min_frames = int((args.min_seg_ms / 1000.0) / frame_dur)

    # 안정 구간 추출: 연속 프레임이 voiced && 같은 반음(±0.5)에 머묾
    segments = []  # (start_frame, end_frame, median_midi)
    i = 0
    n = len(midi)
    while i < n:
        if not (valid[i] and vprob[i] >= args.voiced_thresh):
            i += 1
            continue
        j = i
        anchor = midi[i]
        while j < n and valid[j] and vprob[j] >= args.voiced_thresh and abs(midi[j] - anchor) <= 0.5:
            j += 1
        if j - i >= min_frames:
            seg_mid = float(np.nanmedian(midi[i:j]))
            # harmonic continuity: 해당 pitch bin이 실제로 에너지 유지?
            pidx = int(round(seg_mid)) - base
            if 0 <= pidx < C.shape[0]:
                e = float(np.median(C[pidx, i:j]))
                if e >= 0.5:
                    segments.append((i, j, seg_mid, e, (j - i) * frame_dur))
        i = j

    if not segments:
        print("안정 구간 없음 — 진단 불가", file=sys.stderr)
        return

    # 반음(round) 기준 최저 pitch 반복성
    seg_pitches = [int(round(s[2])) for s in segments]
    pc_counter = Counter(seg_pitches)
    lowest_repeated = min(p for p, c in pc_counter.items() if c >= 3) if any(c >= 3 for c in pc_counter.values()) else min(seg_pitches)

    print(f"\n=== 튜닝/최저음 진단: {args.audio.split('/')[-1]} ===")
    print(f"안정 구간(≥{args.min_seg_ms:.0f}ms, voiced≥{args.voiced_thresh}, harmonic 확인): {len(segments)}개")
    print(f"\n최저 5개 안정 pitch (반복성):")
    for p in sorted(pc_counter)[:8]:
        marker = ""
        if p <= LOW_B + 1:
            marker = " ← 5현 low B 영역"
        elif p <= DROP_D + 1:
            marker = " ← Drop D 영역"
        elif p < STD_TUNING["E1"]:
            marker = " ← 표준 E1 아래"
        print(f"  {nm(p):4s} (MIDI {p}): {pc_counter[p]}회{marker}")

    # 추정
    reliable_low = min(p for p, c in pc_counter.items() if c >= 3) if any(c >= 3 for c in pc_counter.values()) else None
    print()
    if reliable_low is None:
        tuning = "판정불가(반복 부족)"; min_midi_sugg = 28; conf = 0.0
    elif reliable_low <= LOW_B + 1:
        tuning = "5현(low B) 가능성"; min_midi_sugg = 23; conf = min(1.0, pc_counter[reliable_low] / 10)
    elif reliable_low <= DROP_D + 1:
        tuning = "Drop D 가능성"; min_midi_sugg = 26; conf = min(1.0, pc_counter[reliable_low] / 10)
    else:
        tuning = "표준 E-A-D-G"; min_midi_sugg = 28; conf = min(1.0, pc_counter.get(28, 0) / 10)

    print(f"추정 튜닝: {tuning}")
    print(f"신뢰도 높은 최저 안정 pitch: {nm(reliable_low) if reliable_low else 'N/A'} (반복 {pc_counter.get(reliable_low,0)}회)")
    print(f"최저 허용 MIDI 후보(제안): {min_midi_sugg} ({nm(min_midi_sugg)})  — 현재 파이프라인 기본 28(E1)")
    print(f"confidence: {conf:.2f}")
    print("\n※ 자동 적용 안 함. 이 진단으로 min_midi 조정 여부는 사용자 결정.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["seg_start_s", "seg_dur_s", "midi", "note", "cqt_energy"])
            for (a, b, mid, e, dur) in segments:
                w.writerow([round(a * frame_dur, 3), round(dur, 3), int(round(mid)), nm(mid), round(e, 3)])
        print(f"Saved segment audit: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
