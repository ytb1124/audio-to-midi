#!/usr/bin/env python3
"""
가까운 시간에 이어지는 같은 pitch class 노트들("run")의 옥타브를 통일시킨다.

배경: 여러 파일에서 실측한 결과, 인접한 노트가 같은 음이름인데 옥타브만
왔다갔다 하는 "jitter"가 다수 발견됐다 (예: E2 -> E1 -> E2, 9~130ms 간격).
각 스테이지가 노트 하나하나를 독립적으로 채점하다 보니, 점수차가 근소할 때
사실상 동전 던지기가 되어 생기는 문제로 보인다.

이 스테이지는 새 pitch를 "발명"하지 않는다. 같은 run 안에서 이미 존재하는
후보 옥타브들의 CQT 증거를 다 모아서(pool) 하나로 재판정하고, run 전체에
그 옥타브를 적용한다. 증거가 애매하면(margin이 작으면) 다수결로 유지한다.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def midi_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if len(y) == 0:
        raise RuntimeError("Empty audio")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    return y, sr


def build_cqt(y, sr):
    import librosa

    hop = 256
    base_midi = 24

    C = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop,
            fmin=librosa.note_to_hz("C1"),
            n_bins=72,
            bins_per_octave=12,
        )
    )
    C = np.log1p(10.0 * C)
    return C, hop, base_midi


def cqt_energy(C, hop, sr, base_midi, pitch, start, end):
    idx = int(round(pitch - base_midi))
    if idx < 0 or idx >= C.shape[0]:
        return 0.0

    a = max(0, int((start + 0.015) * sr / hop))
    b = min(C.shape[1], int(min(end, start + 0.30) * sr / hop) + 1)
    if b <= a:
        return 0.0

    return float(np.median(C[idx, a:b]))


def harmonic_support(C, hop, sr, base_midi, pitch, start, end):
    e0 = cqt_energy(C, hop, sr, base_midi, pitch, start, end)
    e12 = cqt_energy(C, hop, sr, base_midi, pitch + 12, start, end)
    e19 = cqt_energy(C, hop, sr, base_midi, pitch + 19, start, end)
    e24 = cqt_energy(C, hop, sr, base_midi, pitch + 24, start, end)
    return float(e0 + 0.42 * e12 + 0.22 * e19 + 0.12 * e24)


def read_midi_notes(midi_path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    inst = None
    for candidate in pm.instruments:
        if not candidate.is_drum:
            inst = candidate
            break

    if inst is None:
        return pm, None, []

    inst.notes.sort(key=lambda n: (n.start, n.pitch))
    return pm, inst, inst.notes


def group_runs(notes, max_gap_sec):
    """시간상 가깝고 pitch class가 같은 노트들을 run으로 묶는다."""
    runs = []
    current = [notes[0]]

    for note in notes[1:]:
        prev = current[-1]
        gap = float(note.start) - float(prev.end)

        if gap <= max_gap_sec and int(note.pitch) % 12 == int(prev.pitch) % 12:
            current.append(note)
        else:
            runs.append(current)
            current = [note]

    runs.append(current)
    return runs


def resolve_run_octave(run, C, hop, sr, base_midi, min_midi, max_midi, min_margin):
    """run 안에서 이미 쓰인 옥타브들만 후보로 삼아 pool된 증거로 재판정."""
    pitch_class = int(run[0].pitch) % 12

    candidate_octaves = sorted(set(int(n.pitch) // 12 for n in run))
    if len(candidate_octaves) <= 1:
        return None  # 이미 일관됨, 건드릴 필요 없음

    candidates = [oct_idx * 12 + pitch_class for oct_idx in candidate_octaves]
    candidates = [p for p in candidates if min_midi <= p <= max_midi]
    if len(candidates) <= 1:
        return None

    pooled_scores = {}
    for p in candidates:
        total = 0.0
        for n in run:
            total += harmonic_support(C, hop, sr, base_midi, p, float(n.start), float(n.end))
        pooled_scores[p] = total

    ranked = sorted(pooled_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_pitch, best_score = ranked[0]

    # 현재 run에서 가장 많이 쓰인 옥타브(다수결)를 기본값으로.
    from collections import Counter

    vote_counts = Counter(int(n.pitch) for n in run)
    majority_pitch, _ = vote_counts.most_common(1)[0]

    if best_pitch == majority_pitch:
        return best_pitch, "pooled_agrees_with_majority"

    majority_score = pooled_scores.get(majority_pitch, -1e9)
    margin = best_score - majority_score

    if margin >= min_margin:
        return best_pitch, f"pooled_overrides_majority_margin_{margin:.4f}"

    return majority_pitch, f"kept_majority_pooled_margin_{margin:.4f}_too_small"


def octave_consistency(audio_path, midi_in, midi_out, output_dir, min_midi=28, max_midi=60,
                        max_gap_sec=0.3, min_margin=0.15):
    audio_path = Path(audio_path)
    midi_in = Path(midi_in)
    midi_out = Path(midi_out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path)
    C, hop, base_midi = build_cqt(y, sr)

    pm, inst, notes = read_midi_notes(midi_in)
    if inst is None or not notes:
        eprint("No MIDI notes to process")
        return

    runs = group_runs(notes, max_gap_sec=max_gap_sec)

    audit_rows = []
    changed = 0

    for run in runs:
        if len(run) < 2:
            continue

        result = resolve_run_octave(run, C, hop, sr, base_midi, min_midi, max_midi, min_margin)
        if result is None:
            continue

        target_pitch, reason = result
        target_class = target_pitch % 12

        for n in run:
            old_pitch = int(n.pitch)
            new_pitch = (int(n.pitch) // 12) * 12 + target_class
            # run 안의 각 노트를 target_pitch와 "같은 옥타브"로 맞춘다
            new_pitch = target_pitch

            if new_pitch != old_pitch:
                changed += 1
                audit_rows.append(
                    {
                        "start": round(float(n.start), 6),
                        "old_pitch": old_pitch,
                        "old_note": midi_to_name(old_pitch),
                        "new_pitch": new_pitch,
                        "new_note": midi_to_name(new_pitch),
                        "run_size": len(run),
                        "reason": reason,
                    }
                )
                n.pitch = int(new_pitch)

    pm.write(str(midi_out))

    audit_csv = output_dir / "bass_octave_consistency_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["start", "old_pitch", "old_note", "new_pitch", "new_note", "run_size", "reason"]
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    eprint(f"Octave consistency: {len(runs)} runs checked, {changed} notes changed")
    eprint(f"Saved audit: {audit_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-midi", type=int, default=28)
    parser.add_argument("--max-midi", type=int, default=60)
    parser.add_argument("--max-gap-sec", type=float, default=0.3)
    parser.add_argument("--min-margin", type=float, default=0.15)
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    octave_consistency(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
        max_gap_sec=args.max_gap_sec,
        min_margin=args.min_margin,
    )


if __name__ == "__main__":
    main()
