#!/usr/bin/env python3
"""
같은 pitch로 이어지는 인접 노트 중, 경계에 실제 재타격(attack transient)
증거가 없는 것만 하나로 합친다.

배경: bass_sample.wav 정답 MIDI와 대조해보니, 지속음 하나가 여러 조각으로
쪼개지는 사례들의 경계에서 RMS 진폭이 감쇠(decay) 도중이었을 뿐 실제로는
튀어 오르지 않았다 (스파이크 없음). 반대로 bass_test.wav의 진짜 16분음표
연타 구간은 경계마다 RMS가 뚜렷하게 rise한다. 이 rise 유무로 "진짜 재타격"과
"지속음이 잘못 쪼개진 것"을 구분한다.

노트를 삭제하지 않는다. pitch도 바꾸지 않는다. 같은 pitch로 이어지는 두
노트를 하나로 합칠지 말지만 판단한다.
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


def build_rms_envelope(y, sr):
    import librosa

    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=1024)[0]
    return rms, hop


def rms_at(rms, hop, sr, t):
    idx = int(round(t * sr / hop))
    idx = max(0, min(len(rms) - 1, idx))
    return float(rms[idx])


def has_attack(rms, hop, sr, boundary_time, pre_sec=0.05, post_sec=0.03, rise_ratio=1.2):
    """boundary_time 근처에 실제 재타격(rise)이 있는지 확인.
    pre 구간의 '최소값'과 post 구간의 '최대값'을 비교한다.
    (decay 중에는 pre_max가 이미 높아서 오탐하므로 pre_min을 쓴다.)"""
    pre_a = boundary_time - pre_sec
    pre_b = boundary_time - 0.005
    post_a = boundary_time + 0.005
    post_b = boundary_time + post_sec

    def window_stat(a, b, fn):
        ia = max(0, int(a * sr / hop))
        ib = min(len(rms), int(b * sr / hop) + 1)
        if ib <= ia:
            return None
        return float(fn(rms[ia:ib]))

    pre_min = window_stat(pre_a, pre_b, np.min)
    post_max = window_stat(post_a, post_b, np.max)

    if pre_min is None or post_max is None:
        return True  # 판단 불가하면 안전하게 "재타격 있음"으로 보고 안 건드림

    if pre_min <= 1e-6:
        return post_max > 0.01  # 무음에서 시작하는 진짜 attack

    return (post_max / pre_min) >= rise_ratio


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


def merge_fragments(notes, rms, hop, sr, max_gap_sec=0.06, rise_ratio=1.2):
    if not notes:
        return [], []

    merged = [notes[0]]
    audit_rows = []

    for note in notes[1:]:
        last = merged[-1]

        gap = float(note.start) - float(last.end)
        same_pitch = int(note.pitch) == int(last.pitch)

        if same_pitch and gap <= max_gap_sec:
            boundary = float(note.start)
            attacked = has_attack(rms, hop, sr, boundary, rise_ratio=rise_ratio)

            audit_rows.append(
                {
                    "boundary": round(boundary, 6),
                    "pitch": int(note.pitch),
                    "note": midi_to_name(int(note.pitch)),
                    "gap": round(gap, 6),
                    "attacked": int(attacked),
                    "action": "kept_separate" if attacked else "merged",
                }
            )

            if not attacked:
                last.end = max(last.end, note.end)
                last.velocity = max(last.velocity, note.velocity)
                continue

        merged.append(note)

    return merged, audit_rows


def fragment_merge(audio_path, midi_in, midi_out, output_dir, max_gap_sec=0.06, rise_ratio=1.2):
    audio_path = Path(audio_path)
    midi_in = Path(midi_in)
    midi_out = Path(midi_out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path)
    rms, hop = build_rms_envelope(y, sr)

    pm, inst, notes = read_midi_notes(midi_in)

    if inst is None or not notes:
        eprint("No MIDI notes to merge")
        return

    before_count = len(notes)
    merged, audit_rows = merge_fragments(notes, rms, hop, sr, max_gap_sec=max_gap_sec, rise_ratio=rise_ratio)

    inst.notes = merged
    pm.write(str(midi_out))

    audit_csv = output_dir / "bass_fragment_merge_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["boundary", "pitch", "note", "gap", "attacked", "action"])
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    merged_count = sum(1 for r in audit_rows if r["action"] == "merged")

    eprint(f"Fragment merge: {before_count} -> {len(merged)} notes ({merged_count} merges)")
    eprint(f"Saved fragment merge audit: {audit_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-gap-sec", type=float, default=0.06)
    parser.add_argument("--rise-ratio", type=float, default=1.2)
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    fragment_merge(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        max_gap_sec=args.max_gap_sec,
        rise_ratio=args.rise_ratio,
    )


if __name__ == "__main__":
    main()
