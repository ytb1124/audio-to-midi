#!/usr/bin/env python3
"""
파이프라인의 다른 모든 처리(ghost note 제거, sustain, timing)가 끝난 뒤
마지막에 딱 한 번 옥타브를 정리하는 스테이지.

배경: eval/train_octave_classifier.py 실험 결과, 기존 4개 스테이지
(pitch_validator/post_refine/stable_pitch_guard/octave_anchor)가 각자
CQT harmonic score를 근소한 margin 임계값으로 비교하며 산발적으로
옥타브를 판단하다 보니 서로 결과를 뒤집는 thrash가 생기고 있었다.
반면 "같은 pitch class의 유효한 모든 옥타브 후보를 놓고 harmonic score
argmax를 딱 한 번" 매기는 방식은 bass_test.mid 1284개 노트 전체에서
1283/1284(99.92%) 정확도를 보였다 — 틀린 1개도 구조적으로 불가능한
폴리포니 겹침 케이스였다.

이 스테이지는 새 pitch class를 발명하지 않는다(같은 pitch class 내에서만
옥타브 이동). 노트를 삭제/추가하지 않는다. 옥타브만 재판정한다.
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
            y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
            n_bins=72, bins_per_octave=12,
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


def choose_final_octave(note, C, hop, sr, base_midi, min_midi, max_midi):
    current = int(note.pitch)
    pitch_class = current % 12
    start = float(note.start)
    end = float(note.end)

    candidates = [p for p in range(min_midi, max_midi + 1) if p % 12 == pitch_class]
    if not candidates:
        return current, {}

    scores = {p: harmonic_support(C, hop, sr, base_midi, p, start, end) for p in candidates}
    best_pitch = max(scores, key=scores.get)
    return best_pitch, scores


def octave_final(audio_path, midi_in, midi_out, output_dir, min_midi=28, max_midi=60):
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

    audit_rows = []
    changed = 0

    for note in notes:
        old_pitch = int(note.pitch)
        new_pitch, scores = choose_final_octave(note, C, hop, sr, base_midi, min_midi, max_midi)

        if new_pitch != old_pitch:
            changed += 1
            note.pitch = int(new_pitch)

        audit_rows.append(
            {
                "start": round(float(note.start), 6),
                "old_pitch": old_pitch,
                "old_note": midi_to_name(old_pitch),
                "new_pitch": int(new_pitch),
                "new_note": midi_to_name(int(new_pitch)),
                "changed": int(new_pitch != old_pitch),
                "scores": ";".join(f"{midi_to_name(p)}={s:.4f}" for p, s in sorted(scores.items())),
            }
        )

    pm.write(str(midi_out))

    audit_csv = output_dir / "bass_octave_final_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["start", "old_pitch", "old_note", "new_pitch", "new_note", "changed", "scores"]
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    eprint(f"Octave final: {len(notes)} notes, {changed} changed")
    eprint(f"Saved audit: {audit_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-midi", type=int, default=28)
    parser.add_argument("--max-midi", type=int, default=60)
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    octave_final(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
    )


if __name__ == "__main__":
    main()
