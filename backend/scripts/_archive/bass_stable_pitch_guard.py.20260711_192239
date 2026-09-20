#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def midi_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def hz_to_midi(hz):
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return None
    return 69.0 + 12.0 * math.log2(float(hz) / 440.0)


def load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if len(y) == 0:
        raise RuntimeError("Empty audio")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    return y, sr


def read_midi(midi_path):
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


def compute_pyin(y, sr):
    import librosa

    hop = 256

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("G3"),
        sr=sr,
        frame_length=2048,
        hop_length=hop,
        fill_na=np.nan,
    )

    midi = np.array(
        [hz_to_midi(x) if x is not None and np.isfinite(x) else np.nan for x in f0],
        dtype=float,
    )

    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)

    return midi, voiced_prob, hop


def pyin_stats(pyin_midi, pyin_prob, hop, sr, start, end):
    a = max(0, int(start * sr / hop))
    b = min(len(pyin_midi), int(min(end, start + 0.35) * sr / hop) + 1)

    if b <= a:
        return None, 0.0

    seg = pyin_midi[a:b]
    prob = pyin_prob[a:b]

    valid = np.isfinite(seg)
    if not np.any(valid):
        return None, float(np.median(prob)) if len(prob) else 0.0

    return float(np.nanmedian(seg[valid])), float(np.nanmedian(prob[valid]))


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


def bass_score(C, hop, sr, base_midi, pitch, start, end):
    e0 = cqt_energy(C, hop, sr, base_midi, pitch, start, end)
    e12 = cqt_energy(C, hop, sr, base_midi, pitch + 12, start, end)
    e19 = cqt_energy(C, hop, sr, base_midi, pitch + 19, start, end)

    # 너무 강한 고역 penalty는 피치 불안정의 원인이 되므로 약하게만 둔다.
    score = e0 + 0.45 * e12 + 0.20 * e19

    if pitch >= 48:
        score -= 0.08 * (pitch - 47)

    return float(score)


def choose_stable_pitch(note, py_med, py_conf, C, cqt_hop, sr, base_midi, min_midi, max_midi, prev_pitch=None, next_pitch=None):
    current = int(note.pitch)
    start = float(note.start)
    end = float(note.end)

    candidates = []

    for p in [current - 24, current - 12, current, current + 12]:
        if min_midi <= p <= max_midi and p % 12 == current % 12:
            candidates.append(p)

    candidates = sorted(set(candidates))

    if not candidates:
        return current, "keep_no_candidate", []

    scored = []

    for p in candidates:
        s = bass_score(C, cqt_hop, sr, base_midi, p, start, end)

        # 현재 pitch 유지 보너스.
        # 너무 쉽게 바꾸면 v6처럼 불안정해진다.
        if p == current:
            s += 0.18

        # pYIN은 octave validator로만 사용.
        if py_med is not None and np.isfinite(py_med) and py_conf >= 0.18:
            diff = abs(p - py_med)

            if diff <= 1.25:
                s += 0.55 * py_conf
            elif 10.5 <= diff <= 13.5:
                s -= 0.12 * py_conf
            elif diff >= 5.0:
                s -= 0.28 * py_conf

            # 현재가 pYIN보다 한 옥타브 위일 때만 octave down 강하게 허용.
            if 10.5 <= current - py_med <= 13.5 and p == current - 12:
                s += 0.72 * py_conf

            # 현재가 pYIN보다 두 옥타브 위일 때.
            if 22.0 <= current - py_med <= 25.8 and p == current - 24:
                s += 0.80 * py_conf

        # 앞뒤 문맥이 같은 octave 후보를 지지하면 약간 보너스.
        if prev_pitch is not None:
            if abs(p - prev_pitch) <= 2:
                s += 0.12
            elif abs(p - prev_pitch) >= 12:
                s -= 0.16

        if next_pitch is not None:
            if abs(p - next_pitch) <= 2:
                s += 0.08
            elif abs(p - next_pitch) >= 12:
                s -= 0.10

        scored.append((s, p))

    scored.sort(reverse=True)
    best_score, best_pitch = scored[0]

    current_score = None
    for s, p in scored:
        if p == current:
            current_score = s
            break

    if current_score is None:
        current_score = -999.0

    margin = best_score - current_score

    # 매우 보수적으로 변경.
    # 같은 pitch class octave correction만 허용.
    should_change = False

    if best_pitch != current:
        # pYIN이 확실하게 current가 한 옥타브 위라고 말할 때
        if py_med is not None and py_conf >= 0.22:
            if 10.5 <= current - py_med <= 13.5 and best_pitch == current - 12 and margin >= -0.03:
                should_change = True
            elif 22.0 <= current - py_med <= 25.8 and best_pitch == current - 24 and margin >= -0.06:
                should_change = True

        # 아주 고음 octave만 보수적으로 내림
        if current >= 48 and best_pitch < current and margin >= 0.02:
            should_change = True

        # CQT 점수 차이가 확실할 때만
        if margin >= 0.22:
            should_change = True

    if should_change:
        return int(best_pitch), f"octave_changed_margin_{margin:.4f}_pyin_{py_med}_conf_{py_conf:.3f}", scored

    return current, f"kept_margin_{margin:.4f}_pyin_{py_med}_conf_{py_conf:.3f}", scored


def stable_pitch_guard(audio_path, midi_in, midi_out, output_dir, min_midi=28, max_midi=55):
    import pretty_midi

    audio_path = Path(audio_path)
    midi_in = Path(midi_in)
    midi_out = Path(midi_out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path)
    pm, inst, notes = read_midi(midi_in)

    if inst is None or not notes:
        eprint("No MIDI notes to guard")
        return

    pyin_midi, pyin_prob, pyin_hop = compute_pyin(y, sr)
    C, cqt_hop, base_midi = build_cqt(y, sr)

    old_pitches = [int(n.pitch) for n in notes]
    new_pitches = []
    audit_rows = []

    for i, note in enumerate(notes):
        prev_pitch = old_pitches[i - 1] if i > 0 else None
        next_pitch = old_pitches[i + 1] if i + 1 < len(old_pitches) else None

        py_med, py_conf = pyin_stats(
            pyin_midi,
            pyin_prob,
            pyin_hop,
            sr,
            float(note.start),
            float(note.end),
        )

        new_pitch, reason, scored = choose_stable_pitch(
            note=note,
            py_med=py_med,
            py_conf=py_conf,
            C=C,
            cqt_hop=cqt_hop,
            sr=sr,
            base_midi=base_midi,
            min_midi=min_midi,
            max_midi=max_midi,
            prev_pitch=prev_pitch,
            next_pitch=next_pitch,
        )

        new_pitches.append(new_pitch)

        audit_rows.append(
            {
                "index": i,
                "start": round(float(note.start), 6),
                "end": round(float(note.end), 6),
                "old_pitch": int(note.pitch),
                "old_note": midi_to_name(int(note.pitch)),
                "new_pitch": int(new_pitch),
                "new_note": midi_to_name(int(new_pitch)),
                "changed": int(new_pitch != int(note.pitch)),
                "pyin_median": "" if py_med is None else round(float(py_med), 4),
                "pyin_note": "" if py_med is None else midi_to_name(int(round(py_med))),
                "pyin_conf": round(float(py_conf), 5),
                "reason": reason,
                "scores": "; ".join(
                    [f"{midi_to_name(p)}:{s:.3f}" for s, p in scored[:6]]
                ),
            }
        )

    # pitch만 반영. 노트 삭제 없음.
    for note, p in zip(notes, new_pitches):
        note.pitch = int(p)

    # sustain은 과하게 재계산하지 않음.
    # overlap만 정리하고, 기존 길이는 최대한 보존.
    notes.sort(key=lambda n: (n.start, n.pitch))
    min_len = 0.045

    for i in range(len(notes) - 1):
        if notes[i].end > notes[i + 1].start - 0.008:
            notes[i].end = max(notes[i].start + min_len, notes[i + 1].start - 0.008)

    for n in notes:
        if n.end <= n.start:
            n.end = n.start + min_len
        elif n.end - n.start < min_len:
            n.end = n.start + min_len

    pm.write(str(midi_out))

    audit_csv = output_dir / "bass_stable_pitch_guard_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "start",
                "end",
                "old_pitch",
                "old_note",
                "new_pitch",
                "new_note",
                "changed",
                "pyin_median",
                "pyin_note",
                "pyin_conf",
                "reason",
                "scores",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    final_csv = output_dir / "bass_final_notes_stable_guard.csv"
    with final_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["start", "end", "duration", "pitch", "note", "velocity"],
        )
        writer.writeheader()
        for n in notes:
            writer.writerow(
                {
                    "start": round(float(n.start), 6),
                    "end": round(float(n.end), 6),
                    "duration": round(float(n.end - n.start), 6),
                    "pitch": int(n.pitch),
                    "note": midi_to_name(int(n.pitch)),
                    "velocity": int(n.velocity),
                }
            )

    changed = sum(1 for r in audit_rows if int(r["changed"]) == 1)

    eprint(f"Stable pitch guard notes: {len(notes)}")
    eprint(f"Stable pitch guard pitch changes: {changed}")
    eprint(f"Saved stable guard audit: {audit_csv}")
    eprint(f"Saved stable guard final csv: {final_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-midi", type=int, default=28)
    parser.add_argument("--max-midi", type=int, default=55)
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    stable_pitch_guard(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
    )


if __name__ == "__main__":
    main()
