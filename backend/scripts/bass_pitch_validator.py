#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def midi_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_note // 12 - 1
    return f"{names[midi_note % 12]}{octave}"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak
    return y, sr


def build_cqt(y, sr):
    import librosa

    hop_length = 256
    fmin = librosa.note_to_hz("C1")
    n_bins = 72

    C = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=12,
        )
    )

    C = np.log1p(10.0 * C)
    cqt_base_midi = 24
    return C, hop_length, cqt_base_midi


def cqt_energy(C, hop_length, sr, cqt_base_midi, midi_pitch, start, end):
    bin_idx = int(round(midi_pitch - cqt_base_midi))
    if bin_idx < 0 or bin_idx >= C.shape[0]:
        return 0.0

    a = max(0, int(start * sr / hop_length))
    b = min(C.shape[1], int(end * sr / hop_length) + 1)

    if b <= a:
        return 0.0

    return float(np.median(C[bin_idx, a:b]))


def harmonic_score(C, hop_length, sr, cqt_base_midi, pitch, start, end):
    win_start = max(0.0, start + 0.010)
    win_end = max(win_start + 0.060, min(end, start + 0.300))

    e0 = cqt_energy(C, hop_length, sr, cqt_base_midi, pitch, win_start, win_end)
    e12 = cqt_energy(C, hop_length, sr, cqt_base_midi, pitch + 12, win_start, win_end)
    e19 = cqt_energy(C, hop_length, sr, cqt_base_midi, pitch + 19, win_start, win_end)
    e24 = cqt_energy(C, hop_length, sr, cqt_base_midi, pitch + 24, win_start, win_end)

    # 베이스는 fundamental이 약할 수 있으므로 배음도 함께 본다.
    return float(e0 + 0.62 * e12 + 0.34 * e19 + 0.20 * e24)


def choose_pitch_with_audio(
    current_pitch,
    start,
    end,
    C,
    hop_length,
    sr,
    cqt_base_midi,
    min_midi,
    max_midi,
    prev_pitch=None,
    enable_low_octave_bias=False,
):
    candidates = set()

    for q in [
        current_pitch - 24,
        current_pitch - 12,
        current_pitch,
        current_pitch + 12,
    ]:
        if min_midi <= q <= max_midi:
            candidates.add(q)

    if not candidates:
        return current_pitch, [], "no_candidate"

    scored = []

    for p in sorted(candidates):
        s = harmonic_score(C, hop_length, sr, cqt_base_midi, p, start, end)

        # 기존 pitch를 완전히 무시하면 튀는 보정이 생기므로 keep bonus를 준다.
        if p == current_pitch:
            s += 0.13

        # 베이스에서 한 옥타브 위로 잡히는 문제 완화.
        # 단, 무조건 낮추지는 않고 점수가 비슷할 때만 낮은 쪽이 유리해지게 한다.
        # eval 하니스 실측(Bass_2.wav): 이 보너스+페널티 조합이 E2 등 흔한 음을
        # 근소한 점수차만으로 한 옥타브 아래로 밀어내는 사례가 다수 확인되어 기본 비활성화.
        if enable_low_octave_bias:
            if p < current_pitch:
                s += 0.035

            # 너무 높은 음역은 약간 penalty. 실제로 E3까지 쓸 수는 있으므로 강한 제한은 아님.
            if p >= 48:
                s -= 0.045 * (p - 47)

        # 이전 pitch와의 연결성. 갑자기 큰 점프하는 octave error를 줄인다.
        if prev_pitch is not None:
            jump = abs(p - prev_pitch)
            if jump > 12:
                s -= 0.14 * (jump - 12)
            elif jump > 7:
                s -= 0.045 * (jump - 7)

        scored.append((float(s), int(p)))

    scored.sort(reverse=True)
    best_score, best_pitch = scored[0]

    current_score = None
    for s, p in scored:
        if p == current_pitch:
            current_score = s
            break

    if current_score is None:
        current_score = -9999.0

    margin = best_score - current_score

    # 보정 조건:
    # 1) 대체 pitch가 확실히 더 좋거나
    # 2) 현재 pitch가 높은 octave이고, 낮은 후보가 거의 비슷한 점수면 낮춤
    should_change = False

    if best_pitch != current_pitch and margin >= 0.070:
        should_change = True

    # eval 하니스 실측(Bass_2.wav): 아래 두 조건은 낮은 후보가 실제로는 더 나쁜 점수
    # (margin이 음수)여도 낮은 쪽으로 바꿔버려 흔한 음(E2 등)을 오탐으로 낮췄다.
    # 기본 비활성화.
    if enable_low_octave_bias:
        if best_pitch < current_pitch and current_pitch >= 40 and margin >= -0.010:
            should_change = True

        if best_pitch < current_pitch and current_pitch >= 48 and margin >= -0.080:
            should_change = True

    if should_change:
        reason = f"changed_margin_{margin:.4f}"
        return best_pitch, scored, reason

    return current_pitch, scored, f"kept_margin_{margin:.4f}"


def read_midi_notes(midi_path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []

    for inst_idx, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue

        for note_idx, n in enumerate(inst.notes):
            notes.append(
                {
                    "inst_idx": inst_idx,
                    "note_idx": note_idx,
                    "start": float(n.start),
                    "end": float(n.end),
                    "pitch": int(n.pitch),
                    "velocity": int(n.velocity),
                }
            )

    notes.sort(key=lambda x: (x["start"], x["pitch"]))
    return pm, notes


def write_final_notes_csv(notes, csv_path):
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["start", "end", "duration", "pitch", "note", "velocity"],
        )
        writer.writeheader()

        for n in notes:
            writer.writerow(
                {
                    "start": round(float(n["start"]), 6),
                    "end": round(float(n["end"]), 6),
                    "duration": round(float(n["end"] - n["start"]), 6),
                    "pitch": int(n["pitch"]),
                    "note": midi_to_name(int(n["pitch"])),
                    "velocity": int(n["velocity"]),
                }
            )


def make_overlay_png(audio_path, midi_path, out_png, title="Bass pitch audit"):
    try:
        import librosa
        import matplotlib.pyplot as plt
        import pretty_midi
    except Exception as exc:
        eprint("Overlay skipped:", exc)
        return

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    C = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=256,
            fmin=librosa.note_to_hz("C1"),
            n_bins=48,
            bins_per_octave=12,
        )
    )
    C = librosa.amplitude_to_db(C, ref=np.max)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=256)

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    plt.figure(figsize=(18, 7))
    plt.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 24, 72],
        cmap="magma",
        vmin=-70,
        vmax=0,
    )

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            plt.plot([n.start, n.end], [n.pitch, n.pitch], linewidth=1.3)

    plt.title(title)
    plt.xlabel("time seconds")
    plt.ylabel("MIDI pitch")
    plt.ylim(24, 60)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=140)
    plt.close()


def audit_and_repair_audio_midi(
    audio_path,
    midi_in,
    midi_out,
    output_dir,
    min_midi=28,
    max_midi=55,
    enable_low_octave_bias=False,
):
    audio_path = Path(audio_path)
    midi_in = Path(midi_in)
    midi_out = Path(midi_out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path)
    C, hop_length, cqt_base_midi = build_cqt(y, sr)

    pm, notes = read_midi_notes(midi_in)

    audit_rows = []
    correction_rows = []

    prev_pitch = None
    corrected_pitches = {}

    for idx, n in enumerate(notes):
        old_pitch = int(n["pitch"])

        new_pitch, scored, reason = choose_pitch_with_audio(
            current_pitch=old_pitch,
            start=float(n["start"]),
            end=float(n["end"]),
            C=C,
            hop_length=hop_length,
            sr=sr,
            cqt_base_midi=cqt_base_midi,
            min_midi=min_midi,
            max_midi=max_midi,
            prev_pitch=prev_pitch,
            enable_low_octave_bias=enable_low_octave_bias,
        )

        scored_debug = [
            {
                "score": round(float(s), 6),
                "pitch": int(p),
                "note": midi_to_name(int(p)),
            }
            for s, p in scored
        ]

        audit_rows.append(
            {
                "index": idx,
                "start": round(float(n["start"]), 6),
                "end": round(float(n["end"]), 6),
                "old_pitch": old_pitch,
                "old_note": midi_to_name(old_pitch),
                "new_pitch": int(new_pitch),
                "new_note": midi_to_name(int(new_pitch)),
                "changed": int(new_pitch != old_pitch),
                "reason": reason,
                "scores": json.dumps(scored_debug, ensure_ascii=False),
            }
        )

        if new_pitch != old_pitch:
            correction_rows.append(audit_rows[-1])

        corrected_pitches[(n["inst_idx"], n["note_idx"])] = int(new_pitch)
        prev_pitch = int(new_pitch)

    # 실제 pretty_midi note에 반영
    for inst_idx, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue
        for note_idx, note in enumerate(inst.notes):
            key = (inst_idx, note_idx)
            if key in corrected_pitches:
                note.pitch = int(corrected_pitches[key])

    pm.write(str(midi_out))

    audit_csv = output_dir / "bass_pitch_audit.csv"
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
                "reason",
                "scores",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    corrections_csv = output_dir / "bass_pitch_corrections.csv"
    with corrections_csv.open("w", newline="", encoding="utf-8") as f:
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
                "reason",
                "scores",
            ],
        )
        writer.writeheader()
        for row in correction_rows:
            writer.writerow(row)

    final_csv = output_dir / "bass_final_notes_pitch_validated.csv"
    repaired_notes = []
    _, new_notes = read_midi_notes(midi_out)
    for n in new_notes:
        repaired_notes.append(n)
    write_final_notes_csv(repaired_notes, final_csv)

    overlay_png = output_dir / "bass_pitch_overlay.png"
    make_overlay_png(audio_path, midi_out, overlay_png, title="Bass pitch audit after validation")

    eprint(f"Pitch audit rows: {len(audit_rows)}")
    eprint(f"Pitch corrections: {len(correction_rows)}")
    eprint(f"Saved pitch audit: {audit_csv}")
    eprint(f"Saved pitch corrections: {corrections_csv}")
    eprint(f"Saved pitch validated notes: {final_csv}")
    eprint(f"Saved pitch overlay: {overlay_png}")

    return {
        "audit_csv": str(audit_csv),
        "corrections_csv": str(corrections_csv),
        "final_csv": str(final_csv),
        "overlay_png": str(overlay_png),
        "corrections": len(correction_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-midi", type=int, default=28)
    parser.add_argument("--max-midi", type=int, default=55)
    parser.add_argument("--enable-low-octave-bias", action="store_true")
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    audit_and_repair_audio_midi(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
        enable_low_octave_bias=args.enable_low_octave_bias,
    )


if __name__ == "__main__":
    main()
