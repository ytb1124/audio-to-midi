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

    med = float(np.nanmedian(seg[valid]))
    conf = float(np.nanmedian(prob[valid]))

    return med, conf


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
    b = min(C.shape[1], int(min(end, start + 0.32) * sr / hop) + 1)

    if b <= a:
        return 0.0

    return float(np.median(C[idx, a:b]))


def harmonic_support(C, hop, sr, base_midi, pitch, start, end):
    """
    특정 pitch가 fundamental이라고 가정했을 때의 support.
    E1이면 E1, E2, B2, E3 근처를 같이 본다.
    """
    e0 = cqt_energy(C, hop, sr, base_midi, pitch, start, end)
    e12 = cqt_energy(C, hop, sr, base_midi, pitch + 12, start, end)
    e19 = cqt_energy(C, hop, sr, base_midi, pitch + 19, start, end)
    e24 = cqt_energy(C, hop, sr, base_midi, pitch + 24, start, end)

    return float(e0 + 0.42 * e12 + 0.22 * e19 + 0.12 * e24)


def choose_octave(note, py_med, py_conf, C, cqt_hop, sr, base_midi, min_midi, max_midi, enable_pyin_octave_down=False, enable_cqt_octave_down=False):
    current = int(note.pitch)
    start = float(note.start)
    end = float(note.end)

    # 낮출 수 없는 음은 건드리지 않음
    if current - 12 < min_midi:
        return current, "keep_no_lower_octave", {}

    # E1~B1 같은 저음은 이미 낮으므로 건드리지 않음
    if current < 40:
        return current, "keep_low_register", {}

    lower = current - 12

    current_score = harmonic_support(C, cqt_hop, sr, base_midi, current, start, end)
    lower_score = harmonic_support(C, cqt_hop, sr, base_midi, lower, start, end)

    info = {
        "current_score": current_score,
        "lower_score": lower_score,
        "score_margin_lower_minus_current": lower_score - current_score,
        "pyin_median": py_med,
        "pyin_conf": py_conf,
    }

    # 1) pYIN이 lower octave 근처를 말하면 octave down.
    # eval/ground_truth/bass_test 하니스 실측: 이 규칙이 트리거된 16건 전부 오탐
    # (정답보다 한 옥타브 낮춤, 기존 octave error를 고친 사례 0건).
    # 기본 비활성화. 근거: eval/runs 로 재검증 후에만 다시 켤 것.
    if enable_pyin_octave_down and py_med is not None and np.isfinite(py_med) and py_conf >= 0.10:
        if abs(py_med - lower) <= 1.65 and abs(py_med - current) >= 8.0:
            return lower, "down_pyin_matches_lower_octave", info

    # 2) E2/B2/D3/E3 계열에서 lower harmonic support가 비슷하거나 더 좋으면 down.
    # 베이스는 위 octave 배음이 강해서 current_score가 살짝 높아도 lower가 맞을 수 있다.
    # eval/ground_truth Bass_2 실측: margin 0.08은 사실상 노이즈 수준이라 E2(가장 흔한
    # 개방현 음)의 ~11%가 근소한 점수차만으로 E1로 뒤집힘. 기본 비활성화.
    if enable_cqt_octave_down and current >= 40:
        if lower_score >= current_score - 0.08:
            return lower, "down_lower_cqt_close_enough", info

    # 3) 매우 높은 음역은 더 강하게 의심.
    if enable_cqt_octave_down and current >= 48:
        if lower_score >= current_score - 0.22:
            return lower, "down_high_register_octave_suspect", info

    return current, "keep_octave_not_enough_evidence", info


def octave_anchor(audio_path, midi_in, midi_out, output_dir, min_midi=28, max_midi=55, enable_pyin_octave_down=False, enable_cqt_octave_down=False):
    audio_path = Path(audio_path)
    midi_in = Path(midi_in)
    midi_out = Path(midi_out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path)

    pm, inst, notes = read_midi(midi_in)

    if inst is None or not notes:
        eprint("No MIDI notes found")
        return

    pyin_midi, pyin_prob, pyin_hop = compute_pyin(y, sr)
    C, cqt_hop, base_midi = build_cqt(y, sr)

    audit_rows = []
    changed = 0

    for i, note in enumerate(notes):
        old_pitch = int(note.pitch)

        py_med, py_conf = pyin_stats(
            pyin_midi,
            pyin_prob,
            pyin_hop,
            sr,
            float(note.start),
            float(note.end),
        )

        new_pitch, reason, info = choose_octave(
            note=note,
            py_med=py_med,
            py_conf=py_conf,
            C=C,
            cqt_hop=cqt_hop,
            sr=sr,
            base_midi=base_midi,
            min_midi=min_midi,
            max_midi=max_midi,
            enable_pyin_octave_down=enable_pyin_octave_down,
            enable_cqt_octave_down=enable_cqt_octave_down,
        )

        if new_pitch != old_pitch:
            changed += 1
            note.pitch = int(new_pitch)

        audit_rows.append(
            {
                "index": i,
                "start": round(float(note.start), 6),
                "end": round(float(note.end), 6),
                "old_pitch": old_pitch,
                "old_note": midi_to_name(old_pitch),
                "new_pitch": int(new_pitch),
                "new_note": midi_to_name(int(new_pitch)),
                "changed": int(new_pitch != old_pitch),
                "reason": reason,
                "pyin_median": "" if py_med is None else round(float(py_med), 4),
                "pyin_note": "" if py_med is None else midi_to_name(int(round(py_med))),
                "pyin_conf": round(float(py_conf), 5),
                "current_score": round(float(info.get("current_score", 0.0)), 6),
                "lower_score": round(float(info.get("lower_score", 0.0)), 6),
                "lower_minus_current": round(float(info.get("score_margin_lower_minus_current", 0.0)), 6),
            }
        )

    # 노트 삭제 없음. sustain 변경 없음. overlap만 안전하게 정리.
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

    audit_csv = output_dir / "bass_octave_anchor_audit.csv"
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
                "pyin_median",
                "pyin_note",
                "pyin_conf",
                "current_score",
                "lower_score",
                "lower_minus_current",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    final_csv = output_dir / "bass_final_notes_octave_anchor.csv"
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

    eprint(f"Octave anchor notes: {len(notes)}")
    eprint(f"Octave anchor changed: {changed}")
    eprint(f"Saved octave anchor audit: {audit_csv}")
    eprint(f"Saved octave anchor final CSV: {final_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-midi", type=int, default=28)
    parser.add_argument("--max-midi", type=int, default=55)
    parser.add_argument("--enable-pyin-octave-down", action="store_true")
    parser.add_argument("--enable-cqt-octave-down", action="store_true")
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    octave_anchor(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
        enable_pyin_octave_down=args.enable_pyin_octave_down,
        enable_cqt_octave_down=args.enable_cqt_octave_down,
    )


if __name__ == "__main__":
    main()
