from pathlib import Path
import argparse
import pretty_midi


def clamp(value, low, high):
    return max(low, min(high, value))


def collect_bass_candidates(midi, low=28, high=55):
    """
    Bass guitar core range:
    E1 = 28
    G3 = 55

    high를 너무 높게 열면 배음/기타/신스 노이즈가 들어온다.
    """
    notes = []

    for inst in midi.instruments:
        if inst.is_drum:
            continue

        for note in inst.notes:
            if note.end <= note.start:
                continue

            if low <= note.pitch <= high:
                notes.append(
                    pretty_midi.Note(
                        velocity=int(clamp(note.velocity, 1, 127)),
                        pitch=int(note.pitch),
                        start=float(note.start),
                        end=float(note.end),
                    )
                )

    return sorted(notes, key=lambda n: (n.start, n.pitch))


def remove_tiny_ghosts(notes):
    result = []

    for note in notes:
        duration = note.end - note.start

        # 클릭/잡음 수준
        if duration < 0.035:
            continue

        # 짧고 약한 ghost
        if duration < 0.075 and note.velocity < 42:
            continue

        result.append(note)

    return sorted(result, key=lambda n: (n.start, n.pitch))


def merge_same_pitch_duplicates(notes, window=0.04):
    by_pitch = {}

    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)

    result = []

    for pitch, pitch_notes in by_pitch.items():
        pitch_notes = sorted(pitch_notes, key=lambda n: n.start)
        kept = []

        for note in pitch_notes:
            merged = False

            for existing in kept:
                if abs(note.start - existing.start) <= window:
                    existing.end = max(existing.end, note.end)
                    existing.velocity = max(existing.velocity, note.velocity)
                    merged = True
                    break

            if not merged:
                kept.append(note)

        result.extend(kept)

    return sorted(result, key=lambda n: (n.start, n.pitch))


def remove_octave_harmonic_candidates(notes, window=0.10):
    """
    베이스에서 흔한 문제:
    실제 저음 + 옥타브 위 배음이 같이 잡힘.

    같은 pitch class가 가까운 시간에 있으면 낮은 쪽을 우선.
    """
    result = []

    for note in notes:
        duration = note.end - note.start

        lower_neighbors = [
            n for n in notes
            if n is not note
            and n.pitch % 12 == note.pitch % 12
            and n.pitch < note.pitch
            and abs(n.start - note.start) <= window
        ]

        if lower_neighbors:
            anchor = max(lower_neighbors, key=lambda n: (n.velocity, n.end - n.start))
            anchor_duration = anchor.end - anchor.start

            if note.pitch - anchor.pitch in (12, 24):
                if note.velocity <= anchor.velocity * 1.05 or duration <= anchor_duration * 0.9:
                    continue

        result.append(note)

    return sorted(result, key=lambda n: (n.start, n.pitch))


def group_near_onsets(notes, window=0.085):
    """
    거의 같은 어택에서 나온 후보들을 하나의 그룹으로 묶는다.
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n.start)
    groups = []
    current = [notes[0]]

    for note in notes[1:]:
        if note.start - current[0].start <= window:
            current.append(note)
        else:
            groups.append(current)
            current = [note]

    groups.append(current)

    return groups


def choose_best_bass_note(group):
    """
    베이스 후보 선택 기준:
    - velocity가 강한 음
    - duration이 어느 정도 있는 음
    - 낮은 음 우선
    - 너무 높은 배음 후보는 감점
    """
    def score(note):
        duration = note.end - note.start

        velocity_score = note.velocity * 2.2
        duration_score = min(duration, 1.2) * 55.0
        low_score = max(0, 58 - note.pitch) * 1.6

        # 너무 낮은 잡음성 후보 방지
        if note.pitch < 28:
            low_score -= 30

        return velocity_score + duration_score + low_score

    return max(group, key=score)


def collapse_to_monophonic_onsets(notes, onset_window=0.085):
    """
    같은 순간의 여러 후보 중 하나만 남긴다.
    """
    groups = group_near_onsets(notes, onset_window)
    selected = [choose_best_bass_note(group) for group in groups]

    return sorted(selected, key=lambda n: n.start)


def remove_too_close_retriggers(notes, min_gap=0.055):
    """
    베이스에서 너무 촘촘한 재트리거는 대부분 잘못 잡힌 경우가 많다.
    단, 빠른 연주는 살리기 위해 더 강한 후보만 남긴다.
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n.start)
    result = [notes[0]]

    for note in notes[1:]:
        prev = result[-1]
        gap = note.start - prev.start

        if gap < min_gap:
            prev_score = prev.velocity + (prev.end - prev.start) * 30
            note_score = note.velocity + (note.end - note.start) * 30

            if note_score > prev_score * 1.15:
                result[-1] = note
            else:
                continue
        else:
            result.append(note)

    return sorted(result, key=lambda n: n.start)


def merge_same_pitch_fragments(notes, gap=0.12):
    """
    같은 pitch가 짧게 끊긴 경우 연결.
    """
    by_pitch = {}

    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)

    merged_all = []

    for pitch, pitch_notes in by_pitch.items():
        pitch_notes = sorted(pitch_notes, key=lambda n: n.start)
        merged = []

        for note in pitch_notes:
            if not merged:
                merged.append(note)
                continue

            prev = merged[-1]
            gap_to_prev = note.start - prev.end

            if 0 <= gap_to_prev <= gap:
                prev.end = max(prev.end, note.end)
                prev.velocity = max(prev.velocity, note.velocity)
            else:
                merged.append(note)

        merged_all.extend(merged)

    return sorted(merged_all, key=lambda n: n.start)


def enforce_strict_monophonic(notes, min_duration=0.06, end_gap=0.012):
    """
    최종적으로 어떤 경우에도 동시에 두 베이스 음이 울리지 않게 만든다.
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n.start)
    fixed = []

    for i, note in enumerate(notes):
        if i < len(notes) - 1:
            next_note = notes[i + 1]

            if note.end > next_note.start - end_gap:
                note.end = max(note.start + min_duration, next_note.start - end_gap)

        if note.end > note.start:
            fixed.append(note)

    return fixed


def repair_bass_sustain(notes):
    """
    베이스 sustain은 길게 늘리는 게 아니라,
    다음 노트와 groove를 기준으로 정리해야 한다.

    원칙:
    - 짧게 끊긴 음은 조금 보강
    - 다음 노트를 침범하지 않음
    - 긴 음은 과하게 늘리지 않음
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n.start)

    for i, note in enumerate(notes):
        duration = note.end - note.start

        if duration < 0.12:
            target_end = note.end + 0.16
        elif duration < 0.35:
            target_end = note.end + 0.10
        else:
            target_end = note.end + 0.04

        if i < len(notes) - 1:
            next_note = notes[i + 1]
            target_end = min(target_end, next_note.start - 0.012)

        note.end = max(note.end, target_end)

    return enforce_strict_monophonic(notes)


def normalize_velocity(notes):
    if not notes:
        return notes

    velocities = sorted(n.velocity for n in notes)
    median = velocities[len(velocities) // 2]

    for note in notes:
        if note.velocity < median * 0.55:
            note.velocity = int(clamp(note.velocity * 1.25, 35, 127))
        elif note.velocity > median * 1.75:
            note.velocity = int(clamp(note.velocity * 0.88, 1, 118))

    return notes


def cleanup_bass(input_midi: Path, output_midi: Path, low: int, high: int):
    midi = pretty_midi.PrettyMIDI(str(input_midi))
    before = sum(len(inst.notes) for inst in midi.instruments)

    notes = collect_bass_candidates(midi, low=low, high=high)

    raw_bass_count = len(notes)

    notes = remove_tiny_ghosts(notes)
    notes = merge_same_pitch_duplicates(notes)
    notes = remove_octave_harmonic_candidates(notes)
    notes = collapse_to_monophonic_onsets(notes)
    notes = remove_too_close_retriggers(notes)
    notes = merge_same_pitch_fragments(notes)
    notes = repair_bass_sustain(notes)
    notes = enforce_strict_monophonic(notes)
    notes = normalize_velocity(notes)

    out = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())

    bass = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Electric Bass (finger)"),
        name="Clean Monophonic Bass",
        is_drum=False,
    )

    bass.notes = notes
    out.instruments.append(bass)

    output_midi.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(output_midi))

    print("Done.")
    print(f"Input MIDI: {input_midi}")
    print(f"Output MIDI: {output_midi}")
    print(f"All notes before: {before}")
    print(f"Bass candidates before cleanup: {raw_bass_count}")
    print(f"Bass notes after cleanup: {len(notes)}")
    print(f"Range: {low}-{high}")
    print("Mode: bass cleanup v2, strict monophonic, no quantize")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_midi", type=Path)
    parser.add_argument("output_midi", type=Path)
    parser.add_argument("--low", type=int, default=28)
    parser.add_argument("--high", type=int, default=55)

    args = parser.parse_args()

    cleanup_bass(
        input_midi=args.input_midi,
        output_midi=args.output_midi,
        low=args.low,
        high=args.high,
    )


if __name__ == "__main__":
    main()
