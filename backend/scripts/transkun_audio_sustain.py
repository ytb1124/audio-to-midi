from pathlib import Path
import argparse
import numpy as np
import librosa
import pretty_midi


def midi_to_cqt_bin(midi_pitch: int, bins_per_semitone: int = 3) -> int:
    """
    A0 = MIDI 21을 CQT bin 0으로 보고,
    semitone당 bins_per_semitone개 bin을 사용.
    """
    return int(round((midi_pitch - 21) * bins_per_semitone))


def smooth_envelope(env: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return env

    kernel = np.ones(window) / window
    return np.convolve(env, kernel, mode="same")


def time_to_frame(time: float, sr: int, hop_length: int) -> int:
    return int(round(time * sr / hop_length))


def frame_to_time(frame: int, sr: int, hop_length: int) -> float:
    return frame * hop_length / sr


def analyze_pitch_energy(
    audio_path: Path,
    sr: int = 22050,
    hop_length: int = 512,
    bins_per_octave: int = 36,
):
    """
    피아노 음역 A0~C8 기준 CQT 에너지 분석.
    bins_per_octave=36이면 semitone당 3개 bin.
    """
    print("Loading audio...")
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    print("Computing CQT...")
    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.midi_to_hz(21),
        n_bins=88 * 3,
        bins_per_octave=bins_per_octave,
    )

    mag = np.abs(cqt)
    mag = mag / (np.max(mag) + 1e-9)

    return mag, sr, hop_length


def get_pitch_envelope(
    mag: np.ndarray,
    midi_pitch: int,
    bins_per_semitone: int = 3,
    radius: int = 1,
):
    center = midi_to_cqt_bin(midi_pitch, bins_per_semitone)

    lo = max(0, center - radius)
    hi = min(mag.shape[0], center + radius + 1)

    env = np.max(mag[lo:hi, :], axis=0)
    env = smooth_envelope(env, window=5)

    return env


def find_audio_guided_end(
    note,
    env: np.ndarray,
    sr: int,
    hop_length: int,
    max_extend: float,
    min_extend: float,
    threshold_ratio: float,
    hold_frames: int,
):
    start_frame = time_to_frame(note.start, sr, hop_length)
    midi_end_frame = time_to_frame(note.end, sr, hop_length)

    search_start = midi_end_frame
    search_end = time_to_frame(note.end + max_extend, sr, hop_length)
    search_end = min(search_end, len(env) - 1)

    if search_start >= len(env):
        return note.end

    # note 시작 주변~기존 end까지의 에너지 피크를 기준으로 decay 판단
    local_start = max(0, start_frame - 2)
    local_end = min(len(env), max(midi_end_frame + 1, start_frame + 4))

    local_peak = float(np.max(env[local_start:local_end])) if local_end > local_start else 0.0

    if local_peak <= 1e-6:
        return note.end + min_extend

    threshold = local_peak * threshold_ratio

    below_count = 0
    found_frame = None

    for frame in range(search_start, search_end):
        if env[frame] < threshold:
            below_count += 1
        else:
            below_count = 0

        if below_count >= hold_frames:
            found_frame = frame - hold_frames
            break

    if found_frame is None:
        candidate_end = frame_to_time(search_end, sr, hop_length)
    else:
        candidate_end = frame_to_time(found_frame, sr, hop_length)

    # 너무 조금만 늘어나는 것을 방지
    candidate_end = max(candidate_end, note.end + min_extend)

    return candidate_end


def process_midi(
    audio_path: Path,
    input_midi_path: Path,
    output_midi_path: Path,
    max_extend: float = 1.15,
    min_extend: float = 0.08,
    threshold_ratio: float = 0.22,
    hold_frames: int = 4,
):
    mag, sr, hop_length = analyze_pitch_energy(audio_path)

    midi = pretty_midi.PrettyMIDI(str(input_midi_path))
    out = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())

    notes = []

    for inst in midi.instruments:
        if inst.is_drum:
            continue

        for note in inst.notes:
            if 21 <= note.pitch <= 108 and note.end > note.start:
                notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    )
                )

    notes.sort(key=lambda n: (n.pitch, n.start))

    by_pitch = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)

    processed = []
    extensions = []

    print("Applying audio-guided sustain...")

    for pitch, pitch_notes in by_pitch.items():
        pitch_env = get_pitch_envelope(mag, pitch)

        pitch_notes = sorted(pitch_notes, key=lambda n: n.start)

        for i, note in enumerate(pitch_notes):
            original_end = note.end

            new_end = find_audio_guided_end(
                note=note,
                env=pitch_env,
                sr=sr,
                hop_length=hop_length,
                max_extend=max_extend,
                min_extend=min_extend,
                threshold_ratio=threshold_ratio,
                hold_frames=hold_frames,
            )

            # 같은 pitch의 다음 노트를 침범하지 않음
            if i < len(pitch_notes) - 1:
                next_note = pitch_notes[i + 1]
                if new_end > next_note.start:
                    new_end = max(note.start + 0.035, next_note.start - 0.008)

            note.end = max(note.end, new_end)

            if note.end > note.start:
                processed.append(note)
                extensions.append(note.end - original_end)

    processed.sort(key=lambda n: n.start)

    piano = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"),
        name="Transkun Audio-Guided Sustain",
        is_drum=False,
    )
    piano.notes = processed

    out.instruments.append(piano)
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(output_midi_path))

    if extensions:
        avg_ext = sum(extensions) / len(extensions)
        max_ext = max(extensions)
    else:
        avg_ext = 0
        max_ext = 0

    print("Done.")
    print(f"Audio: {audio_path}")
    print(f"Input MIDI: {input_midi_path}")
    print(f"Output MIDI: {output_midi_path}")
    print(f"Notes: {len(processed)}")
    print(f"Average extension: {avg_ext:.3f}s")
    print(f"Max extension: {max_ext:.3f}s")
    print("Mode: pitch/start preserved, audio-guided note-end repair")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("input_midi", type=Path)
    parser.add_argument("output_midi", type=Path)

    parser.add_argument("--max-extend", type=float, default=1.15)
    parser.add_argument("--min-extend", type=float, default=0.08)
    parser.add_argument("--threshold-ratio", type=float, default=0.22)
    parser.add_argument("--hold-frames", type=int, default=4)

    args = parser.parse_args()

    process_midi(
        audio_path=args.audio,
        input_midi_path=args.input_midi,
        output_midi_path=args.output_midi,
        max_extend=args.max_extend,
        min_extend=args.min_extend,
        threshold_ratio=args.threshold_ratio,
        hold_frames=args.hold_frames,
    )


if __name__ == "__main__":
    main()
