from pathlib import Path
import argparse
from collections import Counter

import numpy as np
import librosa
import pretty_midi


def clamp(value, low, high):
    return max(low, min(high, value))


def smooth_array(x, window=5):
    if window <= 1:
        return x

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def estimate_velocity(rms, start_frame, end_frame):
    if len(rms) == 0:
        return 85

    lo = max(0, start_frame)
    hi = min(len(rms), max(start_frame + 3, end_frame))

    if hi <= lo:
        local = rms[lo] if lo < len(rms) else 0.2
    else:
        local = float(np.max(rms[lo:hi]))

    velocity = int(42 + local * 78)
    return int(clamp(velocity, 38, 120))


def detect_bass_onsets(y, sr, hop_length):
    """
    베이스의 바이브는 onset timing이 핵심.
    pYIN이 아니라 원본 오디오 어택을 기준으로 note를 만든다.
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    onset_env_1 = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    onset_env_2 = librosa.onset.onset_strength(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    onset_env = onset_env_1 * 0.65 + onset_env_2 * 0.35
    onset_env = smooth_array(onset_env, window=3)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.075,
        wait=2,
    )

    return np.array(sorted(set(onset_frames)))


def estimate_pitch_from_pyin(f0_midi, voiced_prob, start_frame, end_frame, low, high):
    """
    onset 구간 안에서 pitch를 추정.
    pYIN은 note boundary 생성에 쓰지 않고 pitch 후보로만 사용한다.
    """
    lo = max(0, start_frame)
    hi = min(len(f0_midi), end_frame)

    values = []

    for i in range(lo, hi):
        pitch = f0_midi[i]
        prob = voiced_prob[i]

        if pitch is None:
            continue

        if prob < 0.25:
            continue

        if low <= pitch <= high:
            values.append(pitch)

    if not values:
        return None

    counts = Counter(values)
    return counts.most_common(1)[0][0]


def estimate_pitch_from_cqt(cqt_mag, start_frame, end_frame, low, high):
    """
    pYIN이 실패했을 때 CQT 에너지로 fallback.
    베이스 음역에서 가장 강한 pitch bin을 고른다.
    """
    lo = max(0, start_frame)
    hi = min(cqt_mag.shape[1], end_frame)

    if hi <= lo:
        return None

    segment = cqt_mag[:, lo:hi]

    if segment.size == 0:
        return None

    energy = np.mean(segment, axis=1)

    if np.max(energy) <= 1e-8:
        return None

    best_bin = int(np.argmax(energy))
    pitch = low + best_bin

    if low <= pitch <= high:
        return pitch

    return None


def remove_too_close_onsets(onset_frames, rms, min_gap_frames=4):
    """
    너무 가까운 onset은 하나로 합친다.
    단, 더 에너지가 강한 쪽을 남긴다.
    """
    if len(onset_frames) == 0:
        return onset_frames

    kept = [int(onset_frames[0])]

    for frame in onset_frames[1:]:
        frame = int(frame)
        prev = kept[-1]

        if frame - prev < min_gap_frames:
            prev_energy = rms[prev] if prev < len(rms) else 0
            frame_energy = rms[frame] if frame < len(rms) else 0

            if frame_energy > prev_energy * 1.15:
                kept[-1] = frame
        else:
            kept.append(frame)

    return np.array(kept)


def fix_octave_context(pitches):
    """
    베이스에서 흔한 옥타브 튐을 문맥으로 보정.
    pitch를 막 바꾸지 않고, 이전/다음 음과 비교해 명확한 12도 튐만 보정.
    """
    if not pitches:
        return pitches

    fixed = pitches[:]

    for i, pitch in enumerate(fixed):
        if pitch is None:
            continue

        neighbors = []

        for j in [i - 2, i - 1, i + 1, i + 2]:
            if 0 <= j < len(fixed) and fixed[j] is not None:
                neighbors.append(fixed[j])

        if not neighbors:
            continue

        median_neighbor = int(round(np.median(neighbors)))

        if pitch - 12 >= 28 and abs((pitch - 12) - median_neighbor) + 3 < abs(pitch - median_neighbor):
            fixed[i] = pitch - 12
        elif pitch + 12 <= 60 and abs((pitch + 12) - median_neighbor) + 3 < abs(pitch - median_neighbor):
            fixed[i] = pitch + 12

    return fixed


def build_notes_from_onsets(
    onset_frames,
    times,
    rms,
    f0_midi,
    voiced_prob,
    cqt_mag,
    low,
    high,
    sr,
    hop_length,
):
    notes_data = []

    for idx, onset_frame in enumerate(onset_frames):
        start_frame = int(onset_frame)

        if idx < len(onset_frames) - 1:
            next_onset_frame = int(onset_frames[idx + 1])
        else:
            next_onset_frame = min(len(times) - 1, start_frame + int(1.2 * sr / hop_length))

        # pitch는 어택 직후 살짝 지난 구간에서 잡는 게 안정적
        pitch_start = start_frame + int(0.025 * sr / hop_length)
        pitch_end = min(next_onset_frame, start_frame + int(0.55 * sr / hop_length))

        if pitch_end <= pitch_start:
            pitch_end = min(len(times) - 1, start_frame + int(0.25 * sr / hop_length))

        pitch = estimate_pitch_from_pyin(
            f0_midi=f0_midi,
            voiced_prob=voiced_prob,
            start_frame=pitch_start,
            end_frame=pitch_end,
            low=low,
            high=high,
        )

        if pitch is None:
            pitch = estimate_pitch_from_cqt(
                cqt_mag=cqt_mag,
                start_frame=pitch_start,
                end_frame=pitch_end,
                low=low,
                high=high,
            )

        if pitch is None:
            continue

        start_time = float(times[start_frame])

        # 기본 sustain은 다음 어택 직전까지.
        # 베이스 groove를 살리려면 pYIN이 끊긴 지점이 아니라 다음 onset이 기준이어야 함.
        end_frame = max(start_frame + 3, next_onset_frame - 1)
        end_time = float(times[min(end_frame, len(times) - 1)])

        if end_time - start_time < 0.055:
            continue

        velocity = estimate_velocity(rms, start_frame, end_frame)

        notes_data.append(
            {
                "pitch": int(pitch),
                "start": start_time,
                "end": end_time,
                "velocity": velocity,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )

    # 옥타브 튐 보정
    pitches = [n["pitch"] for n in notes_data]
    fixed_pitches = fix_octave_context(pitches)

    for note, pitch in zip(notes_data, fixed_pitches):
        note["pitch"] = pitch

    return notes_data


def remove_weak_false_notes(notes_data):
    """
    너무 약하고 짧은 false onset 제거.
    단, 베이스 ghost note를 완전히 죽이지 않기 위해 조건은 보수적으로.
    """
    if not notes_data:
        return []

    velocities = sorted(n["velocity"] for n in notes_data)
    median_velocity = velocities[len(velocities) // 2]

    result = []

    for n in notes_data:
        duration = n["end"] - n["start"]

        if duration < 0.075 and n["velocity"] < median_velocity * 0.62:
            continue

        result.append(n)

    return result


def enforce_monophonic(notes_data):
    if not notes_data:
        return []

    notes_data = sorted(notes_data, key=lambda n: n["start"])
    fixed = []

    for i, n in enumerate(notes_data):
        start = n["start"]
        end = n["end"]

        if i < len(notes_data) - 1:
            next_start = notes_data[i + 1]["start"]
            if end > next_start - 0.012:
                end = max(start + 0.055, next_start - 0.012)

        if end > start:
            n = dict(n)
            n["end"] = end
            fixed.append(n)

    return fixed


def normalize_velocity(notes_data):
    if not notes_data:
        return notes_data

    velocities = sorted(n["velocity"] for n in notes_data)
    median = velocities[len(velocities) // 2]

    for n in notes_data:
        if n["velocity"] < median * 0.55:
            n["velocity"] = int(clamp(n["velocity"] * 1.2, 38, 127))
        elif n["velocity"] > median * 1.75:
            n["velocity"] = int(clamp(n["velocity"] * 0.9, 1, 118))

    return notes_data


def write_midi(notes_data, output_midi_path):
    out = pretty_midi.PrettyMIDI(initial_tempo=120)

    bass = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Electric Bass (finger)"),
        name="Onset Guided Bass",
        is_drum=False,
    )

    for n in notes_data:
        bass.notes.append(
            pretty_midi.Note(
                velocity=int(clamp(n["velocity"], 1, 127)),
                pitch=int(clamp(n["pitch"], 0, 127)),
                start=float(n["start"]),
                end=float(n["end"]),
            )
        )

    out.instruments.append(bass)
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(output_midi_path))


def bass_audio_to_midi(
    audio_path: Path,
    output_midi_path: Path,
    low: int = 28,
    high: int = 55,
    sr: int = 22050,
    hop_length: int = 256,
):
    print("Loading bass audio...")
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    print("Preparing audio features...")
    y_harmonic = librosa.effects.harmonic(y)

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=hop_length,
    )[0]
    rms = rms / (np.max(rms) + 1e-9)

    print("Detecting original audio onsets...")
    onset_frames = detect_bass_onsets(y, sr, hop_length)
    onset_frames = remove_too_close_onsets(onset_frames, rms, min_gap_frames=4)

    print("Tracking pitch for each onset...")
    fmin = librosa.midi_to_hz(low)
    fmax = librosa.midi_to_hz(high)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_harmonic,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=4096,
        hop_length=hop_length,
        fill_na=np.nan,
    )

    f0_midi = []
    for hz in f0:
        if np.isnan(hz):
            f0_midi.append(None)
        else:
            f0_midi.append(int(round(librosa.hz_to_midi(hz))))

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length,
    )

    print("Computing bass CQT fallback...")
    cqt = librosa.cqt(
        y_harmonic,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.midi_to_hz(low),
        n_bins=high - low + 1,
        bins_per_octave=12,
    )
    cqt_mag = np.abs(cqt)
    cqt_mag = cqt_mag / (np.max(cqt_mag) + 1e-9)

    notes_data = build_notes_from_onsets(
        onset_frames=onset_frames,
        times=times,
        rms=rms,
        f0_midi=f0_midi,
        voiced_prob=voiced_prob,
        cqt_mag=cqt_mag,
        low=low,
        high=high,
        sr=sr,
        hop_length=hop_length,
    )

    before_cleanup = len(notes_data)

    notes_data = remove_weak_false_notes(notes_data)
    notes_data = enforce_monophonic(notes_data)
    notes_data = normalize_velocity(notes_data)

    write_midi(notes_data, output_midi_path)

    print("Done.")
    print(f"Audio: {audio_path}")
    print(f"Output MIDI: {output_midi_path}")
    print(f"Detected onsets: {len(onset_frames)}")
    print(f"Notes before cleanup: {before_cleanup}")
    print(f"Final notes: {len(notes_data)}")
    print(f"Range: {low}-{high}")
    print("Mode: bass onset-guided transcription, monophonic, sustain to next onset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output_midi", type=Path)
    parser.add_argument("--low", type=int, default=28)
    parser.add_argument("--high", type=int, default=55)

    args = parser.parse_args()

    bass_audio_to_midi(
        audio_path=args.audio,
        output_midi_path=args.output_midi,
        low=args.low,
        high=args.high,
    )


if __name__ == "__main__":
    main()
