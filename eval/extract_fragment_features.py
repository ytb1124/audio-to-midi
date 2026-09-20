#!/usr/bin/env python3
"""
생성된 최종 MIDI에서 인접한 같은 pitch 노트 쌍(경계)마다, 그 경계를
합쳐야 할지(fragmentation) 그대로 둬야 할지(진짜 재타격) 학습용 피처를 뽑는다.

라벨: boundary_time(=두 번째 노트의 시작) 근처(±50ms)에 정답 노트의 실제
onset이 있으면 1(keep separate, 진짜 재타격), 없으면 0(should merge, 가짜 분절).

이 정의는 "정답에 그 순간 진짜 새 노트가 시작되는가"를 직접 묻는 것이라
RMS attack 유무보다 더 직접적인 정답 신호다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

MAX_GAP_SEC = 0.10
ONSET_MATCH_TOL = 0.05


def load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
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


def cqt_at(C, hop, sr, base_midi, pitch, t0, t1):
    idx = int(round(pitch - base_midi))
    if idx < 0 or idx >= C.shape[0]:
        return 0.0
    a = max(0, int(t0 * sr / hop))
    b = min(C.shape[1], int(t1 * sr / hop) + 1)
    if b <= a:
        return 0.0
    return float(np.median(C[idx, a:b]))


def build_rms(y, sr):
    import librosa

    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=1024)[0]
    return rms, hop


def rms_window(rms, hop, sr, t0, t1, fn):
    a = max(0, int(t0 * sr / hop))
    b = min(len(rms), int(t1 * sr / hop) + 1)
    if b <= a:
        return 0.0
    return float(fn(rms[a:b]))


def build_onset_env(y, sr):
    import librosa

    hop = 256
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        y_for = 0.65 * y_harm + 0.35 * y_perc
    except Exception:
        y_for = y
    env = librosa.onset.onset_strength(y=y_for, sr=sr, hop_length=hop, aggregate=np.median, fmax=2200)
    return env, hop


def onset_env_at(env, hop, sr, t):
    idx = int(round(t * sr / hop))
    idx = max(0, min(len(env) - 1, idx))
    return float(env[idx])


def load_notes(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        [(n.start, n.end, n.pitch, n.velocity) for inst in pm.instruments for n in inst.notes if not inst.is_drum]
    )
    return notes


def extract_boundaries(audio_path, gen_midi_path, ref_midi_path, song_id):
    y, sr = load_audio(audio_path)
    C, cqt_hop, base_midi = build_cqt(y, sr)
    rms, rms_hop = build_rms(y, sr)
    onset_env, onset_hop = build_onset_env(y, sr)

    gen_notes = load_notes(gen_midi_path)
    ref_notes = load_notes(ref_midi_path)
    ref_onsets = sorted(r[0] for r in ref_notes)

    rows = []
    for i in range(len(gen_notes) - 1):
        a_start, a_end, a_pitch, a_vel = gen_notes[i]
        b_start, b_end, b_pitch, b_vel = gen_notes[i + 1]

        if a_pitch != b_pitch:
            continue
        gap = b_start - a_end
        if gap > MAX_GAP_SEC or gap < -0.02:
            continue

        boundary = b_start

        has_ref_onset = any(abs(t - boundary) <= ONSET_MATCH_TOL for t in ref_onsets)
        label = int(has_ref_onset)

        pre_short_min = rms_window(rms, rms_hop, sr, boundary - 0.05, boundary - 0.005, np.min)
        post_short_max = rms_window(rms, rms_hop, sr, boundary + 0.005, boundary + 0.03, np.max)
        pre_long_min = rms_window(rms, rms_hop, sr, boundary - 0.15, boundary - 0.005, np.min)
        post_long_max = rms_window(rms, rms_hop, sr, boundary + 0.005, boundary + 0.08, np.max)

        rms_ratio_short = post_short_max / pre_short_min if pre_short_min > 1e-6 else (10.0 if post_short_max > 1e-6 else 1.0)
        rms_ratio_long = post_long_max / pre_long_min if pre_long_min > 1e-6 else (10.0 if post_long_max > 1e-6 else 1.0)

        cqt_pre = cqt_at(C, cqt_hop, sr, base_midi, a_pitch, boundary - 0.1, boundary - 0.01)
        cqt_post = cqt_at(C, cqt_hop, sr, base_midi, a_pitch, boundary + 0.01, boundary + 0.1)
        cqt_ratio = cqt_post / cqt_pre if cqt_pre > 1e-6 else (10.0 if cqt_post > 1e-6 else 1.0)

        onset_strength = onset_env_at(onset_env, onset_hop, sr, boundary)

        dur_a = a_end - a_start
        dur_b = b_end - b_start
        vel_ratio = b_vel / a_vel if a_vel > 0 else 1.0

        rows.append(
            {
                "song_id": song_id,
                "boundary": round(boundary, 6),
                "pitch": a_pitch,
                "gap": gap,
                "label": label,
                "dur_a": dur_a,
                "dur_b": dur_b,
                "dur_ratio": dur_b / dur_a if dur_a > 0 else 1.0,
                "vel_a": a_vel,
                "vel_b": b_vel,
                "vel_ratio": vel_ratio,
                "rms_pre_short": pre_short_min,
                "rms_post_short": post_short_max,
                "rms_ratio_short": rms_ratio_short,
                "rms_pre_long": pre_long_min,
                "rms_post_long": post_long_max,
                "rms_ratio_long": rms_ratio_long,
                "cqt_pre": cqt_pre,
                "cqt_post": cqt_post,
                "cqt_ratio": cqt_ratio,
                "onset_strength": onset_strength,
            }
        )

    return rows


def main():
    # gen_midi_path: 2026-07-13 온셋 스냅/시간정렬 수정이 반영된 최신 실행
    # 결과(snap3_*)로 갱신. ref_mid: bass_sample3만 시간 오프셋까지 보정된
    # _fullcorrected 사용 (extract_octave_features.py와 동일 사유 — 50ms
    # 허용오차 안에서도 +60ms 오프셋은 라벨을 상당 부분 오염시킴).
    files = [
        ("bass_test", "eval/ground_truth/bass_test.wav", "eval/runs/snap3_bass_test/bass_hybrid.mid", "eval/ground_truth/bass_test.mid"),
        ("bass_sample", "eval/ground_truth/bass_sample.wav", "eval/runs/snap2_bass_sample/bass_hybrid.mid", "eval/ground_truth/bass_sample.mid"),
        ("bass_sample2", "eval/ground_truth/bass_sample2.wav", "eval/runs/snap3_bass_sample2/bass_hybrid.mid", "eval/ground_truth/bass_sample2.mid"),
        ("bass_sample3", "eval/ground_truth/bass_sample3.wav", "eval/runs/snap3_bass_sample3/bass_hybrid.mid", "eval/ground_truth/bass_sample3_fullcorrected.mid"),
        ("bass_sample4", "eval/ground_truth/bass_sample4.wav", "eval/runs/snap3_bass_sample4/bass_hybrid.mid", "eval/ground_truth/bass_sample4.mid"),
        ("bass_sample5", "eval/ground_truth/bass_sample5.wav", "eval/runs/snap3_bass_sample5/bass_hybrid.mid", "eval/ground_truth/bass_sample5.mid"),
    ]

    all_rows = []
    for song_id, wav, gen_mid, ref_mid in files:
        print(f"Extracting fragment boundaries for {song_id}...", file=sys.stderr)
        rows = extract_boundaries(wav, gen_mid, ref_mid, song_id)
        all_rows.extend(rows)
        n_merge = sum(1 for r in rows if r["label"] == 0)
        n_keep = sum(1 for r in rows if r["label"] == 1)
        print(f"  {len(rows)} boundaries (merge={n_merge}, keep_separate={n_keep})", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    out_path = Path("eval/fragment_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
