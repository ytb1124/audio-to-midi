#!/usr/bin/env python3
"""
정답 MIDI 노트마다, 후보 옥타브들(true_pitch + {-24,-12,0,12,24})에 대해
CQT/pYIN/Basic Pitch confidence 피처를 뽑아서 학습용 CSV로 저장한다.

label: 1 = 이 후보가 정답 pitch, 0 = 아님 (note당 정확히 1개만 1)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

MIN_MIDI = 24
MAX_MIDI = 72
OFFSETS = [-24, -12, 0, 12, 24]


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


def cqt_energy(C, hop, sr, base_midi, pitch, start, end):
    idx = int(round(pitch - base_midi))
    if idx < 0 or idx >= C.shape[0]:
        return 0.0
    a = max(0, int((start + 0.015) * sr / hop))
    b = min(C.shape[1], int(min(end, start + 0.30) * sr / hop) + 1)
    if b <= a:
        return 0.0
    return float(np.median(C[idx, a:b]))


def compute_pyin(y, sr):
    import librosa

    hop = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz("E1"), fmax=librosa.note_to_hz("C5"),
        sr=sr, frame_length=2048, hop_length=hop, fill_na=np.nan,
    )
    midi = np.full(len(f0), np.nan)
    valid = np.isfinite(f0)
    with np.errstate(divide="ignore"):
        midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)
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


def bp_note_conf(note_probs, frame_rate, pitch, start, end):
    idx = pitch - 21
    if idx < 0 or idx >= note_probs.shape[1]:
        return 0.0
    a = max(0, int(start * frame_rate))
    b = min(note_probs.shape[0], int(end * frame_rate) + 1)
    if b <= a:
        return 0.0
    return float(np.mean(note_probs[a:b, idx]))


def compute_crepe(y, sr):
    import torch
    import torchcrepe

    target_sr = 16000
    import librosa as _librosa

    y16 = _librosa.resample(y, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y
    audio = torch.tensor(y16, dtype=torch.float32).unsqueeze(0)
    hop = 160

    pitch, periodicity = torchcrepe.predict(
        audio, target_sr, hop_length=hop, fmin=27.5, fmax=1000.0,
        model="tiny", batch_size=1024, return_periodicity=True, device="cpu",
    )
    pitch = pitch.squeeze(0).numpy()
    periodicity = periodicity.squeeze(0).numpy()
    frame_rate = target_sr / hop
    return pitch, periodicity, frame_rate


def crepe_periodicity_at(crepe_pitch, crepe_periodicity, crepe_rate, pitch, start, end):
    """candidate pitch 주파수 근처(半음 이내)에서 CREPE가 보고한 periodicity(신뢰도)
    중앙값. CREPE 자체 추정 pitch가 candidate와 가까울 때만 그 confidence를 인정."""
    a = max(0, int(start * crepe_rate))
    b = min(len(crepe_pitch), int(end * crepe_rate) + 1)
    if b <= a:
        return 0.0

    target_hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    seg_pitch = crepe_pitch[a:b]
    seg_per = crepe_periodicity[a:b]

    cents = 1200.0 * np.log2(np.maximum(seg_pitch, 1e-6) / target_hz)
    close = np.abs(cents) <= 50.0  # ±50 cents 이내

    if not np.any(close):
        return 0.0
    return float(np.median(seg_per[close]))


def bp_onset_conf(onset_probs, frame_rate, pitch, start):
    idx = pitch - 21
    if idx < 0 or idx >= onset_probs.shape[1]:
        return 0.0
    a = max(0, int(start * frame_rate) - 2)
    b = min(onset_probs.shape[0], int(start * frame_rate) + 3)
    if b <= a:
        return 0.0
    return float(np.max(onset_probs[a:b, idx]))


NEIGHBOR_WINDOW_SEC = 2.5


def extract_features(audio_path, midi_path, song_id, skip_slow=True):
    """skip_slow=True: CREPE/BasicPitch confidence 스킵 (배포 모델 기준
    permutation importance ~0으로 이미 확인됨 — train_octave_classifier.py의
    DEPLOY_FEATURE_COLS 참고). 재검증 목적이 아니면 속도를 위해 기본 스킵."""
    y, sr = load_audio(audio_path)
    C, cqt_hop, base_midi = build_cqt(y, sr)
    pyin_midi, pyin_prob, pyin_hop = compute_pyin(y, sr)

    if skip_slow:
        note_probs = onset_probs = None
        frame_rate = 0.0
        crepe_pitch = crepe_periodicity = None
        crepe_rate = 0.0
    else:
        from basic_pitch.inference import predict

        model_output, _, _ = predict(str(audio_path))
        note_probs = model_output["note"]
        onset_probs = model_output["onset"]
        duration = len(y) / sr
        frame_rate = note_probs.shape[0] / duration
        crepe_pitch, crepe_periodicity, crepe_rate = compute_crepe(y, sr)

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted(
        [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes if not inst.is_drum]
    )

    # 이웃 노트 컨텍스트용: 노트별 raw harmonic score를 캐시해서 재사용
    # (자기 자신을 뺀 근처 같은 pitch class 노트들의 '실제 발성 pitch class'
    # 기준 harmonic score 합만 필요하므로, note index -> pitch class -> score)
    def raw_harmonic(pitch, start, end):
        e0 = cqt_energy(C, cqt_hop, sr, base_midi, pitch, start, end)
        e12 = cqt_energy(C, cqt_hop, sr, base_midi, pitch + 12, start, end)
        e19 = cqt_energy(C, cqt_hop, sr, base_midi, pitch + 19, start, end)
        e24 = cqt_energy(C, cqt_hop, sr, base_midi, pitch + 24, start, end)
        return e0 + 0.42 * e12 + 0.22 * e19 + 0.12 * e24

    rows = []
    for i, (start, end, true_pitch) in enumerate(notes):
        py_med, py_conf = pyin_stats(pyin_midi, pyin_prob, pyin_hop, sr, start, end)
        pitch_class = true_pitch % 12

        # 근처(±NEIGHBOR_WINDOW_SEC) 같은 pitch class 노트들 (자기 자신 제외)
        neighbors = [
            (ns, ne, np_) for j, (ns, ne, np_) in enumerate(notes)
            if j != i and np_ % 12 == pitch_class and abs(ns - start) <= NEIGHBOR_WINDOW_SEC
        ]

        for offset in OFFSETS:
            cand = true_pitch + offset
            if cand < MIN_MIDI or cand > MAX_MIDI:
                continue

            e0 = cqt_energy(C, cqt_hop, sr, base_midi, cand, start, end)
            e12 = cqt_energy(C, cqt_hop, sr, base_midi, cand + 12, start, end)
            e_minus12 = cqt_energy(C, cqt_hop, sr, base_midi, cand - 12, start, end)
            e19 = cqt_energy(C, cqt_hop, sr, base_midi, cand + 19, start, end)
            e24 = cqt_energy(C, cqt_hop, sr, base_midi, cand + 24, start, end)
            harmonic = e0 + 0.42 * e12 + 0.22 * e19 + 0.12 * e24

            if skip_slow:
                bp_conf = bp_conf_up = bp_conf_down = bp_onset = 0.0
                crepe_per = crepe_per_up = crepe_per_down = 0.0
            else:
                bp_conf = bp_note_conf(note_probs, frame_rate, cand, start, end)
                bp_conf_up = bp_note_conf(note_probs, frame_rate, cand + 12, start, end)
                bp_conf_down = bp_note_conf(note_probs, frame_rate, cand - 12, start, end)
                bp_onset = bp_onset_conf(onset_probs, frame_rate, cand, start)
                crepe_per = crepe_periodicity_at(crepe_pitch, crepe_periodicity, crepe_rate, cand, start, end)
                crepe_per_up = crepe_periodicity_at(crepe_pitch, crepe_periodicity, crepe_rate, cand + 12, start, end)
                crepe_per_down = crepe_periodicity_at(crepe_pitch, crepe_periodicity, crepe_rate, cand - 12, start, end)

            pyin_dist = abs(py_med - cand) if py_med is not None else 99.0

            # 이웃 컨텍스트: 이 후보와 같은 옥타브(cand)를 기준으로, 근처 같은
            # pitch class 노트들이 이 옥타브를 얼마나 지지하는지(각자 자기
            # harmonic score 기준)를 집계. "이웃들도 다 이 옥타브를 좋아하는가"
            neighbor_support = 0.0
            neighbor_count = len(neighbors)
            for ns, ne, np_ in neighbors:
                neighbor_support += raw_harmonic(cand, ns, ne)
            neighbor_support_mean = neighbor_support / neighbor_count if neighbor_count else 0.0

            rows.append(
                {
                    "song_id": song_id,
                    "note_start": start,
                    "note_end": end,
                    "true_pitch": true_pitch,
                    "candidate": cand,
                    "offset": offset,
                    "label": int(cand == true_pitch),
                    "cqt_e0": e0,
                    "cqt_e12": e12,
                    "cqt_e_minus12": e_minus12,
                    "cqt_e19": e19,
                    "cqt_e24": e24,
                    "cqt_harmonic": harmonic,
                    "bp_note_conf": bp_conf,
                    "bp_note_conf_up": bp_conf_up,
                    "bp_note_conf_down": bp_conf_down,
                    "bp_onset_conf": bp_onset,
                    "pyin_dist": pyin_dist,
                    "pyin_conf": py_conf,
                    "note_duration": end - start,
                    "neighbor_support_mean": neighbor_support_mean,
                    "neighbor_count": neighbor_count,
                    "crepe_periodicity": crepe_per,
                    "crepe_periodicity_up": crepe_per_up,
                    "crepe_periodicity_down": crepe_per_down,
                }
            )

    return rows


def main():
    # 옥타브 시프트 보정된 정답 사용 (eval/apply_octave_shift_correction.py로 생성).
    # 전체 파일 단위 옥타브 밀림은 사용자 편집으로 해결 가능한 낮은 우선순위
    # 문제라 학습 라벨에서 제외 — 산발적 오류만 학습 대상으로 남긴다.
    # bass_sample3만 온셋 교차상관으로 감지된 +60ms 전역 시간 오프셋도 추가
    # 보정(_fullcorrected.mid, 2026-07-13) — 이게 없으면 오디오 피처 창 자체가
    # 실제 어택보다 60ms 앞을 보게 되어 학습 신호가 오염됨. score.py의
    # detect_global_time_offset로 6곡 전체 재확인: 나머지는 <=20ms(측정
    # 해상도 잡음 수준, 50ms 온셋 허용오차 내)라 보정 불필요.
    files = [
        ("bass_test", "eval/ground_truth/bass_test.wav", "eval/ground_truth/bass_test.mid"),
        ("bass_sample", "eval/ground_truth/bass_sample.wav", "eval/ground_truth/bass_sample_octcorrected.mid"),
        ("bass_sample2", "eval/ground_truth/bass_sample2.wav", "eval/ground_truth/bass_sample2_octcorrected.mid"),
        ("bass_sample3", "eval/ground_truth/bass_sample3.wav", "eval/ground_truth/bass_sample3_fullcorrected.mid"),
        ("bass_sample4", "eval/ground_truth/bass_sample4.wav", "eval/ground_truth/bass_sample4_octcorrected.mid"),
        ("bass_sample5", "eval/ground_truth/bass_sample5.wav", "eval/ground_truth/bass_sample5_octcorrected.mid"),
    ]

    all_rows = []
    for song_id, wav, mid in files:
        print(f"Extracting features for {song_id}...", file=sys.stderr)
        rows = extract_features(wav, mid, song_id)
        all_rows.extend(rows)
        print(f"  {len(rows)} candidate rows", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    out_path = Path("eval/octave_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
