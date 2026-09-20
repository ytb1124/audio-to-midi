#!/usr/bin/env python3
"""
Note proposal pool(규칙 적용 전 후보열, bass_candidates.csv)의 각 후보에
대해 note-validity + onset-type 라벨과 오디오 피처를 뽑는다.
(사용자 제안 아키텍처의 "note verification" 헤드 학습용, 2026-07-13)

라벨:
  validity = 1  후보 onset이 정답 노트와 ±50ms 안에서 매칭 (진짜 노트)
           = 0  매칭 없음 (no_onset — 유령/가짜 후보)
  onset_type (validity=1인 경우만):
    reattack  매칭된 정답 노트가 직전 정답 노트와 같은 pitch, gap<0.15s
    legato    다른 pitch인데 직전 노트가 연속(작은 gap + 약한 어택 전이)
    pluck     그 외 (신선한 어택)

이 라벨의 핵심 장점: **validity는 정답 MIDI 매칭으로 완벽하게 깨끗하다.**
fragment 라벨(6곡 27개)과 달리 no_onset 예시가 6곡에 걸쳐 수백 개 —
지금까지 규칙(유령 꾸밈음 제거 + 누락 복구)으로 씨름하던 문제를 하나의
학습 헤드로 통합한다. onset_type은 pluck/legato를 오디오 어택으로 라벨링해
다소 순환적이므로 validity 대비 보조 신호로만 취급.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

TOL = 0.05
REATTACK_GAP = 0.15


def load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y, sr


def build_onset_env(y, sr):
    import librosa

    hop = 256
    try:
        y_h, y_p = librosa.effects.hpss(y)
        y_for = 0.65 * y_h + 0.35 * y_p
    except Exception:
        y_for = y
    env = librosa.onset.onset_strength(y=y_for, sr=sr, hop_length=hop, aggregate=np.median, fmax=2200)
    return env, hop


def build_rms(y, sr):
    import librosa

    hop = 256
    return librosa.feature.rms(y=y, hop_length=hop, frame_length=1024)[0], hop


def build_cqt(y, sr):
    import librosa

    hop = 256
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
                           n_bins=72, bins_per_octave=12))
    C = np.log1p(10.0 * C)
    return C, hop, 24


def compute_pyin(y, sr):
    import librosa

    hop = 256
    f0, _, vprob = librosa.pyin(y, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C5"),
                                sr=sr, frame_length=2048, hop_length=hop, fill_na=np.nan)
    midi = np.full(len(f0), np.nan)
    valid = np.isfinite(f0)
    with np.errstate(divide="ignore"):
        midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)
    return midi, np.nan_to_num(vprob, nan=0.0), hop


def hf_flux(y, sr):
    """고주파 스펙트럼 flux (pluck/pick 어택은 HF 상승이 큼)."""
    import librosa

    hop = 256
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf = S[freqs >= 800, :]
    flux = np.sqrt(np.sum(np.maximum(np.diff(hf, axis=1), 0) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])
    return flux, hop


def win_stat(arr, hop, sr, t0, t1, fn, default=0.0):
    a = max(0, int(t0 * sr / hop))
    b = min(len(arr), int(t1 * sr / hop) + 1)
    if b <= a:
        return default
    return float(fn(arr[a:b]))


def cqt_harmonic(C, hop, sr, base, pitch, t0, t1):
    def e(p):
        idx = int(round(p - base))
        if idx < 0 or idx >= C.shape[0]:
            return 0.0
        a = max(0, int(t0 * sr / hop)); b = min(C.shape[1], int(t1 * sr / hop) + 1)
        return float(np.median(C[idx, a:b])) if b > a else 0.0
    return e(pitch) + 0.42 * e(pitch + 12) + 0.22 * e(pitch + 19) + 0.12 * e(pitch + 24)


def cqt_pitch_row(C, hop, sr, base, pitch):
    idx = int(round(pitch - base))
    if idx < 0 or idx >= C.shape[0]:
        return None
    return C[idx]


def decay_time(row, hop, sr, t_start, t_cap, ratio=0.25):
    """pitch bin 에너지가 어택 피크의 ratio 아래로 처음 떨어지는 시각까지의
    경과 시간. 다음 온셋(t_cap)까지만 탐색. 없으면 t_cap까지의 길이."""
    if row is None:
        return 0.0
    a = int(t_start * sr / hop)
    b = min(len(row), int(t_cap * sr / hop) + 1)
    if b <= a + 2:
        return 0.0
    seg = row[a:b]
    peak = float(seg[: max(3, int(0.08 * sr / hop))].max())
    if peak < 0.3:
        return 0.0
    below = np.nonzero(seg < ratio * peak)[0]
    if len(below):
        return float(below[0]) * hop / sr
    return (b - a) * hop / sr


def extract(song_id, cand_csv, audio_path, gt_mid):
    y, sr = load_audio(audio_path)
    env, ehop = build_onset_env(y, sr)
    rms, rhop = build_rms(y, sr)
    C, chop, base = build_cqt(y, sr)
    pyin_midi, pyin_prob, phop = compute_pyin(y, sr)
    flux, fhop = hf_flux(y, sr)

    env_lo, env_hi = np.percentile(env, 10), np.percentile(env, 95)
    env_sc = max(env_hi - env_lo, 1e-9)
    flux_hi = max(np.percentile(flux, 95), 1e-9)

    cand = pd.read_csv(cand_csv).sort_values("time").reset_index(drop=True)
    pm = pretty_midi.PrettyMIDI(gt_mid)
    gt = sorted((n.start, n.pitch) for i in pm.instruments for n in i.notes if not i.is_drum)
    gt_t = np.array([g[0] for g in gt])
    gt_p = np.array([g[1] for g in gt])

    def pyin_win(t0, t1, pitch):
        a = max(0, int(t0 * sr / phop)); b = min(len(pyin_midi), int(t1 * sr / phop) + 1)
        if b <= a:
            return 0.0, 99.0, 0.0
        seg = pyin_midi[a:b]; valid = np.isfinite(seg)
        vf = float(valid.sum()) / len(seg)
        if not valid.any():
            return vf, 99.0, 0.0
        v = seg[valid]
        med = float(np.nanmedian(v))
        pstd = float(np.std(v)) if len(v) >= 2 else 0.0
        return vf, abs(med - pitch), pstd

    times = cand["time"].values
    rows = []
    for i, r in cand.iterrows():
        t = float(r["time"])
        cp = r["chosen_pitch"]
        if pd.isna(cp):
            continue  # pitch 미결정 후보 (드묾) — 학습 대상 아님
        pitch = int(round(cp))

        # 라벨: GT 매칭
        if len(gt_t):
            j = int(np.argmin(np.abs(gt_t - t)))
            matched = abs(gt_t[j] - t) <= TOL
        else:
            matched = False
        validity = int(matched)

        onset_type = "no_onset"
        if matched:
            # 직전 GT 노트로 subtype
            prev_idx = j - 1
            if prev_idx >= 0:
                prev_gap = gt_t[j] - gt_t[prev_idx]
                if gt_p[prev_idx] == gt_p[j] and prev_gap < REATTACK_GAP:
                    onset_type = "reattack"
                elif prev_gap < REATTACK_GAP:
                    # 다른 pitch, 작은 gap: 어택 세기로 pluck/legato
                    att = (win_stat(env, ehop, sr, t - 0.02, t + 0.04, np.max) - env_lo) / env_sc
                    onset_type = "legato" if att < 0.35 else "pluck"
                else:
                    onset_type = "pluck"
            else:
                onset_type = "pluck"

        # 피처 (validity 판정용 — GT 안 쓰는 오디오 신호만)
        att_peak = (win_stat(env, ehop, sr, t - 0.02, t + 0.04, np.max) - env_lo) / env_sc
        pre_min = win_stat(rms, rhop, sr, t - 0.05, t - 0.005, np.min, 1e-6)
        post_max = win_stat(rms, rhop, sr, t + 0.005, t + 0.05, np.max)
        rms_ratio = post_max / pre_min if pre_min > 1e-6 else (10.0 if post_max > 1e-6 else 1.0)
        flux_peak = win_stat(flux, fhop, sr, t - 0.01, t + 0.04, np.max) / flux_hi
        vf, pdist, _ = pyin_win(t + 0.02, t + 0.17, pitch)
        harm = cqt_harmonic(C, chop, sr, base, pitch, t + 0.01, t + 0.15)
        gap_prev = t - times[i - 1] if i > 0 else 9.9
        gap_next = times[i + 1] - t if i + 1 < len(times) else 9.9
        srcs = str(r.get("sources", ""))

        # 옵션2: 서스테인 안정성 피처 — 어려운 톤의 약한 어택 '진짜' 노트를
        # no_onset(프렛 노이즈)과 구분. 진짜 노트는 어택이 약해도 노트 중간부에
        # 안정된 pitch와 지속되는 배음이 있다. 노이즈는 어택만 있고 지속 없음.
        sustain_voiced, sustain_dist, pitch_std = pyin_win(t + 0.08, t + 0.25, pitch)
        harm_attack = cqt_harmonic(C, chop, sr, base, pitch, t + 0.02, t + 0.08)
        harm_mid = cqt_harmonic(C, chop, sr, base, pitch, t + 0.15, t + 0.28)
        harm_continuity = harm_mid / harm_attack if harm_attack > 1e-6 else 0.0

        # 옵션3: offset(듀레이션) regression 타깃 + 감쇠 피처
        # 타깃은 정답 노트 길이(validity=1일 때만 유효). 다음 온셋까지 캡.
        t_cap = min(t + 1.5, times[i + 1] if i + 1 < len(times) else t + 1.5)
        row_pitch = cqt_pitch_row(C, chop, sr, base, pitch)
        cqt_decay = decay_time(row_pitch, chop, sr, t, t_cap)
        rms_row = rms  # 전체 rms, decay_time이 t~t_cap만 봄
        rms_decay = decay_time(rms_row, rhop, sr, t, t_cap, ratio=0.25)
        gt_duration = 0.0
        onset_delta = 0.0
        gt_velocity = 0
        if matched:
            # 매칭된 GT 노트: offset(길이)/onset-correction(시각차)/velocity 타깃
            gnote = min(
                (n for inst in pm.instruments if not inst.is_drum for n in inst.notes),
                key=lambda n: abs(n.start - gt_t[j]),
            )
            gt_duration = float(gnote.end - gnote.start)
            onset_delta = float(gnote.start - t)  # 이 proposal을 얼마나 이동해야 GT에 맞나
            gt_velocity = int(gnote.velocity)

        rows.append({
            "song_id": song_id,
            "time": round(t, 6),
            "pitch": pitch,
            "validity": validity,
            "onset_type": onset_type,
            "att_peak": att_peak,
            "rms_ratio": rms_ratio,
            "flux_peak": flux_peak,
            "pyin_voiced": vf,
            "pyin_dist": pdist,
            "cqt_harmonic": harm,
            "gap_prev": min(gap_prev, 9.9),
            "gap_next": min(gap_next, 9.9),
            "audio_strength": float(r.get("audio_strength", 0.0)),
            "src_audio": int("audio" in srcs),
            "src_raw": int("raw" in srcs),
            "dur": float(r.get("end", t)) - t,
            # 옵션2 추가 피처
            "sustain_voiced": sustain_voiced,
            "sustain_dist": sustain_dist,
            "pitch_std": pitch_std,
            "harm_continuity": harm_continuity,
            # 옵션3 감쇠 피처 + offset 타깃
            "cqt_decay": cqt_decay,
            "rms_decay": rms_decay,
            "gt_duration": round(gt_duration, 6),
            # 옵션2 라운드2: onset-correction / velocity 타깃
            "onset_delta": round(onset_delta, 6),
            "gt_velocity": gt_velocity,
        })
    return rows


def main():
    files = [
        ("bass_test", "eval/runs/snap3_bass_test/bass_candidates.csv",
         "eval/ground_truth/bass_test.wav", "eval/ground_truth/bass_test.mid"),
        ("bass_sample", "eval/runs/snap2_bass_sample/bass_candidates.csv",
         "eval/ground_truth/bass_sample.wav", "eval/ground_truth/bass_sample_octcorrected.mid"),
        ("bass_sample2", "eval/runs/snap3_bass_sample2/bass_candidates.csv",
         "eval/ground_truth/bass_sample2.wav", "eval/ground_truth/bass_sample2_octcorrected.mid"),
        ("bass_sample3", "eval/runs/snap3_bass_sample3/bass_candidates.csv",
         "eval/ground_truth/bass_sample3.wav", "eval/ground_truth/bass_sample3_fullcorrected.mid"),
        ("bass_sample4", "eval/runs/snap3_bass_sample4/bass_candidates.csv",
         "eval/ground_truth/bass_sample4.wav", "eval/ground_truth/bass_sample4_octcorrected.mid"),
        ("bass_sample5", "eval/runs/snap3_bass_sample5/bass_candidates.csv",
         "eval/ground_truth/bass_sample5.wav", "eval/ground_truth/bass_sample5_octcorrected.mid"),
    ]
    all_rows = []
    for song_id, cand, wav, mid in files:
        if not Path(cand).exists():
            print(f"{song_id}: skip (no {cand})", file=sys.stderr)
            continue
        print(f"Extracting proposals for {song_id}...", file=sys.stderr)
        rows = extract(song_id, cand, wav, mid)
        all_rows.extend(rows)
        n_real = sum(r["validity"] for r in rows)
        n_fake = len(rows) - n_real
        from collections import Counter
        types = Counter(r["onset_type"] for r in rows)
        print(f"  {len(rows)} proposals: real={n_real} no_onset={n_fake} | {dict(types)}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    out = Path("eval/proposal_features.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
