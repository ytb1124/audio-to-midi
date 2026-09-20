#!/usr/bin/env python3
"""파이프라인 스테이지 간 고비용 오디오 변환 공유 캐시 (2026-07-11).

배경: 각 스테이지가 독립 모듈이라 286초 파일 기준 librosa.load(리샘플 포함)
10회, pYIN 4회, CQT 10회를 중복 계산 — 전체 변환 200초의 대부분.
output_dir에 npy로 캐시해서 같은 job 안에서 한 번만 계산한다.

계산 코드는 각 스테이지의 기존 함수와 '동일 파라미터'여야 함 — 캐시는
비트 동일 결과를 반환해야 하고, 파라미터가 다르면 다른 캐시 키를 쓴다.
캐시 파일이 없으면 계산 후 저장, 저장 실패는 무시(계산 결과는 반환).
"""
import sys
from pathlib import Path

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def load_audio_22050(audio_path, output_dir=None):
    """librosa.load(sr=22050, mono) + peak normalize, output_dir 캐시."""
    cache = None
    if output_dir is not None:
        cache = Path(output_dir) / "cache_y22050.npy"
        if cache.exists():
            try:
                y = np.load(str(cache))
                return y, 22050
            except Exception:
                pass

    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if len(y) == 0:
        raise RuntimeError("Empty audio")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    if cache is not None:
        try:
            np.save(str(cache), y.astype(np.float32))
        except Exception as exc:
            eprint(f"Warning: audio cache save failed: {exc}")
    return y, sr


def pyin_track(y, sr, fmin_note, fmax_note, output_dir=None, tag=None):
    """librosa.pyin(frame=2048, hop=256, fill_na=nan) → (midi, voiced_prob, hop).

    midi는 nan 포함 float 배열. tag는 캐시 키 (예: 'e1c5', 'c1c5', 'e1g3').
    """
    cache = None
    if output_dir is not None and tag:
        cache = Path(output_dir) / f"cache_pyin_{tag}.npz"
        if cache.exists():
            try:
                z = np.load(str(cache))
                return z["midi"], z["voiced_prob"], 256
            except Exception:
                pass

    import librosa

    hop = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz(fmin_note), fmax=librosa.note_to_hz(fmax_note),
        sr=sr, frame_length=2048, hop_length=hop, fill_na=np.nan,
    )
    midi = np.full(len(f0), np.nan)
    valid = np.isfinite(f0)
    with np.errstate(divide="ignore"):
        midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)

    if cache is not None:
        try:
            np.savez(str(cache), midi=midi, voiced_prob=voiced_prob)
        except Exception as exc:
            eprint(f"Warning: pyin cache save failed: {exc}")
    return midi, voiced_prob, hop


def onset_env_hpss(y, sr, output_dir=None):
    """HPSS(0.65h+0.35p) → onset_strength(median, fmax=2200, hop=256).

    main의 extract_audio_onsets_and_strength와 fragment_merge_v2의
    build_onset_env가 동일 계산을 중복 수행 — 공유.
    """
    cache = None
    if output_dir is not None:
        cache = Path(output_dir) / "cache_onset_env.npy"
        if cache.exists():
            try:
                return np.load(str(cache)), 256
            except Exception:
                pass

    import librosa

    hop = 256
    try:
        y_h, y_p = librosa.effects.hpss(y)
        y_for = 0.65 * y_h + 0.35 * y_p
    except Exception:
        y_for = y
    env = librosa.onset.onset_strength(
        y=y_for, sr=sr, hop_length=hop, aggregate=np.median, fmax=2200
    )

    if cache is not None:
        try:
            np.save(str(cache), env.astype(np.float32))
        except Exception as exc:
            eprint(f"Warning: onset env cache save failed: {exc}")
    return env, hop


def cqt_c1_72(y, sr, output_dir=None):
    """공통 CQT: fmin=C1, 72 bins, 12/oct, hop=256, log1p(10x). base_midi=24."""
    cache = None
    if output_dir is not None:
        cache = Path(output_dir) / "cache_cqt_c1_72.npy"
        if cache.exists():
            try:
                return np.load(str(cache)), 256, 24
            except Exception:
                pass

    import librosa

    hop = 256
    C = np.abs(
        librosa.cqt(
            y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
            n_bins=72, bins_per_octave=12,
        )
    )
    C = np.log1p(10.0 * C)

    if cache is not None:
        try:
            np.save(str(cache), C.astype(np.float32))
        except Exception as exc:
            eprint(f"Warning: cqt cache save failed: {exc}")
    return C, hop, 24
