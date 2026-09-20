#!/usr/bin/env python3
"""
전원 험(mains hum, 50Hz/60Hz + 배음)을 좁은 노치 필터로 제거하는 전처리 단계.

배경: bass_sample3.wav 실측 — 첫 50초 구간에서 50.00Hz 성분이 스펙트럼
최댓값의 90%를 차지 (다른 정답 파일들은 5~8% 수준). Basic Pitch가 이걸
49초짜리 가짜 G1 노트로 감지해서 파이프라인 전체를 오염시켰다.

실제 연주된 음(예: 진짜 G1=49Hz)은 자연스러운 피치 변동/비브라토로 주파수가
살짝 흔들리는 반면, 전원 험은 정확히 50.000Hz(또는 60.000Hz)에 고정되어
있다. 그래서 아주 좁은 대역(Q가 높은 노치)만 제거하면 험은 없어지고
실제 저음역 노트는 거의 안 건드린다.
"""
import sys

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def remove_mains_hum(y, sr, fundamentals=(50.0, 60.0), n_harmonics=3, q=30.0):
    """50Hz/60Hz 및 그 배음을 좁은 노치 필터로 제거."""
    from scipy.signal import iirnotch, filtfilt

    out = y.copy()

    for f0 in fundamentals:
        for h in range(1, n_harmonics + 1):
            freq = f0 * h
            if freq >= sr / 2 - 1:
                continue
            b, a = iirnotch(freq, q, sr)
            out = filtfilt(b, a, out)

    return out.astype(np.float32)


def detect_hum_ratio(y, sr, window_sec=4.0, hop_sec=10.0):
    """구간별로 50Hz 근처 성분이 전체 스펙트럼에서 얼마나 지배적인지 진단."""
    ratios = []
    n = len(y)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)

    for start in range(0, max(1, n - win), hop):
        seg = y[start : start + win]
        if len(seg) < sr:
            continue
        fft = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        hum_mask = (freqs > 49.5) & (freqs < 50.5)
        overall_mask = (freqs > 20) & (freqs < 500)
        if not np.any(hum_mask) or not np.any(overall_mask):
            continue
        r = float(fft[hum_mask].max() / fft[overall_mask].max())
        ratios.append((start / sr, r))

    return ratios


def main():
    import argparse
    import soundfile as sf
    import librosa

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_in")
    parser.add_argument("audio_out")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio_in, sr=None, mono=True)

    ratios = detect_hum_ratio(y, sr)
    max_ratio = max((r for _, r in ratios), default=0.0)
    eprint(f"Max hum ratio detected: {max_ratio:.3f}")

    cleaned = remove_mains_hum(y, sr)
    sf.write(args.audio_out, cleaned, sr)
    eprint(f"Saved hum-filtered audio: {args.audio_out}")


if __name__ == "__main__":
    main()
