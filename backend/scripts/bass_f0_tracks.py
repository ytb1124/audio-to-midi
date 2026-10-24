"""베이스 오디오의 f0 곡선 추출기 모음 (R7 피치 벤드 비교용, 2026-10-25).

모든 추출기는 같은 형식을 돌려준다: times(초), f0(Hz, 무성은 0), conf(0~1).
- pyin       librosa.pyin (fmin C1, frame 2048 @22.05 kHz), conf = voiced_prob
- crepe      torchcrepe full, 10 ms, viterbi, conf = periodicity
- pesto      pesto-pitch (mir-1k_g7), 10 ms, conf = confidence
- rmvpe      RVC의 RMVPE(third_party/rmvpe, MIT), 10 ms, conf = 최대 salience
- basicpitch Basic Pitch contour 출력(3 bins/semitone)에서 베이스 음역 argmax, conf = salience

basicpitch는 .venv-backend에서, 나머지는 .venv-f0(또는 같은 패키지가 있는 venv)에서 돈다.
CLI: python bass_f0_tracks.py <audio> <out.npz> --method pyin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FMIN, FMAX = 32.70, 500.0  # C1 ~ B4: 베이스 기본음 + 여유


def _load(path: str, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def f0_pyin(path: str):
    import librosa

    sr, hop = 22050, 256
    y = _load(path, sr)
    f0, voiced, prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=2048, hop_length=hop)
    f0 = np.where(voiced & np.isfinite(f0), f0, 0.0)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    return t, f0, np.nan_to_num(prob)


def f0_crepe(path: str):
    import os

    import torch
    import torchcrepe

    device = os.environ.get("CREPE_DEVICE", "cpu")  # MPS는 CPU 대비 중앙값 8 cent 차이(실측)라 기본 CPU
    sr = 16000
    y = torch.from_numpy(_load(path, sr))[None]
    hop = 160
    f0, per = torchcrepe.predict(y, sr, hop, FMIN, FMAX, "full", batch_size=512,
                                 device=device, decoder=torchcrepe.decode.viterbi, return_periodicity=True)
    f0, per = f0[0].cpu().numpy(), per[0].cpu().numpy()
    t = np.arange(len(f0)) * hop / sr
    return t, f0, per


def f0_pesto(path: str):
    import pesto
    import torch

    sr = 16000
    y = torch.from_numpy(_load(path, sr))
    t, f0, conf, _ = pesto.predict(y, sr, step_size=10.0)
    t = t.numpy() / 1000.0 if t.max() > 1000 else t.numpy()
    return t, f0.numpy(), conf.numpy()


def f0_rmvpe(path: str):
    sys.path.insert(0, str(ROOT / "third_party" / "rmvpe"))
    from rmvpe_lib import RMVPE  # noqa: E402

    sr = 16000
    y = _load(path, sr)
    m = RMVPE(str(ROOT / "third_party" / "rmvpe" / "rmvpe.pt"), is_half=False, device="cpu")
    mel = m.extract_mel(y, center=True)
    hidden = m.mel2hidden(mel).squeeze(0).cpu().numpy()
    f0 = m.decode(hidden, thred=0.03)
    conf = hidden.max(axis=1)
    t = np.arange(len(f0)) * 160 / sr
    return t, f0, conf


def f0_basicpitch(path: str):
    from basic_pitch.inference import predict

    out, _, _ = predict(path)
    c = out["contour"]  # (frames, 264), bin k = MIDI 21 + k/3
    lo, hi = int((24 - 21) * 3), int((71 - 21) * 3)  # C1 ~ B4
    sub = c[:, lo:hi]
    k = sub.argmax(axis=1)
    # 포물선 보간으로 1/3 반음 양자화를 완화
    kk = np.clip(k, 1, sub.shape[1] - 2)
    a, b, d = sub[np.arange(len(k)), kk - 1], sub[np.arange(len(k)), kk], sub[np.arange(len(k)), kk + 1]
    den = a - 2 * b + d
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - d) / den, 0.0)
    midi = 21 + (lo + kk + np.clip(off, -0.5, 0.5)) / 3.0
    conf = sub.max(axis=1)
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(len(f0)) * 256 / 22050
    return t, f0, conf


METHODS = {"pyin": f0_pyin, "crepe": f0_crepe, "pesto": f0_pesto, "rmvpe": f0_rmvpe, "basicpitch": f0_basicpitch}


def extract(path: str, method: str):
    t, f0, conf = METHODS[method](path)
    return np.asarray(t, float), np.asarray(f0, float), np.asarray(conf, float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    a = ap.parse_args()
    t, f0, conf = extract(a.audio, a.method)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, t=t, f0=f0, conf=conf)
    print(f"{a.method}: {len(t)} frames, voiced {np.mean(f0 > 0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
