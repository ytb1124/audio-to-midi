"""Round 2 N4 — Beat This! 활성값 + CASM 후처리(arXiv 2610.09318). .venv-beat/bin/python으로 실행.

CASM 패키지는 .venv-beat(제품 비트 격자가 쓰는 venv)에 설치하지 않고, 클론한 소스를 sys.path로만 읽는다
(eval/heldout/runs/_r2/src/casm, gitignore). CASM은 학습 가중치가 없고 SMC로 고정한 기본 설정(7F)만 쓴다 —
우리 쪽에서 튜닝할 파라미터가 없다.

같은 로짓에서 Beat This! 기본 후처리(minimal)도 다시 내서 기존 runs/_beats/beat_this와 일치하는지 확인한다.
출력: runs/_beats/beat_this_casm/<song>.json, runs/_beats/_logits/<song>.npz
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runs" / "_r2" / "src" / "casm" / "casm" / "src"))


def main():
    import torch
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import load_audio
    from casm_beat_tracking import CASMDecoder

    torch.set_num_threads(2)
    songs = [m["song"] for m in json.load(open(HERE / "data" / "manifest.json"))["slakh"]]
    out = HERE / "runs" / "_beats" / "beat_this_casm"
    logit_dir = HERE / "runs" / "_beats" / "_logits"
    out.mkdir(parents=True, exist_ok=True)
    logit_dir.mkdir(parents=True, exist_ok=True)
    a2f = Audio2Frames(checkpoint_path="final0", device="cpu")
    minimal = Postprocessor(type="minimal")
    dec = CASMDecoder()
    for s in songs:
        t = time.time()
        signal, sr = load_audio(HERE / "data" / "slakh" / s / "mix.wav")
        bl, dl = a2f(signal, sr)
        t_act = time.time() - t
        bl_np, dl_np = bl.cpu().numpy().astype(float), dl.cpu().numpy().astype(float)
        np.savez(logit_dir / f"{s}.npz", beat=bl_np, downbeat=dl_np)
        t = time.time()
        beats, downs = dec.decode(bl_np, dl_np, input_type="logits")
        t_casm = time.time() - t
        mb, md = minimal(bl, dl)
        ref = json.load(open(HERE / "runs" / "_beats" / "beat_this" / f"{s}.json"))
        same = len(mb) == len(ref["beats"]) and np.allclose(mb, ref["beats"], atol=1e-3)
        json.dump({"beats": list(map(float, beats)), "downbeats": list(map(float, downs)),
                   "seconds": t_act + t_casm, "casm_seconds": t_casm, "minimal_matches_r5": bool(same)},
                  open(out / f"{s}.json", "w"))
        print(f"{s}: casm {len(beats)} beats {len(downs)} downbeats ({t_casm:.2f}s), minimal==R5: {same}", flush=True)


if __name__ == "__main__":
    main()
