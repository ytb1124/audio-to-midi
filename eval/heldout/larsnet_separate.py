"""Round 2 N2 — LarsNet(polimi-ispl/larsnet) 드럼 피스 분리 워커. .venv-adtstr/bin/python으로 실행한다
(LarsNet 의존성은 torch·torchaudio·soundfile·pyyaml·tqdm뿐이라 ADT_STR용 격리 venv를 같이 쓴다).

코드·가중치(CC BY-NC 4.0, 저자 Google Drive)는 eval/heldout/runs/_r2/src/larsnet(gitignore)에 있다.
입력: --jobs <json> [[drums_stem.wav, out_dir], ...] → out_dir/{kick,snare,toms,hihat,cymbals}.wav (16-bit)
Wiener 필터 없음(저장소 기본값).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "runs" / "_r2" / "src" / "larsnet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    jobs = [(str(Path(i).resolve()), str(Path(o).resolve())) for i, o in json.loads(Path(a.jobs).read_text())]
    import soundfile as sf
    import torch

    torch.set_num_threads(2)
    os.chdir(REPO)  # config.yaml의 가중치 경로가 저장소 기준 상대 경로다
    sys.path.insert(0, str(REPO))
    from larsnet import LarsNet

    net = LarsNet(device=a.device, config="config.yaml")
    for src, out in jobs:
        t = time.time()
        stems = net(src)
        Path(out).mkdir(parents=True, exist_ok=True)
        for stem, wav in stems.items():
            sf.write(str(Path(out) / f"{stem}.wav"), wav.cpu().numpy().T, net.sr, subtype="PCM_16")
        print(json.dumps({"out": out, "seconds": round(time.time() - t, 1)}), flush=True)


if __name__ == "__main__":
    main()
