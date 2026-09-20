"""HT-Demucs stem separation.

업로드된 1곡을 4-stem(vocals / drums / bass / other)으로 분리한다.
research 문서 1번(Music Source Separation)의 첫 단계 — 이후 bass.wav를 Bass Mode에,
drums.wav를 (추후) Drum Mode에 태우는 파이프라인의 Stage 0.

사용법:
    python demucs_separate.py <input_audio> <output_dir> [--model htdemucs] [--device cpu|mps]

출력:
    <output_dir>/vocals.wav, drums.wav, bass.wav, other.wav
    stdout 마지막 줄에 JSON으로 stem 경로를 찍는다(백엔드가 파싱).

설계 원칙(CLAUDE.md 계승):
- 기존 piano/bass/general 모드는 건드리지 않는다. 이 스크립트는 완전히 독립.
- htdemucs 표준 4-stem 우선 도입. 기타 전용 stem(htdemucs_6s)은 품질 트레이드오프가 있어
  옵션으로만 열어둔다(--model htdemucs_6s).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# htdemucs 4-stem 기준. 6-stem 모델이면 guitar/piano가 추가된다.
STANDARD_STEMS = ["vocals", "drums", "bass", "other"]


def pick_device(requested: str) -> str:
    """요청한 디바이스가 쓸 수 있으면 그대로, 아니면 cpu로 폴백."""
    if requested == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


def separate(input_audio: Path, output_dir: Path, model: str, device: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # demucs는 <demucs_out>/<model>/<track_stem>/<stem>.wav 구조로 쓴다.
    demucs_out = output_dir / "_demucs_raw"
    demucs_out.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model,
        "-d",
        device,
        "-o",
        str(demucs_out),
        "--filename",
        "{stem}.{ext}",
        str(input_audio),
    ]

    # 진행 로그는 stderr로 흐르게 두고(백엔드가 상태만 갱신), 실패 시 예외.
    subprocess.run(cmd, check=True)

    track_dir = demucs_out / model
    # --filename '{stem}.{ext}' 를 주면 track 하위 폴더 없이 model 폴더 바로 아래에 stem이 생긴다.
    stems = {}
    for stem_wav in track_dir.glob("*.wav"):
        stem_name = stem_wav.stem
        dest = output_dir / f"{stem_name}.wav"
        shutil.move(str(stem_wav), str(dest))
        stems[stem_name] = dest.name

    # 정리
    shutil.rmtree(demucs_out, ignore_errors=True)

    if not stems:
        raise RuntimeError("Demucs가 stem을 생성하지 못했습니다.")

    return stems


def main():
    parser = argparse.ArgumentParser(description="HT-Demucs stem separation")
    parser.add_argument("input_audio")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="htdemucs(4-stem, 기본) 또는 htdemucs_6s(guitar/piano 포함 6-stem)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu(기본, 안정) 또는 mps(Apple Silicon 가속, 폴백 있음)",
    )
    args = parser.parse_args()

    input_audio = Path(args.input_audio).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_audio.exists():
        raise SystemExit(f"입력 오디오가 없습니다: {input_audio}")

    device = pick_device(args.device)
    stems = separate(input_audio, output_dir, args.model, device)

    # 마지막 줄에 JSON — 백엔드가 이 줄만 파싱.
    print(json.dumps({"stems": stems, "model": args.model, "device": device}))


if __name__ == "__main__":
    main()
