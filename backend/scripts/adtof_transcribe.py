"""ADTOF 드럼 전사 래퍼 — 격리 venv(.venv-adtof)에서 실행된다.

모델/코드 출처: 공식 GitHub `MZehren/ADTOF` (Frame_RNN, adtofAll fold 0 — README 노트북의 기본값).
TensorFlow + madmom 스택이라 메인 venv와 분리한다.

ADTOF 출력은 5 class(BD 35 / SD 38 / HH 42 / TT 47 / CY+RD 49)다. 오픈/클로즈 하이햇,
크래시/라이드를 구분하지 않으므로 우리 GM 체계로 다음과 같이 옮긴다:
kick36 / snare38 / hihat_closed42 / tom45 / crash49.

사용법:
    python adtof_transcribe.py input.wav output.mid
마지막 줄에 {"counts": {...}, "engine": "adtof"} JSON을 출력한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

ADTOF_TO_GM = {35: 36, 38: 38, 42: 42, 47: 45, 49: 49}
GM_NAMES = {36: "kick", 38: "snare", 42: "hihat", 45: "tom", 49: "cymbal"}
NOTE_SEC = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--weights", type=Path, help="파인튜닝 가중치(eval/adtof_finetune). 생략하면 공개 adtofAll fold 0")
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # ADTOF는 Keras 2 API(optimizers.legacy)를 쓰므로 tf_keras로 고정한다.
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import pretty_midi
    from adtof.model.model import Model

    model, hparams = Model.modelFactory(modelName="Frame_RNN", scenario="adtofAll", fold=0)
    if not model.weightLoadedFlag:
        raise RuntimeError("ADTOF weights failed to load")
    if args.weights:
        model.model.load_weights(str(args.weights))

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        src_dir.mkdir()
        track = src_dir / f"track{args.input.suffix.lower() or '.wav'}"
        shutil.copyfile(args.input, track)
        model.predictFolder(str(track), str(out_dir), writeMidi=False, **hparams)
        # ADTOF는 확장자를 포함한 파일명(track.wav.txt)으로 결과를 쓴다.
        outputs = sorted(out_dir.glob("*.txt")) if out_dir.exists() else []
        if not outputs:
            raise RuntimeError("ADTOF produced no output")
        txt = outputs[0]
        events = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                events.append((float(parts[0]), int(float(parts[1]))))

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums (ADTOF)")
    counts: Counter = Counter()
    for t, pitch in sorted(events):
        gm = ADTOF_TO_GM.get(pitch)
        if gm is None:
            continue
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=gm, start=t, end=t + NOTE_SEC))
        counts[GM_NAMES[gm]] += 1
    pm.instruments.append(inst)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(args.output))
    print(json.dumps({"counts": dict(counts), "engine": "adtof"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
