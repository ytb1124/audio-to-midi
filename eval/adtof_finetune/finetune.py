"""R4: ADTOF(Frame_RNN, adtofAll fold 0)를 htdemucs 분리 드럼 스템에 파인튜닝한다 (.venv-adtof에서 실행).

계획서(eval/ADTOF_FINETUNE_PLAN.md)와 다른 점 — 데이터 접근 때문에 줄인 범위:
  - 되새김용 ADTOF 데이터셋은 신청제(Zenodo 10084510)라 쓸 수 없다 → A3 불가.
  - 검증셋 ENST 드러머 3 대신 Slakh train 분리 스템 일부(13곡)를 검증에 쓴다(렌더링 도메인).
  - 조건: A0(원본), A1(전체 파인튜닝), A4f(합성곱 고정, GRU·출력층만). 학습 = Slakh train 40곡 스템.
판정 규칙(계획서 6절 그대로): 검증셋에서 A0를 넘는 조건이 없으면 MDB를 채점하지 않고 끝낸다.
넘는 조건이 있으면 그중 하나만 MDB 23곡에서 1회 채점하고, 곡별 paired CI 하한 > 0일 때만 채택한다.
전처리·라벨은 ADTOF 패키지의 DataLoader/Track(madmom 스펙트로그램, MIDI_REDUCED_5 매핑)을 그대로 쓴다.

사용법: .venv-adtof/bin/python eval/adtof_finetune/finetune.py --condition A1
출력: eval/adtof_finetune/weights/<condition>.weights.h5 (gitignore), 로그 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STEMS = ROOT / "eval" / "heldout" / "runs" / "_sep" / "slakh_train"
LABELS = ROOT / "eval" / "heldout" / "data" / "slakh_train"
SEED = 0


def split():
    songs = sorted(p.name for p in STEMS.iterdir() if (p / "stems.json").exists() and (LABELS / p.name / "drums.mid").exists())
    return songs[:40], songs[40:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("A1", "A4f"), required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--val-steps", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=12)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf
    from adtof.model.dataLoader import DataLoader
    from adtof.model.model import Model
    from adtof.ressources import instrumentsMapping

    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    model, hp = Model.modelFactory(modelName="Frame_RNN", scenario="adtofAll", fold=0)
    assert model.weightLoadedFlag, "pretrained weights failed to load"
    train, val = split()

    def loader(songs):
        return DataLoader([str(STEMS / s / "drums.wav") for s in songs], [str(LABELS / s / "drums.mid") for s in songs],
                          crossValidation=False, checkFilesNameMatch=False,
                          mappingDictionaries=[instrumentsMapping.MIDI_REDUCED_5], **hp)

    dl_t, dl_v = loader(train), loader(val)
    ds_t = DataLoader._getDataset(dl_t.getGen(training=True, **hp), **hp)
    ds_v = DataLoader._getDataset(dl_v.getGen(training=True, **hp), **hp)

    km = model.model
    if a.condition == "A4f":
        # Frame_RNN의 합성곱·배치정규화 블록은 중첩된 Sequential 한 덩어리다(층 이름 "sequential").
        for layer in km.layers:
            if type(layer).__name__ == "Sequential":
                layer.trainable = False
        print("trainable params:", sum(int(np.prod(w.shape)) for w in km.trainable_weights))
    km.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=a.lr), loss=tf.keras.backend.binary_crossentropy,
               weighted_metrics=[])  # ADTOF 원래 설정과 같은 손실, 학습률만 1/5

    out = HERE / "weights"
    out.mkdir(parents=True, exist_ok=True)
    wpath = out / f"{a.condition}.weights.h5"
    t0 = time.time()
    hist = km.fit(
        ds_t, epochs=a.max_epochs, steps_per_epoch=a.steps, validation_data=ds_v, validation_steps=a.val_steps,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, min_delta=1e-4),
                   tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)],
        verbose=2,
    )
    km.save_weights(str(wpath))
    log = {"condition": a.condition, "train_songs": train, "val_songs": val, "lr": a.lr, "seed": SEED,
           "epochs_run": len(hist.history["loss"]), "loss": hist.history["loss"], "val_loss": hist.history["val_loss"],
           "seconds": round(time.time() - t0, 1), "weights": str(wpath)}
    (HERE / f"log_{a.condition}.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k not in ("train_songs", "val_songs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
