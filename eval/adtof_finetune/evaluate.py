"""R4 채점: 조건(A0 = 공개 가중치, A1, A4f)별로 분리 드럼 스템을 ADTOF로 전사해 5클래스 F1을 낸다.

  --set val : Slakh train 검증 13곡(finetune.py의 split) — 조건 선택에만 쓴다
  --set mdb : MDB Drums 23곡(실제 녹음, 최종 테스트) — 검증에서 A0를 넘은 조건 하나만, 1회
채점은 eval/heldout/run_heldout.py의 score_drum(채점 v2)과 같다. paired CI는 A0 대비.

사용법: .venv-backend/bin/python eval/adtof_finetune/evaluate.py --set val --conditions A0 A1 A4f
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval" / "heldout"))
import run_heldout as R  # noqa: E402

PY = ROOT / ".venv-adtof" / "bin" / "python"
SCRIPT = ROOT / "backend" / "scripts" / "adtof_transcribe.py"


def items(set_name: str):
    if set_name == "val":
        songs = json.loads((HERE / "log_A1.json").read_text())["val_songs"]
        return [(s, R.RUNS / "_sep" / "slakh_train" / s / "drums.wav", R.DATA / "slakh_train" / s / "drums.mid") for s in songs]
    return [(s, R.RUNS / "_sep" / "mdb" / s / "drums.wav", R.DATA / "mdb" / s / "drums.mid") for s in R._songs(R._manifest(), "mdb")]


def run(cond: str, set_name: str) -> dict:
    out = {}
    for song, audio, gt in items(set_name):
        mid = HERE / "outputs" / cond / set_name / f"{song}.mid"
        if not mid.exists():
            mid.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(PY), str(SCRIPT), str(audio), str(mid)]
            if cond != "A0":
                cmd += ["--weights", str(HERE / "weights" / f"{cond}.weights.h5")]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        out[song] = R.score_drum(mid, gt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("val", "mdb"), required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    a = ap.parse_args()
    res = {c: run(c, a.set) for c in a.conditions}
    base = res.get("A0")
    summary = {}
    for c, r in res.items():
        f = [v["f1"] for v in r.values()]
        lo, hi = R.bootstrap_ci(f)
        row = {"mean": float(np.mean(f)), "ci": [lo, hi], "n": len(f)}
        if base is not None and c != "A0":
            d = [r[s]["f1"] - base[s]["f1"] for s in r if s in base]
            dlo, dhi = R.bootstrap_ci(d)
            row["vs_A0"] = [float(np.mean(d)), dlo, dhi]
        per_class = {}
        for cls in R.CLASSES5:
            v = [x["per_class"][cls]["f1"] for x in r.values() if cls in x["per_class"] and x["per_class"][cls]["n_ref"] > 0]
            per_class[cls] = float(np.mean(v)) if v else None
        row["per_class"] = per_class
        summary[c] = row
        print(c, json.dumps(row))
    (HERE / f"results_{a.set}.json").write_text(json.dumps({"summary": summary, "per_song": res}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
