"""R6 벨로시티 평가 — 홀드아웃 정답 벨로시티 대비 출력 벨로시티의 곡별 Spearman ρ.

run_heldout.py가 남긴 결과 MIDI(runs/<run>/<kind>.json의 "midi")를 정답과 온셋으로 짝지은 뒤
(베이스: 같은 pitch class·50 ms, 드럼: 같은 5클래스·50 ms) 곡별 Spearman ρ를 낸다.

라벨 점검: Slakh 정답은 곡에 따라 벨로시티가 상수거나 거의 상수다(20곡 중 다수가 고유값 1~5개).
이런 곡은 순위 상관이 정의되지 않으므로 "정답 벨로시티 고유값 ≥ 5, 표준편차 ≥ 5, 짝 ≥ 20"인 곡만 쓴다.
Slakh 렌더러가 벨로시티를 소리 크기에 실제로 반영하는지는 --oracle(정답 스템 어택 dB vs 정답
벨로시티)로 따로 확인한다.

사용법:
    .venv-backend/bin/python eval/heldout/velocity_eval.py --kind bass --runs stem_basicpitch stem_yourmt3 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))

import run_heldout as R  # noqa: E402
import score_drums  # noqa: E402

TOL = 0.05
MIN_UNIQUE, MIN_STD, MIN_PAIRS = 5, 5.0, 20


def bass_notes(path: Path, drums: bool | None = None):
    pm = pretty_midi.PrettyMIDI(str(path))
    out = []
    for inst in pm.instruments:
        if drums is not None and inst.is_drum != drums:
            continue
        out += [(n.start, n.pitch % 12, n.velocity) for n in inst.notes if n.start >= R.IGNORE_HEAD]
    return sorted(out)


def drum_notes(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    out = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            c9 = score_drums.CLASS_MAP.get(n.pitch)
            c5 = R.COLLAPSE5.get(c9) if c9 else None
            if c5 is not None and n.start >= R.IGNORE_HEAD:
                out.append((n.start, c5, n.velocity))
    return sorted(out)


def match(ref, est):
    """같은 키(pitch class 또는 드럼 클래스) 안에서 온셋이 가장 가까운 미사용 출력과 짝짓는다."""
    used = np.zeros(len(est), bool)
    et = np.array([e[0] for e in est]) if est else np.zeros(0)
    pairs = []
    for t, key, v in ref:
        lo, hi = np.searchsorted(et, t - TOL), np.searchsorted(et, t + TOL)
        cand = [i for i in range(lo, hi) if not used[i] and est[i][1] == key]
        if cand:
            i = min(cand, key=lambda k: abs(et[k] - t))
            used[i] = True
            pairs.append((v, est[i][2]))
    return pairs


def eligible(ref) -> bool:
    v = np.array([r[2] for r in ref])
    return len(np.unique(v)) >= MIN_UNIQUE and v.std() >= MIN_STD


def eval_run(kind: str, run: str, sets: list[str]) -> dict:
    s = R.load_summary(kind, run)
    out = {}
    for key, res in s.items():
        set_name, song = key.split("/", 1)
        if set_name not in sets or "midi" not in res:
            continue
        gt = R.DATA / set_name / song / ("bass.mid" if kind == "bass" else "drums.mid")
        ref = bass_notes(gt) if kind == "bass" else drum_notes(gt)
        if not eligible(ref):
            continue
        est = bass_notes(ROOT / res["midi"], drums=False) if kind == "bass" else drum_notes(ROOT / res["midi"])
        pairs = match(ref, est)
        if len(pairs) < MIN_PAIRS:
            continue
        a, b = np.array(pairs).T
        rho = spearmanr(a, b).statistic if np.std(b) > 0 else 0.0
        out[key] = {"rho": float(rho), "pairs": len(pairs), "est_unique": int(len(np.unique(b)))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("bass", "drums"), required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["slakh"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    allres = {}
    for run in args.runs:
        r = eval_run(args.kind, run, args.sets)
        allres[run] = r
        rhos = [v["rho"] for v in r.values()]
        lo, hi = R.bootstrap_ci(rhos)
        print(f"{run:28s} n={len(rhos):2d} mean rho={np.mean(rhos) if rhos else float('nan'):.3f} [{lo:.3f}, {hi:.3f}]"
              f"  est_unique_med={np.median([v['est_unique'] for v in r.values()]) if r else 0}")
    if args.out:
        Path(args.out).write_text(json.dumps(allres, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
