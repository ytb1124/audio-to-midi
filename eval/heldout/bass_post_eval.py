"""베이스 후처리 후보 2종 (2026-10-25 overnight, 사용자 요청 "3번·1번").

3번 onset_refine: 현행 하이브리드 출력의 노트 시작을 분리 스템의 실제 어택으로 당긴다.
  실베이스(R1)에서 피치가 어택 뒤 100~160 ms에 걸쳐 올라와 노트 시작이 늦게 잡히는 것이 점수 손실의 주원인이었다.
  노트 시작 s에 대해 [s − back, s + fwd] 안의 온셋 강도 피크(곡 최대 대비 thr 이상)를 찾고, 피크 직전 국소 최소(발치)로 옮긴다.
  직전 노트 시작 + 30 ms보다 앞으로는 가지 않는다. 노트 끝은 그대로(길이만 늘어남).
1번 ensemble: 현행 하이브리드(스템)와 YourMT3+(믹스 입력)를 합친다. 믹스가 있을 때만 가능한 경로다.
  and   : 하이브리드 노트 중 YourMT3+가 같은 pitch class로 ±tol 안에 낸 것만
  or    : 하이브리드 ∪ (하이브리드가 놓친 YourMT3+ 노트)
  y_snap: YourMT3+ 노트를 쓰되 시작을 ±tol 안의 가장 가까운 하이브리드 노트 시작으로 맞춘다(없으면 그대로)
설정은 dev_realbass(Slakh train 30곡, 테스트와 다른 일렉 음색)에서만 고르고, 테스트는 후보마다 1회.

사용법:
  .venv-backend/bin/python eval/heldout/bass_post_eval.py tune
  .venv-backend/bin/python eval/heldout/bass_post_eval.py test
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_heldout as R  # noqa: E402

OUT = R.RUNS / "_bass_post"
TEST_SETS = ["slakh", "slakh_realbass", "slakh_realbass_up"]
REFINE_GRID = {"back": [0.06, 0.10, 0.15], "fwd": [0.0, 0.02], "thr": [0.1, 0.2, 0.3]}
ENS_GRID = {"mode": ["and", "or", "y_snap"], "tol": [0.05, 0.08, 0.12]}
SR, HOP = 22050, 110  # 5 ms


def notes_of(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted(((n.start, n.end, n.pitch, n.velocity) for i in pm.instruments if not i.is_drum for n in i.notes))


def write(notes, path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=33)
    inst.notes = [pretty_midi.Note(int(v), int(p), float(s), float(max(e, s + 0.03))) for s, e, p, v in notes]
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


_env_cache: dict = {}


def onset_env(audio: Path):
    if audio not in _env_cache:
        y, _ = librosa.load(str(audio), sr=SR, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        _env_cache[audio] = (env / (env.max() or 1.0), np.arange(len(env)) * HOP / SR)
    return _env_cache[audio]


def onset_refine(notes, audio: Path, back=0.1, fwd=0.0, thr=0.2):
    env, t = onset_env(audio)
    out, prev = [], -1.0
    for s, e, p, v in notes:
        lo, hi = max(s - back, prev + 0.03), s + fwd
        m = np.flatnonzero((t >= lo) & (t <= hi))
        new = s
        if len(m) > 2:
            seg = env[m]
            pk = [i for i in range(1, len(seg) - 1) if seg[i] >= thr and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
            if pk:
                i = min(pk, key=lambda k: abs(t[m[k]] - s))
                j = i
                while j > 0 and seg[j - 1] < seg[j]:
                    j -= 1
                new = float(t[m[j]])
        out.append((new, e, p, v))
        prev = new
    return out


def ensemble(h, y, mode="and", tol=0.08):
    def match(a, pool):
        return [b for b in pool if abs(b[0] - a[0]) <= tol and b[2] % 12 == a[2] % 12]

    if mode == "and":
        return [a for a in h if match(a, y)]
    if mode == "or":
        return sorted(h + [b for b in y if not match(b, h)])
    out = []
    for b in y:
        near = [a for a in h if abs(a[0] - b[0]) <= tol]
        s = min(near, key=lambda a: abs(a[0] - b[0]))[0] if near else b[0]
        out.append((s, max(b[1], s + 0.03), b[2], b[3]))
    return sorted(out)


def paths(set_name: str, song: str):
    hyb = R.RUNS / "stem_basicpitch" / "bass" / set_name / song / "bass_hybrid.mid"
    ymt = R.RUNS / "mix_yourmt3" / "bass" / set_name / song / "bass.mid"
    stem = R.RUNS / "_sep" / set_name / song / "bass.wav"
    gt = R.DATA / set_name / song / "bass.mid"
    return hyb, ymt, stem, gt


def evaluate(set_name: str, kind: str, params: dict, tag: str | None = None) -> dict:
    res = {}
    for song in R._songs(R._manifest(), set_name):
        hyb, ymt, stem, gt = paths(set_name, song)
        if not (hyb.exists() and gt.exists()) or (kind == "ensemble" and not ymt.exists()):
            continue
        h = notes_of(hyb)
        new = onset_refine(h, stem, **params) if kind == "refine" else ensemble(h, notes_of(ymt), **params)
        out = OUT / (tag or "tmp") / set_name / f"{song}.mid"
        res[f"{set_name}/{song}"] = R.score_bass(write(new, out), gt)
        res[f"{set_name}/{song}"]["midi"] = str(out.relative_to(R.ROOT))
    return res


def cmd_tune(_args) -> int:
    best = {}
    for kind, grid in (("refine", REFINE_GRID), ("ensemble", ENS_GRID)):
        rows = []
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid, vals))
            r = evaluate("dev_realbass", kind, p)
            rows.append((float(np.mean([v["f1"] for v in r.values()])), p, len(r)))
            print(kind, f"{rows[-1][0]:.4f}", p, f"n={len(r)}", flush=True)
        base = R.load_summary("bass", "stem_basicpitch")
        b = [v["f1"] for k, v in base.items() if k.startswith("dev_realbass/")]
        top = max(rows, key=lambda r: r[0])
        best[kind] = {"params": top[1], "dev_f1": top[0], "dev_baseline_f1": float(np.mean(b)), "n": top[2]}
        print("BEST", kind, best[kind], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "best.json").write_text(json.dumps(best, indent=1))
    return 0


def cmd_test(_args) -> int:
    best = json.loads((OUT / "best.json").read_text())
    for kind, cfg in best.items():
        run = f"bass_{kind}"
        summ = {}
        for s in TEST_SETS:
            summ.update(evaluate(s, kind, cfg["params"], tag=run))
        (R.RUNS / run).mkdir(parents=True, exist_ok=True)
        (R.RUNS / run / "bass.json").write_text(json.dumps(summ, indent=1))
        for s in TEST_SETS:
            m, lo, hi, n = R.paired("bass", s, run, "stem_basicpitch")
            print(f"{kind:8s} {s:18s} n={n:2d} Δ={m:+.4f} [{lo:+.3f}, {hi:+.3f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tune")
    sub.add_parser("test")
    a = ap.parse_args()
    return {"tune": cmd_tune, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
