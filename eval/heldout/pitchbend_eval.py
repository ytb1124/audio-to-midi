"""R7 피치 벤드 평가 — f0 추출기 × 변환 파라미터를 정답 벤드 곡선과 비교한다.

노트 골격은 정답 노트를 쓴다(벤드 추정만 떼어 잰다). 노트마다 10 ms 격자에서
  - 정답 곡선: 정답 MIDI의 pitch bend × 벤드 범위(RPN 0, 없으면 ±2반음)
  - 추정 곡선: backend/scripts/bass_pitch_bend.py가 내는 곡선(벤드 아님 판정이면 0)
지표
  - 벤드 노트 검출 P/R/F1: 곡선 |c| ≥ 50 cent가 30 ms 이상 이어지는 노트를 "벤드 노트"로 본다(양쪽 같은 규칙)
  - 벤드 노트 곡선 오차: 정답 벤드 노트의 프레임 평균 |추정 − 정답| (cent)
  - 무벤드 노트 오차: 정답 무벤드 노트의 같은 값. 기준선(벤드 없음)은 0이다 → 흔들림 지표
세트
  - dev:   Slakh train 중 베이스 벤드가 있는 곡(htdemucs 스템). 파라미터는 여기서만 고른다.
  - slakh: Slakh test 20곡 30~90초(htdemucs 스템). 후보마다 1회 채점.
  - realbass: slakh_realbass의 실제 베이스 렌더(벤드를 렌더링에 그대로 반영 — 정답 곡선이 정확)

사용법:
  .venv-f0/bin/python eval/heldout/pitchbend_eval.py f0 --set slakh --methods pyin pesto rmvpe crepe
  .venv-f0/bin/python eval/heldout/pitchbend_eval.py tune --set dev
  .venv-f0/bin/python eval/heldout/pitchbend_eval.py test --set slakh --config best_configs.json
  .venv-f0/bin/python eval/heldout/pitchbend_eval.py labelcheck --set slakh
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from bass_pitch_bend import BendParams, compute_bends  # noqa: E402
sys.path.insert(0, str(HERE))
from run_heldout import bootstrap_ci  # noqa: E402

RUNS = HERE / "runs"
F0_DIR = RUNS / "_f0"
OUT_DIR = RUNS / "_pitchbend"
BASS_PROGRAMS = range(32, 40)
WIN = (30.0, 90.0)
BEND_THR, BEND_DUR = 50.0, 0.030
GRID = 0.01
METHODS = ["basicpitch", "pyin", "crepe", "pesto", "rmvpe", "swiftf0"]


# ---------------------------------------------------------------- 정답

def _rpn_range(inst) -> list[tuple[float, float]]:
    """(시각, 반음) 벤드 범위 변화. RPN 0(101=0,100=0) 다음 CC6/CC38."""
    sel = {101: None, 100: None}
    out = [(0.0, 2.0)]
    semi = 2.0
    for cc in sorted(inst.control_changes, key=lambda c: c.time):
        if cc.number in (101, 100):
            sel[cc.number] = cc.value
        elif cc.number == 6 and sel[101] == 0 and sel[100] == 0:
            semi = float(cc.value)
            out.append((cc.time, semi))
        elif cc.number == 38 and sel[101] == 0 and sel[100] == 0:
            out.append((cc.time, semi + cc.value / 100.0))
    return out


def gt_items(all_src: Path, window=WIN, assume_range: float | None = None):
    """창 안의 베이스 노트와 정답 cent 곡선(10 ms 격자). 같은 pitch·10 ms 중복은 하나로."""
    pm = pretty_midi.PrettyMIDI(str(all_src))
    items, seen = [], set()
    for inst in pm.instruments:
        if inst.is_drum or inst.program not in BASS_PROGRAMS:
            continue
        rng = _rpn_range(inst)
        pbs = sorted(inst.pitch_bends, key=lambda b: b.time)
        pt = np.array([b.time for b in pbs])
        pv = np.array([b.pitch for b in pbs], float)
        rt = np.array([r[0] for r in rng])
        rv = np.array([r[1] for r in rng])
        for n in inst.notes:
            if not (window[0] <= n.start < window[1]):
                continue
            key = (n.pitch, round(n.start, 2))
            if key in seen:
                continue
            seen.add(key)
            end = min(n.end, window[1])
            g = np.arange(n.start, end, GRID)
            if len(g) == 0:
                continue
            k = np.searchsorted(pt, g, side="right") - 1
            val = np.where(k >= 0, pv[np.clip(k, 0, None)] if len(pv) else 0.0, 0.0)
            kr = np.searchsorted(rt, g, side="right") - 1
            semis = assume_range if assume_range is not None else rv[np.clip(kr, 0, None)]
            cents = val / 8192.0 * semis * 100.0
            items.append({"start": n.start - window[0], "end": end - window[0], "pitch": n.pitch,
                          "grid": g - window[0], "gt": cents})
    items.sort(key=lambda d: d["start"])
    return items


def is_bend(curve: np.ndarray) -> bool:
    over = np.abs(curve) >= BEND_THR
    run = best = 0
    for o in over:
        run = run + 1 if o else 0
        best = max(best, run)
    return best * GRID >= BEND_DUR


# ---------------------------------------------------------------- 세트

def set_songs(set_name: str) -> list[dict]:
    """[{song, audio, gt_src, window, rendered_gt?}]"""
    if set_name == "slakh":
        man = json.loads((HERE / "data" / "manifest.json").read_text())
        return [{"song": i["song"], "audio": RUNS / "_sep" / "slakh" / i["song"] / "bass.wav",
                 "gt_src": HERE / "data" / "_raw" / "slakh" / i["song"] / "all_src.mid", "window": WIN}
                for i in man["slakh"]]
    if set_name == "dev":
        man = json.loads((HERE / "data" / "dev_bends.json").read_text())
        return [{"song": s, "audio": RUNS / "_sep" / "slakh_train" / s / "bass.wav",
                 "gt_src": ROOT / "data" / "raw" / "slakh_train" / s / "all_src.mid", "window": WIN} for s in man]
    if set_name == "bendtest":
        man = json.loads((HERE / "data" / "bendtest_manifest.json").read_text())
        return [{"song": i["song"], "audio": RUNS / "_sep" / "slakh_bendtest" / i["song"] / "bass.wav",
                 "gt_src": HERE / "data" / "_raw" / "slakh" / i["song"] / "all_src.mid", "window": WIN} for i in man]
    if set_name in ("dev_bends_up", "bendtest_up"):
        man = json.loads((HERE / "data" / f"{set_name}_manifest.json").read_text())
        return [{"song": i["song"], "audio": HERE / "data" / set_name / i["song"] / "bass_only.wav",
                 "gt_src": HERE / "data" / set_name / i["song"] / "bass_bends.mid", "window": (0.0, 60.0)} for i in man]
    if set_name == "realbass":
        man = json.loads((HERE / "data" / "manifest.json").read_text())
        return [{"song": i["song"], "audio": HERE / "data" / "slakh_realbass" / i["song"] / "bass_only.wav",
                 "gt_src": HERE / "data" / "slakh_realbass" / i["song"] / "bass_bends.mid", "window": (0.0, 60.0)}
                for i in man.get("slakh_realbass", [])]
    raise SystemExit(f"unknown set {set_name}")


def f0_path(method: str, set_name: str, song: str) -> Path:
    return F0_DIR / method / set_name / f"{song}.npz"


def cmd_f0(args) -> int:
    for method in args.methods:
        py = ROOT / (".venv-backend" if method == "basicpitch" else ".venv-f0") / "bin" / "python"
        for it in set_songs(args.set):
            out = f0_path(method, args.set, it["song"])
            if out.exists() or not Path(it["audio"]).exists():
                continue
            subprocess.run([str(py), "-W", "ignore", str(ROOT / "backend/scripts/bass_f0_tracks.py"),
                            str(it["audio"]), str(out), "--method", method], check=True,
                           capture_output=True, text=True)
            print(method, args.set, it["song"], flush=True)
    return 0


# ---------------------------------------------------------------- 채점

def evaluate(set_name: str, method: str | None, p: BendParams, songs=None, assume_range=None) -> dict:
    """method=None이면 기준선(벤드 없음)."""
    per_song = {}
    for it in songs or set_songs(set_name):
        items = gt_items(it["gt_src"], it["window"], assume_range)
        if not items:
            continue
        if method is not None:
            fp = f0_path(method, set_name, it["song"])
            if not fp.exists():
                continue
            z = np.load(fp)
            notes = [(d["start"], d["end"], d["pitch"]) for d in items]
            bends = compute_bends(notes, z["t"], z["f0"], z["conf"], p)
        else:
            bends = {}
        tp = fp_ = fn = 0
        err_b, err_nb = [], []
        for i, d in enumerate(items):
            est = np.zeros(len(d["grid"]))
            if i in bends:
                tt, cv = bends[i]
                est = np.interp(d["grid"], tt, cv, left=0.0, right=cv[-1])
                est[d["grid"] < tt[0]] = 0.0
            gb, eb = is_bend(d["gt"]), is_bend(est)
            tp += gb and eb
            fp_ += (not gb) and eb
            fn += gb and not eb
            e = float(np.mean(np.abs(est - d["gt"])))
            (err_b if gb else err_nb).append(e)
        per_song[it["song"]] = {"tp": tp, "fp": fp_, "fn": fn, "err_bend": err_b, "err_nonbend": err_nb}
    return per_song


def summarize(per_song: dict) -> dict:
    tp = sum(v["tp"] for v in per_song.values())
    fp = sum(v["fp"] for v in per_song.values())
    fn = sum(v["fn"] for v in per_song.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    eb = [e for v in per_song.values() for e in v["err_bend"]]
    enb = [e for v in per_song.values() for e in v["err_nonbend"]]
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "n_bend": len(eb), "n_nonbend": len(enb),
            "err_bend": float(np.mean(eb)) if eb else float("nan"),
            "err_nonbend": float(np.mean(enb)) if enb else float("nan")}


GRID_PARAMS = {
    "smooth_ms": [30.0, 60.0, 100.0],
    "deadband": [20.0, 35.0, 50.0],
    "min_bend": [50.0, 80.0, 120.0],
    "slide_only": [True, False],
    # 신뢰도 척도가 추출기마다 다르다(RMVPE는 베이스에서 0.5 초과 프레임이 14%뿐). 모두 같은 격자로 고른다.
    "min_conf": [0.05, 0.2, 0.5],
}
NONBEND_TOL = 2.0  # cent: 무벤드 노트 평균 오차 허용 상한(기준선 0) — 사전 등록


def cmd_tune(args) -> int:
    """개발셋에서 추출기별 최적 파라미터: 무벤드 오차 ≤ NONBEND_TOL 조건에서 F1 최대, 동률이면 벤드 오차 최소."""
    best = {}
    for method in args.methods:
        rows = []
        for vals in itertools.product(*GRID_PARAMS.values()):
            p = BendParams(**dict(zip(GRID_PARAMS, vals)))
            s = summarize(evaluate(args.set, method, p))
            rows.append((s, p))
        ok = [r for r in rows if r[0]["err_nonbend"] <= NONBEND_TOL]
        pick = max(ok, key=lambda r: (round(r[0]["f1"], 4), -r[0]["err_bend"])) if ok else None
        if pick:
            best[method] = {"params": asdict(pick[1]), "dev": pick[0]}
            print(f"{method:10s} {json.dumps(pick[0])} {asdict(pick[1])}")
        else:
            print(f"{method:10s} 조건 만족 설정 없음")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # round 2: 다른 추출기의 기존 선택을 지우지 않도록 합친다
    path = OUT_DIR / "best_configs.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(best)
    path.write_text(json.dumps(merged, indent=2))
    return 0


def cmd_test(args) -> int:
    best = json.loads((OUT_DIR / "best_configs.json").read_text())
    out = OUT_DIR / f"test_{args.set}.json"
    prev = json.loads(out.read_text()) if out.exists() else {"summary": {}, "per_song": {}}
    res, per = prev["summary"], prev["per_song"]
    res["baseline"] = summarize(evaluate(args.set, None, BendParams()))
    per["baseline"] = evaluate(args.set, None, BendParams())
    for method, cfg in best.items():
        if args.methods and method not in args.methods:
            continue
        if method in res:  # 후보마다 1회: 이미 채점한 후보는 다시 돌리지 않는다
            continue
        p = BendParams(**cfg["params"])
        per[method] = evaluate(args.set, method, p)
        res[method] = summarize(per[method])
    out.write_text(json.dumps({"summary": res, "per_song": per}, indent=2, default=float))
    for k, v in res.items():
        print(f"{k:10s} F1={v['f1']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"bendErr={v['err_bend']:.1f}c nonbendErr={v['err_nonbend']:.2f}c (bend n={v['n_bend']}, nonbend n={v['n_nonbend']})")
    return 0


def cmd_labelcheck(args) -> int:
    """정답 벤드 곡선이 오디오와 맞는지: 정답 벤드 노트에서 f0(cent)와 정답 곡선의 중앙 오차를
    RPN 범위 그대로 / ±2 가정 / 벤드 무시(0) 세 가지로 비교한다."""
    for it in set_songs(args.set):
        fp = f0_path(args.method, args.set, it["song"])
        if not fp.exists():
            continue
        z = np.load(fp)
        out = []
        for label, rng in (("rpn", None), ("range2", 2.0), ("zero", 0.0)):
            items = gt_items(it["gt_src"], it["window"], rng)
            errs = []
            ref = gt_items(it["gt_src"], it["window"], None)
            for d, dr in zip(items, ref):
                if not is_bend(dr["gt"]):
                    continue
                m = (z["t"] >= d["start"] + 0.03) & (z["t"] < d["end"]) & (z["f0"] > 0) & (z["conf"] > 0.5)
                if m.sum() < 3:
                    continue
                c = 1200 * np.log2(z["f0"][m] / (440 * 2 ** ((d["pitch"] - 69) / 12)))
                c = c - 1200 * np.round(c / 1200)
                g = np.interp(z["t"][m], d["grid"], d["gt"])
                errs.append(float(np.median(np.abs(c - g))))
            out.append(f"{label}={np.median(errs):.0f}c(n={len(errs)})" if errs else f"{label}=-")
        print(it["song"], " ".join(out))
    return 0


REAL = {  # 평가 2: 실녹음(정답 MIDI 없음). 노트 골격은 현행 베이스 파이프라인 출력.
    "Bass_2": (ROOT.parent / "sample audio" / "bass sample" / "Bass_2.wav", ROOT / "eval" / "runs" / "mtlB_bass2" / "bass_hybrid.mid"),
    "Mosquito": (ROOT / "uploads" / "0e987dac-fdb4-44a2-81b6-44b140c99219.wav", ROOT / "eval" / "runs" / "mtlB_mosquito" / "bass_hybrid.mid"),
}
REF_METHODS = ["pyin", "crepe", "rmvpe", "pesto"]


def cmd_real_f0(args) -> int:
    for method in args.methods:
        py = ROOT / (".venv-backend" if method == "basicpitch" else ".venv-f0") / "bin" / "python"
        for name, (audio, _) in REAL.items():
            out = f0_path(method, "real", name)
            if not out.exists():
                subprocess.run([str(py), "-W", "ignore", str(ROOT / "backend/scripts/bass_f0_tracks.py"),
                                str(audio), str(out), "--method", method], check=True, capture_output=True)
                print(method, name, flush=True)
    return 0


def _ref_curve(name: str, exclude: str, grid: np.ndarray) -> np.ndarray:
    """평가 대상 추출기를 뺀 나머지의 f0 중앙값(Hz, 무성은 nan) — 순환 평가를 피한다."""
    cols = []
    for m in REF_METHODS:
        if m == exclude:
            continue
        z = np.load(f0_path(m, "real", name))
        f = np.where((z["f0"] > 0) & (z["conf"] >= {"rmvpe": 0.2, "pesto": 0.5}.get(m, 0.5)), z["f0"], np.nan)
        cols.append(np.interp(grid, z["t"], f, left=np.nan, right=np.nan))
    c = np.vstack(cols)
    # 다수(과반)가 유성일 때만 쓰고, 옥타브가 엇갈리면 중앙값이 망가지므로 log 영역 중앙값
    ok = np.sum(~np.isnan(c), axis=0) >= max(2, len(cols) // 2 + 1)
    med = np.exp(np.nanmedian(np.log(c), axis=0))
    return np.where(ok, med, np.nan)


def cmd_real(args) -> int:
    """평가 2: 렌더 피치(노트 피치 + 벤드)와 오디오 f0의 cent 오차.
    방법 m마다 기준 f0 = m을 뺀 나머지 추출기의 중앙값(순환 방지). 같은 기준·같은 노트에서
    기준선(벤드 없음)과 m을 짝지어 비교한다. 안정 노트(기준 f0가 노트 피치 ±30c 안에 80% 이상) = 무벤드 노트."""
    best = json.loads((OUT_DIR / "best_configs.json").read_text())
    rows = {}
    for method, cfg in best.items():
        p = BendParams(**cfg["params"])
        pairs = {"stable": [], "moving": []}
        for name, (audio, mid) in REAL.items():
            pm = pretty_midi.PrettyMIDI(str(mid))
            notes = sorted(((n.start, n.end, n.pitch) for i in pm.instruments if not i.is_drum for n in i.notes))
            z = np.load(f0_path(method, "real", name))
            bends = compute_bends(notes, z["t"], z["f0"], z["conf"], p)
            for i, (s0, e, pitch) in enumerate(notes):
                g = np.arange(s0 + 0.03, e, GRID)
                if len(g) < 3:
                    continue
                ref = _ref_curve(name, method, g)
                m = ~np.isnan(ref)
                if m.mean() < 0.5:
                    continue
                rc = 1200 * np.log2(ref[m] / (440 * 2 ** ((pitch - 69) / 12)))
                rc = rc - 1200 * np.round(rc / 1200)
                est = np.zeros(len(g))
                if i in bends:
                    tt, cv = bends[i]
                    est = np.interp(g, tt, cv)
                kind = "stable" if np.mean(np.abs(rc) <= 30) >= 0.8 else "moving"
                pairs[kind].append((float(np.mean(np.abs(rc))), float(np.mean(np.abs(est[m] - rc)))))
        rows[method] = {}
        for kind, v in pairs.items():
            a = np.array(v)
            d = a[:, 1] - a[:, 0]
            lo, hi = bootstrap_ci(d)
            rows[method][kind] = {"n": len(a), "baseline": float(a[:, 0].mean()), "method": float(a[:, 1].mean()),
                                  "diff": float(d.mean()), "ci": [lo, hi]}
            print(f"{method:8s} {kind:6s} n={len(a):4d} 기준선 {a[:, 0].mean():6.2f}c → {a[:, 1].mean():6.2f}c "
                  f"(차이 {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}])")
    (OUT_DIR / "real_eval.json").write_text(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("f0")
    a.add_argument("--set", required=True)
    a.add_argument("--methods", nargs="+", default=METHODS)
    t = sub.add_parser("tune")
    t.add_argument("--set", default="dev")
    t.add_argument("--methods", nargs="+", default=METHODS)
    s = sub.add_parser("test")
    s.add_argument("--set", required=True)
    s.add_argument("--methods", nargs="*")
    lc = sub.add_parser("labelcheck")
    lc.add_argument("--set", required=True)
    lc.add_argument("--method", default="pyin")
    rf = sub.add_parser("real_f0")
    rf.add_argument("--methods", nargs="+", default=METHODS)
    sub.add_parser("real")
    args = ap.parse_args()
    return {"f0": cmd_f0, "tune": cmd_tune, "test": cmd_test, "labelcheck": cmd_labelcheck,
            "real_f0": cmd_real_f0, "real": cmd_real}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
