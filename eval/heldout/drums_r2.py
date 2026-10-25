"""Round 2 드럼 후보(N2 LarsNet 피스 분리, N3 ADT_STR) — run_heldout.py와 같은 세트·채점 v2.

시스템(런 디렉터리 = runs/<시스템>/, 요약 = runs/<시스템>/drums.json, run_heldout.py report로 그대로 집계됨)
  adtstr_stem_t04|t06|t08   htdemucs drums 스템 → ADT_STR(setting-tau-0.x)
  adtstr_mix_t08            믹스 그대로 → ADT_STR
  larsnet_*                 htdemucs drums 스템 → LarsNet 5스템 → 클래스별 검출 (larsnet_detect.py)
  muscriptor_mix            믹스 → MuScriptor medium(악기 조건 없음, 기본값) — N1 주 시스템(사전 등록)
  muscriptor_stem           htdemucs drums 스템 → MuScriptor (참고)
  muscriptor_mix_drumcond   믹스 → MuScriptor, instruments=["drums"] 조건 (참고)

세트
  slakh_train  Slakh train 중 R4 검증 분할(songs[40:], 17곡) = 개발셋. 변형·입력 선택은 여기서만.
  mdb          MDB Drums 23곡(실제 녹음, 대표)
  slakh        Slakh test 20곡(보조)

채점
  f1   공통 5클래스 micro F1 (run_heldout.score_drum, 정답 중복 제거 v2)
  f1_8 8클래스 micro F1: kick, snare, tom, hihat_closed(42·44), hihat_open(46), crash(49·52·55·57), ride(51·53·59),
       other(그 밖의 퍼커션 전부 — 탬버린 등). 정답 중복 제거는 5클래스와 같은 규칙(같은 클래스 20 ms).
       5클래스만 내는 시스템(현행 ADTOF)은 42 → hihat_closed, 49 → crash로 읽힌다(그대로가 현행 출력이다).

사용법:
  .venv-backend/bin/python eval/heldout/drums_r2.py run --systems adtstr_stem_t08 --sets slakh_train
  .venv-backend/bin/python eval/heldout/drums_r2.py rescore8 --runs stem_adtof   # 기존 MIDI에 8클래스 점수만 추가
  .venv-backend/bin/python eval/heldout/drums_r2.py report --runs stem_adtof adtstr_stem_t08 --sets mdb
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
import run_heldout as rh  # noqa: E402
import score_drums  # noqa: E402

DATA, RUNS = rh.DATA, rh.RUNS
CLASSES8 = ["kick", "snare", "tom", "hihat_closed", "hihat_open", "crash", "ride", "other"]
NINE_TO_8 = {"kick": "kick", "snare": "snare", "tom": "tom", "hihat_closed": "hihat_closed",
             "hihat_open": "hihat_open", "crash": "crash", "ride": "ride"}
ADTSTR_VARIANT = {"t04": "setting-tau-0.4", "t06": "setting-tau-0.6", "t08": "setting-tau-0.8"}


# ---------------------------------------------------------------- 세트

def songs_of(set_name: str) -> list[str]:
    if set_name == "slakh_train":
        stems = RUNS / "_sep" / "slakh_train"
        s = sorted(p.name for p in stems.iterdir()
                   if (p / "stems.json").exists() and (DATA / "slakh_train" / p.name / "drums.mid").exists())
        return s[40:]  # eval/adtof_finetune/finetune.py split()의 검증 분할
    return rh._songs(rh._manifest(), set_name)


def drum_stem(set_name: str, song: str) -> Path:
    return Path(json.loads((RUNS / "_sep" / set_name / song / "stems.json").read_text())["drums"])


# ---------------------------------------------------------------- 8클래스 채점

def _by8(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    by = {c: [] for c in CLASSES8}
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            if n.start < rh.IGNORE_HEAD:
                continue
            c9 = score_drums.CLASS_MAP.get(n.pitch)
            by[NINE_TO_8.get(c9, "other")].append(n.start)
    return {c: np.array(sorted(v)) for c, v in by.items()}


def score_drum8(est_path: Path, gt_path: Path, tol: float = 0.05) -> dict:
    est, ref = _by8(est_path), _by8(gt_path)
    for c in CLASSES8:
        v = ref[c]
        if len(v) > 1:
            keep = [v[0]]
            for t in v[1:]:
                if t - keep[-1] >= rh.GT_DEDUPE_DRUM:
                    keep.append(t)
            ref[c] = np.array(keep)
    tm = te = tr = 0
    per = {}
    for c in CLASSES8:
        m, eo, ro = score_drums.match_onsets(est[c], ref[c], tol)
        tm, te, tr = tm + m, te + eo, tr + ro
        if len(ref[c]) or len(est[c]):
            per[c] = {"f1": score_drums.prf(m, eo, ro)[2], "n_ref": int(len(ref[c])), "n_est": int(len(est[c]))}
    return {"f1_8": score_drums.prf(tm, te, tr)[2], "per_class8": per}


def score_both(midi: Path, gt: Path) -> dict:
    res = rh.score_drum(midi, gt)
    res.update(score_drum8(midi, gt))
    return res


# ---------------------------------------------------------------- 실행

def _gt(set_name, song):
    return DATA / set_name / song / "drums.mid"


def run_adtstr(system: str, set_name: str, todo: list[str]) -> dict[str, Path]:
    _, inp, tau = system.split("_")
    out = {s: RUNS / system / "drums" / set_name / s / "drums.mid" for s in todo}
    jobs = [[str(drum_stem(set_name, s) if inp == "stem" else DATA / set_name / s / "mix.wav"), str(out[s])]
            for s in todo if not out[s].exists()]
    if jobs:
        jf = RUNS / system / f"jobs_{set_name}.json"
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(jobs))
        env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore")
        subprocess.run(["nice", "-n", "15", str(ROOT / ".venv-adtstr/bin/python"), str(HERE / "adtstr_transcribe.py"),
                        "--variant", ADTSTR_VARIANT[tau], "--jobs", str(jf)], check=True, env=env)
    return out


def run_muscriptor(system: str, set_name: str, todo: list[str]) -> dict[str, Path]:
    variant = system.removeprefix("muscriptor_")
    out = {s: RUNS / system / "drums" / set_name / s / "drums.mid" for s in todo}
    jobs = [[str(drum_stem(set_name, s) if variant == "stem" else DATA / set_name / s / "mix.wav"), str(out[s]),
             ["drums"] if variant == "mix_drumcond" else None] for s in todo if not out[s].exists()]
    if jobs:
        jf = RUNS / system / f"jobs_{set_name}.json"
        jf.parent.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps(jobs))
        env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore")
        subprocess.run(["nice", "-n", "15", str(ROOT / ".venv-muscriptor/bin/python"), str(HERE / "muscriptor_transcribe.py"),
                        "--model", os.environ.get("MUSCRIPTOR_MODEL", "medium"), "--jobs", str(jf)], check=True, env=env)
    return out


def run_larsnet(system: str, set_name: str, todo: list[str]) -> dict[str, Path]:
    import larsnet_detect  # 같은 디렉터리, 메인 venv

    return {s: larsnet_detect.transcribe(system, set_name, s, drum_stem(set_name, s)) for s in todo}


def cmd_run(args) -> int:
    for system in args.systems:
        summary_path = RUNS / system / "drums.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        for set_name in args.sets:
            todo = [s for s in songs_of(set_name) if f"{set_name}/{s}" not in summary or args.force]
            if args.songs:
                todo = [s for s in todo if s in args.songs]
            if not todo:
                continue
            t0 = time.time()
            runner = (run_adtstr if system.startswith("adtstr_") else
                      run_muscriptor if system.startswith("muscriptor_") else run_larsnet)
            mids = runner(system, set_name, todo)
            per = (time.time() - t0) / len(todo)
            for s, midi in mids.items():
                key = f"{set_name}/{s}"
                try:
                    res = score_both(midi, _gt(set_name, s))
                    res["midi"] = str(midi.relative_to(ROOT))
                except Exception as e:  # noqa: BLE001
                    res = {"error": f"{type(e).__name__}: {e}"[:500]}
                res["seconds"] = round(per, 1)
                summary[key] = res
                print(f"{system:22s} {key:40s} F1={res.get('f1', float('nan')):.4f} F1_8={res.get('f1_8', float('nan')):.4f}", flush=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_rescore8(args) -> int:
    for run in args.runs:
        p = RUNS / run / "drums.json"
        s = json.loads(p.read_text())
        for key, res in s.items():
            if "midi" not in res:
                continue
            set_name, song = key.split("/", 1)
            res.update(score_drum8(ROOT / res["midi"], _gt(set_name, song)))
        p.write_text(json.dumps(s, indent=2, ensure_ascii=False))
        print(f"{run}: 8클래스 점수 {len(s)}곡")
    return 0


def cmd_report(args) -> int:
    for set_name in args.sets:
        songs = [f"{set_name}/{s}" for s in songs_of(set_name)]
        for metric in ("f1", "f1_8"):
            print(f"\n[{set_name} — {metric}, 기준 {args.baseline}]\n")
            print(f"| 시스템 | 평균 {metric} [95% CI] | {args.baseline} 대비 [95% CI] | n |\n|---|---|---|---|")
            base = rh.load_summary("drums", args.baseline)
            for run in args.runs:
                s = rh.load_summary("drums", run)
                ok = [k for k in songs if k in s and metric in s[k]]
                if not ok:
                    continue
                v = [s[k][metric] for k in ok]
                lo, hi = rh.bootstrap_ci(v)
                d = [s[k][metric] - base[k][metric] for k in ok if k in base and metric in base[k]]
                dlo, dhi = rh.bootstrap_ci(d)
                diff = "—" if run == args.baseline else f"{np.mean(d):+.4f} [{dlo:+.3f}, {dhi:+.3f}] (n={len(d)})"
                print(f"| {run} | {np.mean(v):.4f} [{lo:.3f}, {hi:.3f}] | {diff} | {len(ok)} |")
        cls = CLASSES8
        print(f"\n[{set_name} — 8클래스 클래스별 곡평균 F1(정답에 있는 곡만)]\n")
        print("| 시스템 | " + " | ".join(cls) + " |\n|---|" + "---|" * len(cls))
        for run in args.runs:
            s = rh.load_summary("drums", run)
            ok = [k for k in songs if k in s and "per_class8" in s[k]]
            if not ok:
                continue
            cells = []
            for c in cls:
                v = [s[k]["per_class8"][c]["f1"] for k in ok if c in s[k]["per_class8"] and s[k]["per_class8"][c]["n_ref"] > 0]
                cells.append(f"{np.mean(v):.3f}" if v else "—")
            print(f"| {run} | " + " | ".join(cells) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--systems", nargs="+", required=True)
    r.add_argument("--sets", nargs="+", required=True)
    r.add_argument("--songs", nargs="*")
    r.add_argument("--force", action="store_true")
    r8 = sub.add_parser("rescore8")
    r8.add_argument("--runs", nargs="+", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--runs", nargs="+", required=True)
    rp.add_argument("--sets", nargs="+", default=["mdb", "slakh"])
    rp.add_argument("--baseline", default="stem_adtof")
    args = ap.parse_args()
    return {"run": cmd_run, "rescore8": cmd_rescore8, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
