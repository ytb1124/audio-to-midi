"""R2 후보: PESTO f0 + 온셋 분할 단성 베이스 전사 (보고서 P1-1 "PESTO F0 + 음역 prior").

분리 베이스 스템에서
  1) PESTO(Riou et al., pesto-pitch) f0·신뢰도를 10 ms로 뽑는다(runs/_f0/pesto 캐시 재사용).
  2) 경계 = 오디오 온셋(librosa, delta) ∪ 반올림 피치가 바뀌어 min_stable 이상 유지되는 지점 ∪ 유성/무성 전환.
  3) 구간마다 유성 비율 ≥ 0.5, 길이 ≥ min_dur이면 노트 하나(피치 = 어택 30 ms 이후 중앙값 반올림).
  4) 음역 prior: MIDI 23~67(B0~G4) 밖은 버린다.
CLAUDE.md 실패 기록("pYIN 직접 노트 생성 → 노트 유실")과 같은 계열이라, 온셋으로 같은 음 재타격을 분리하는 점이 다르다.
파라미터는 dev(Slakh train)에서만 고르고(runs/_pesto_notes/best.json), 테스트 세트에는 그 값으로 1회 돌린다.

사용법:
  .venv-f0/bin/python eval/heldout/pesto_notes.py f0 --set dev|slakh|slakh_realbass|slakh_realbass_up
  .venv-f0/bin/python eval/heldout/pesto_notes.py tune
  .venv-f0/bin/python eval/heldout/pesto_notes.py run --set slakh   # runs/stem_pesto/bass.json 작성
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eval"))
import run_heldout as R  # noqa: E402

F0 = HERE / "runs" / "_f0" / "pesto"
OUT = HERE / "runs" / "_pesto_notes"
GRID = {"conf": [0.3, 0.5, 0.7], "delta": [0.05, 0.1, 0.2], "min_dur": [0.05, 0.08]}


def items(set_name: str):
    if set_name == "dev":
        songs = sorted(p.name for p in (HERE / "runs" / "_sep" / "slakh_train").iterdir()
                       if (p / "stems.json").exists())
        return [(s, HERE / "runs" / "_sep" / "slakh_train" / s / "bass.wav", HERE / "data" / "slakh_train" / s / "bass.mid")
                for s in songs]
    man = json.loads((HERE / "data" / "manifest.json").read_text())
    return [(i["song"], HERE / "runs" / "_sep" / set_name / i["song"] / "bass.wav",
             HERE / "data" / set_name / i["song"] / "bass.mid") for i in man[set_name]]


def f0_path(set_name, song):
    return F0 / ("dev_train" if set_name == "dev" else set_name) / f"{song}.npz"


def cmd_f0(args):
    for song, audio, _ in items(args.set):
        out = f0_path(args.set, song)
        if out.exists() or not audio.exists():
            continue
        subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "backend/scripts/bass_f0_tracks.py"),
                        str(audio), str(out), "--method", "pesto"], check=True, capture_output=True)
        print(song, flush=True)


def transcribe(audio: Path, z, conf=0.5, delta=0.1, min_dur=0.05, min_stable=0.05):
    t, f0, cf = z["t"], z["f0"], z["conf"]
    dt = float(np.median(np.diff(t)))
    voiced = (cf >= conf) & (f0 > 30) & (f0 < 500)
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440.0), np.nan)
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=220, units="time", delta=delta, backtrack=False)
    bounds = set(np.searchsorted(t, on).tolist())
    r = np.where(np.isnan(midi), -1, np.round(midi)).astype(int)
    k = int(round(min_stable / dt))
    for i in range(1, len(r)):
        if r[i] != r[i - 1] and (r[i] == -1 or r[i - 1] == -1 or np.all(r[i:i + k] == r[i])):
            bounds.add(i)
    bounds = sorted(b for b in bounds if 0 < b < len(r)) + [len(r)]
    notes, a = [], 0
    for b in bounds:
        seg = midi[a:b]
        if (b - a) * dt >= min_dur and np.mean(~np.isnan(seg)) >= 0.5:
            body = seg[int(0.03 / dt):] if b - a > int(0.03 / dt) + 2 else seg
            p = int(np.round(np.nanmedian(body))) if np.any(~np.isnan(body)) else None
            if p is not None and 23 <= p <= 67:
                notes.append(pretty_midi.Note(80, p, float(t[a]), float(t[b - 1] + dt)))
        a = b
    return notes


def write(notes, path: Path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=33)
    inst.notes = notes
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def cmd_tune(_args):
    rows = []
    data = [(s, a, g, np.load(f0_path("dev", s))) for s, a, g in items("dev") if f0_path("dev", s).exists()]
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        f1 = []
        for s, a, g, z in data:
            tmp = OUT / "tune.mid"
            write(transcribe(a, z, **p), tmp)
            f1.append(R.score_bass(tmp, g)["f1"])
        rows.append((float(np.mean(f1)), p))
        print(f"{np.mean(f1):.4f} {p}", flush=True)
    best = max(rows, key=lambda r: r[0])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "best.json").write_text(json.dumps({"dev_f1": best[0], "params": best[1], "n_dev": len(data)}, indent=1))
    print("best", best)


def cmd_run(args):
    best = json.loads((OUT / "best.json").read_text())["params"]
    run = R.RUNS / "stem_pesto"
    summ_path = run / "bass.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    for song, audio, gt in items(args.set):
        fp = f0_path(args.set, song)
        if not fp.exists():
            continue
        out = run / "bass" / args.set / song / "bass_pesto.mid"
        write(transcribe(audio, np.load(fp), **best), out)
        res = R.score_bass(out, gt)
        res["midi"] = str(out.relative_to(ROOT))
        summ[f"{args.set}/{song}"] = res
        print(song, round(res["f1"], 4), flush=True)
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps(summ, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("f0")
    a.add_argument("--set", required=True)
    sub.add_parser("tune")
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    args = ap.parse_args()
    {"f0": cmd_f0, "tune": cmd_tune, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
