"""홀드아웃 테스트셋(실제 믹스) 내려받기 + 정답 MIDI 생성.

기존 eval/ground_truth(베이스 6곡·드럼 8곡)는 현행 파이프라인의 튜닝·학습에 쓰인 in-sample 데이터라,
모델 비교용으로는 이 홀드아웃셋을 쓴다. 두 데이터셋 모두 비교하는 세 시스템(현행 파이프라인,
YourMT3+, ADTOF)의 학습 데이터에 들어 있지 않다(RESULTS_HELDOUT.md "오염 점검" 참고).

- Slakh2100 test split (렌더링 믹스, 베이스·드럼 정답): HF 미러
  DreamyWanderer/Slakh2100-FLAC-Redux-Reduced (원본 Zenodo 4599666은 단일 104GB tar라 미러 사용).
  트랙 ID 순으로 조건을 만족하는 앞 N곡을 고정 선택하고 30~90초 구간만 쓴다.
- MDB Drums (MedleyDB 실제 녹음 믹스, 드럼 정답): GitHub CarlSouthall/MDBDrums, 23곡 전곡.

사용법:
    .venv-backend/bin/python eval/heldout/fetch_heldout.py [--slakh-n 20]

출력: eval/heldout/data/{slakh,mdb}/<song>/{mix.wav, bass.mid?, drums.mid}, eval/heldout/data/manifest.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "_raw"

SLAKH_HF = "https://huggingface.co/datasets/DreamyWanderer/Slakh2100-FLAC-Redux-Reduced"
MDB_RAW = "https://raw.githubusercontent.com/CarlSouthall/MDBDrums/master/MDB%20Drums"
MDB_API = "https://api.github.com/repos/CarlSouthall/MDBDrums/git/trees/master?recursive=1"

WIN_START, WIN_END = 30.0, 90.0
BASS_PROGRAMS = range(32, 40)
MIN_BASS_NOTES = 40
MIN_DRUM_HITS = 80

# MDB Drums subclass 라벨 → GM 드럼 pitch (eval/score_drums.py CLASS_MAP과 맞물리도록)
MDB_SUBCLASS_GM = {
    "KD": 36,
    "SD": 38, "SDB": 38, "SDD": 38, "SDF": 38, "SDG": 38, "SDNS": 38, "SST": 37,
    "CHH": 42, "PHH": 44, "OHH": 46,
    "HIT": 50, "MHT": 47, "HFT": 43, "LFT": 41,
    "RDC": 51, "RDB": 53,
    "CRC": 49, "CHC": 52, "SPC": 55,
    "TMB": 54,  # 탬버린: 채점 클래스 밖(제외)
}


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "trackform-eval"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def cut_wav(src: Path, dest: Path, start: float | None, end: float | None) -> None:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start}", "-to", f"{end}"]
    cmd += ["-i", str(src), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True)


def window_notes(notes, start: float, end: float):
    out = []
    for n in notes:
        if start <= n.start < end:
            out.append(pretty_midi.Note(n.velocity, n.pitch, n.start - start, min(n.end, end) - start))
    return out


def write_midi(path: Path, notes, is_drum: bool, program: int = 0) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
    inst.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    pm.instruments.append(inst)
    pm.write(str(path))


def slakh_test_tracks() -> list[str]:
    with urllib.request.urlopen(SLAKH_HF.replace("/datasets/", "/api/datasets/"), timeout=60) as r:
        sib = [s["rfilename"] for s in json.load(r)["siblings"]]
    return sorted({p.split("/")[2] for p in sib if p.startswith("data/test/") and p.endswith("mix.flac")})


def build_slakh(n: int) -> list[dict]:
    picked = []
    for track in slakh_test_tracks():
        if len(picked) >= n:
            break
        mid = fetch(f"{SLAKH_HF}/resolve/main/data/test/{track}/all_src.mid", RAW / "slakh" / track / "all_src.mid")
        pm = pretty_midi.PrettyMIDI(str(mid))
        if pm.get_end_time() < WIN_END + 5:
            continue
        bass = [x for i in pm.instruments if not i.is_drum and i.program in BASS_PROGRAMS for x in i.notes]
        drums = [x for i in pm.instruments if i.is_drum for x in i.notes]
        bass_w = window_notes(bass, WIN_START, WIN_END)
        drums_w = window_notes(drums, WIN_START, WIN_END)
        if len(bass_w) < MIN_BASS_NOTES or len(drums_w) < MIN_DRUM_HITS:
            continue
        flac = fetch(f"{SLAKH_HF}/resolve/main/data/test/{track}/mix.flac", RAW / "slakh" / track / "mix.flac")
        out = DATA / "slakh" / track
        out.mkdir(parents=True, exist_ok=True)
        cut_wav(flac, out / "mix.wav", WIN_START, WIN_END)
        write_midi(out / "bass.mid", bass_w, is_drum=False, program=33)
        write_midi(out / "drums.mid", drums_w, is_drum=True)
        picked.append({
            "song": track, "set": "slakh", "window": [WIN_START, WIN_END],
            "bass_notes": len(bass_w), "drum_hits": len(drums_w),
            "bass_programs": sorted({int(i.program) for i in pm.instruments if not i.is_drum and i.program in BASS_PROGRAMS}),
        })
        print(f"slakh {track}: bass {len(bass_w)} notes, drums {len(drums_w)} hits", flush=True)
    return picked


def build_mdb() -> list[dict]:
    with urllib.request.urlopen(MDB_API, timeout=60) as r:
        tree = [x["path"] for x in json.load(r)["tree"]]
    mixes = sorted(p for p in tree if p.startswith("MDB Drums/audio/full_mix/") and p.endswith(".wav"))
    picked = []
    for p in mixes:
        name = Path(p).name.replace("_MIX.wav", "")
        wav = fetch(f"{MDB_RAW}/audio/full_mix/{urllib.parse.quote(Path(p).name)}", RAW / "mdb" / Path(p).name)
        ann = fetch(f"{MDB_RAW}/annotations/subclass/{name}_subclass.txt", RAW / "mdb" / f"{name}_subclass.txt")
        notes, unknown = [], {}
        for line in ann.read_text().split("\n"):
            parts = line.split()
            if len(parts) < 2:
                continue
            t, lab = float(parts[0]), parts[1].strip()
            gm = MDB_SUBCLASS_GM.get(lab)
            if gm is None:
                unknown[lab] = unknown.get(lab, 0) + 1
                continue
            notes.append(pretty_midi.Note(100, gm, t, t + 0.05))
        out = DATA / "mdb" / name
        out.mkdir(parents=True, exist_ok=True)
        cut_wav(wav, out / "mix.wav", None, None)
        write_midi(out / "drums.mid", notes, is_drum=True)
        picked.append({"song": name, "set": "mdb", "drum_hits": len(notes), "unknown_labels": unknown})
        print(f"mdb {name}: {len(notes)} hits {unknown or ''}", flush=True)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-n", type=int, default=20)
    args = ap.parse_args()
    manifest = {"slakh": build_slakh(args.slakh_n), "mdb": build_mdb()}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"slakh {len(manifest['slakh'])}곡, mdb {len(manifest['mdb'])}곡")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
