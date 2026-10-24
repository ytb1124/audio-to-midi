"""에디터 가상악기용 멀티샘플을 public/samples/로 변환한다 (2026-10-25).

원본(data/raw/samples_src/, gitignore)은 재배포 가능한 라이선스만 쓴다 — public/samples/LICENSES.md.
- 피아노: Salamander Grand Piano V3 (CC BY 3.0) — A·C·D#·F# 전 옥타브(3반음 간격, A0~C8), 4 벨로시티 층
- 일렉 베이스: Karoryfer Fashionbass (CC0) — 3반음 간격 F#0~A4, 3층
- 업라이트 베이스: D. Smolken double bass pizz (CC0) — 3반음 간격 C1~A3, 2층
- 어쿠스틱 드럼: Virtuosity Drums mid mix (CC0) — GM 35~59 주요 피스, 피스마다 3~4층
- 일렉 피아노: Greg Sullivan CP80 (CC BY 3.0), 2층
- 스트링 앙상블: VSCO 2 CE sustain vibrato (CC0), 2층. 패드는 같은 샘플에 느린 어택·필터를 쓴다(추가 용량 0)
층 사이 크기 차이는 보존한다(악기 단위로 하나의 게인만 적용). mp3 128 kbps.
manifest.json 형식: {inst: {"layers": [{"maxVel": 0~127, "notes": {"<midi>": "file.mp3"}}], ...}}
드럼은 "notes"의 키가 GM 번호이고 피치 이동 없이 재생한다.

사용법: .venv-backend/bin/python packaging/build_editor_samples.py
"""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "samples_src"
OUT = ROOT / "public" / "samples"
PC = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7,
      "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}


def midi_of(name: str) -> int:
    """'D#1', 'gb0', 'fs2' → MIDI (C-1 = 0)."""
    m = re.match(r"([a-gA-G])(#|b|s)?(-?\d)", name)
    acc = {"#": "#", "s": "#", "b": "b"}.get(m.group(2) or "", "")
    return 12 * (int(m.group(3)) + 1) + PC[m.group(1).lower() + acc]


def peak(path: Path, max_dur: float) -> float:
    info = sf.info(str(path))
    y, _ = sf.read(str(path), frames=int(min(info.duration, max_dur) * info.samplerate), always_2d=True)
    return float(np.max(np.abs(y))) if y.size else 0.0


def encode(src: Path, dst: Path, max_dur: float, fade: float, gain_db: float, mono: bool) -> None:
    dur = min(sf.info(str(src)).duration, max_dur)
    fade = min(fade, dur * 0.5)
    dst.parent.mkdir(parents=True, exist_ok=True)
    af = f"atrim=0:{dur:.3f},afade=t=out:st={dur - fade:.3f}:d={fade:.3f},volume={gain_db:.2f}dB"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-af", af, "-ar", "44100"]
    cmd += ["-ac", "1", "-b:a", "96k"] if mono else ["-ac", "2", "-b:a", "128k"]
    cmd += ["-codec:a", "libmp3lame", str(dst)]
    subprocess.run(cmd, check=True)


def build(name: str, layers: list[tuple[int, dict[int, Path]]], max_dur: float, fade: float, mono: bool,
          extra: dict, pool: ThreadPoolExecutor) -> dict:
    """layers: [(maxVel, {midi: src})]. 악기 전체 최대 피크를 −1 dBFS로 맞춘다."""
    pk = max(peak(p, max_dur) for _, notes in layers for p in notes.values())
    gain_db = -1.0 - 20 * np.log10(max(pk, 1e-6))
    man_layers, jobs = [], []
    for li, (max_vel, notes) in enumerate(layers):
        files = {}
        for midi, src in sorted(notes.items()):
            fn = f"{midi}_v{li}.mp3"
            files[str(midi)] = fn
            jobs.append(pool.submit(encode, src, OUT / name / fn, max_dur, fade, gain_db, mono))
        man_layers.append({"maxVel": max_vel, "notes": files})
    for j in jobs:
        j.result()
    return {"dir": f"samples/{name}/", "layers": man_layers, **extra}


def piano():
    d = next(SRC.glob("salamander_v3/extracted/*/44.1khz16bit"))
    layers = []
    for max_vel, v in ((40, 4), (70, 8), (100, 12), (127, 16)):
        notes = {}
        for f in d.glob(f"*v{v}.wav"):
            m = re.match(r"([A-G]#?)(\d)v\d+$", f.stem)
            if m:
                notes[midi_of(m.group(1) + m.group(2))] = f
        layers.append((max_vel, notes))
    return layers


def fashionbass():
    d = SRC / "karoryfer.fashionbass" / "notes"
    out = []
    for max_vel, lay in ((60, "p"), (95, "mf"), (127, "ff")):
        out.append((max_vel, {midi_of(f.stem.split("_")[0]): f for f in d.glob(f"*_{lay}_rr1.wav")}))
    return out


def upright():
    d = SRC / "dsmolken.double-bass" / "pizz"
    out = []
    for max_vel, lay in ((75, "p"), (127, "f")):
        out.append((max_vel, {midi_of(f.stem.split("_")[1]): f for f in d.glob(f"pizz_*_{lay}a.wav")}))
    return out


def drums():
    d = SRC / "virtuosity_drums" / "Samples" / "mid"
    pieces = {  # GM 번호 → (폴더, 접두사, 사용할 층 번호들)
        "kick": ("kick", "mid_kick_snoff", [1, 2, 3, 4]),
        "snare": ("snare", "mid_snare_center", [9, 18, 27, 36]),
        "stick": ("snare", "mid_snare_crossstick", [4, 8, 12, 16]),
        "hh_closed": ("hh", "mid_hh_closed", [1, 2, 3, 4]),
        "hh_pedal": ("hh", "mid_hh_pedal", [1, 2, 3]),
        "hh_open": ("hh", "mid_hh_open", [1, 2, 3]),
        "ltom": ("ltom", "mid_ltom_center", [4, 8, 12, 16]),
        "htom": ("htom", "mid_htom_center", [4, 8, 12, 16]),
        "crash": ("crash", "mid_crash_crash", [1, 2, 3]),
        "ride": ("ride", "mid_ride_ride", [1, 2, 3]),
        "bell": ("ride", "mid_ride_bell", [1, 2, 3]),
    }
    gm = {35: "kick", 36: "kick", 37: "stick", 38: "snare", 40: "snare", 42: "hh_closed", 44: "hh_pedal",
          46: "hh_open", 41: "ltom", 43: "ltom", 45: "ltom", 47: "htom", 48: "htom", 50: "htom",
          49: "crash", 52: "crash", 55: "crash", 57: "crash", 51: "ride", 59: "ride", 53: "bell"}

    def file_for(folder, prefix, vl):
        c = sorted((d / folder).glob(f"{prefix}_vl{vl}_rr1.*")) or sorted((d / folder).glob(f"{prefix}_vl{vl}.*"))
        return c[0] if c else None

    layers = []
    for li, max_vel in enumerate((45, 75, 100, 127)):
        notes = {}
        for g, piece in gm.items():
            folder, prefix, vls = pieces[piece]
            vl = vls[min(li, len(vls) - 1)] if li < len(vls) else vls[-1]
            f = file_for(folder, prefix, vl)
            if f is not None:
                notes[g] = f
        layers.append((max_vel, notes))
    return layers


def cp80():
    d = SRC / "gregsullivan_epianos" / "CP80" / "Samples"
    out = []
    for max_vel, lays in ((70, ("MP", "PP")), (127, ("F", "FF"))):
        notes = {}
        for f in sorted(d.glob("*.flac")):
            num, _, lay = f.stem.split("-")
            if lay in lays and int(num) not in notes:
                notes[int(num)] = f
        out.append((max_vel, notes))
    return out


def strings():
    base = SRC / "vsco2ce_strings" / "Strings"
    srcs = [("Cello Section/susvib", 0, 47), ("Viola Section/susvib", 48, 59), ("Violin Section/susVib", 60, 127)]
    out = []
    for max_vel, soft in ((80, True), (127, False)):
        notes = {}
        for folder, lo, hi in srcs:
            for f in sorted((base / folder).glob("*.wav")):
                m = re.search(r"_([A-G]#?\d)_v(\d)", f.name)
                if not m:
                    continue
                v = int(m.group(2))
                if (v == 1) != soft:
                    continue
                midi = midi_of(m.group(1))
                if lo <= midi <= hi and midi not in notes:
                    notes[midi] = f
        out.append((max_vel, notes))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    man = {}
    with ThreadPoolExecutor(8) as pool:
        man["piano"] = build("piano", piano(), 7.0, 1.2, False, {"release": 1.0}, pool)
        man["electric_bass"] = build("electric_bass", fashionbass(), 4.0, 0.5, True, {"release": 0.15}, pool)
        man["upright_bass"] = build("upright_bass", upright(), 3.5, 0.5, True, {"release": 0.2}, pool)
        man["drums"] = build("drums", drums(), 4.0, 0.8, False, {"drum": True, "release": 0.3}, pool)
        man["electric_piano"] = build("electric_piano", cp80(), 6.0, 1.0, False, {"release": 0.6}, pool)
        man["strings"] = build("strings", strings(), 8.0, 1.5, False, {"release": 0.8, "attack": 0.08}, pool)
    man["pad"] = {**man["strings"], "attack": 0.6, "release": 1.8, "filterHz": 2200}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    for k, v in man.items():
        print(k, [len(l["notes"]) for l in v["layers"]])
    print(f"total {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
