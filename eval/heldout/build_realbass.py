"""R1: slakh_realbass 세트 — Slakh test 20곡의 베이스를 실제 녹음 베이스 단음 샘플로 다시 렌더링한다.

왜: 기존 홀드아웃 베이스 결과(YourMT3+ 0.85 vs 현행 0.57)는 Slakh 신스 렌더 조건뿐이고, YourMT3+는
같은 렌더러의 Slakh train으로 학습했다. 실제 녹음 베이스 음색에서도 순위가 유지되는지 본다.

만드는 법(곡마다):
  1) 반주 = 원래 믹스를 htdemucs로 분리한 vocals+drums+other (runs/_sep 캐시 재사용).
     Slakh 개별 스템은 공개 미러에 창 길이만큼 없어서(tvergho 미러는 곡마다 무작위 30초 조각) 이렇게 한다.
     반주에 새어 남은 신스 베이스는 정답과 같은 음을 치므로 라벨을 오염시키지 않는다(음색만 섞인다).
  2) 베이스 = 정답 MIDI(all_src.mid, 베이스 프로그램, 30~90초, 중복 제거)를 실제 녹음 단음 샘플로 렌더링.
     - 프로그램 32(어쿠스틱) → 업라이트 세트, 그 밖 → 일렉 세트
     - 샘플은 가장 가까운 피치(±반음 이동은 리샘플링), 벨로시티 층은 가장 가까운 층
     - 피치 벤드(RPN 범위 반영)를 가변 재생 속도로 그대로 반영 → bass_bends.mid가 실제 렌더된 곡선이다
     - 노트 끝에서 60 ms 릴리스, 단성 겹침은 그대로 둔다
  3) 베이스 레벨 = 원래 믹스의 htdemucs 베이스 스템 RMS에 맞춘다.
출력: data/slakh_realbass/<song>/{mix.wav, bass_only.wav, bass.mid, bass_bends.mid, drums.mid}
샘플 인덱스: data/raw/samples_src/realbass_index.json
  {"electric": [{"file": 절대경로, "pitch": 40, "vel": 100}, ...], "upright": [...]}
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from pitchbend_eval import _rpn_range  # noqa: E402

SR = 44100
WIN = (30.0, 90.0)
BASS_PROGRAMS = range(32, 40)
RELEASE = 0.06
INDEX = ROOT / "data" / "raw" / "samples_src" / "realbass_index.json"
OUT = HERE / "data" / "slakh_realbass"
# 음색 세트: slakh_realbass = 일렉(Karoryfer Fashionbass), slakh_realbass_up = 업라이트 피치카토(D. Smolken).
# 일렉 샘플은 첫 40~50 ms가 피치 없는 손가락 잡음이고 기본음·2배음이 100~160 ms에 걸쳐 올라온다(실측).
# 한 라이브러리의 어택 특성이 결론을 좌우하지 않도록 어택이 빠른 업라이트로도 같은 곡을 렌더링한다.
SETS = {"slakh_realbass": "electric", "slakh_realbass_up": "upright",
        # R7 전용(벤드 곡선이 정확한 정답): 개발 = train 벤드곡, 테스트 = bendtest 20곡. 베이스 단독만 쓴다.
        "dev_bends_up": "upright", "bendtest_up": "upright"}
SRC_OF = {"dev_bends_up": ROOT / "data" / "raw" / "slakh_train", "bendtest_up": HERE / "data" / "_raw" / "slakh"}


def load_samples() -> dict:
    idx = json.loads(INDEX.read_text())
    bank = {}
    for kind, rows in idx.items():
        bank[kind] = []
        for r in rows:
            y, sr = sf.read(r["file"], always_2d=True)
            y = y.mean(axis=1)
            if sr != SR:
                import librosa

                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            # 앞 무음 제거(−50 dB)
            thr = np.max(np.abs(y)) * 10 ** (-50 / 20)
            k = int(np.argmax(np.abs(y) > thr))
            bank[kind].append({"y": y[k:].astype(np.float32), "pitch": r["pitch"], "vel": r.get("vel", 100)})
    return bank


def pick(bank_kind: list, pitch: int, vel: int) -> dict:
    d = min(abs(s["pitch"] - pitch) for s in bank_kind)
    near = [s for s in bank_kind if abs(s["pitch"] - pitch) == d]
    return min(near, key=lambda s: abs(s["vel"] - vel))


def render_note(sample: dict, pitch: int, dur: float, bend_fn, start: float) -> np.ndarray:
    """가변 재생 속도로 샘플을 읽는다. bend_fn(t절대) → cent."""
    n = int((dur + RELEASE) * SR)
    t = start + np.arange(n) / SR
    semis = (pitch - sample["pitch"]) + bend_fn(t) / 100.0
    rate = 2.0 ** (semis / 12.0)
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    y = sample["y"]
    valid = pos < len(y) - 1
    out = np.zeros(n, np.float32)
    out[valid] = np.interp(pos[valid], np.arange(len(y)), y)
    rel0 = int(dur * SR)
    if rel0 < n:
        out[rel0:] *= np.exp(-np.arange(n - rel0) / SR / (RELEASE / 4))
    fade = min(64, n)
    out[:fade] *= np.linspace(0, 1, fade)
    return out


def build_song(song: str, bank: dict, set_name: str = "slakh_realbass") -> dict:
    forced = SETS[set_name]
    src = SRC_OF.get(set_name, HERE / "data" / "_raw" / "slakh") / song / "all_src.mid"
    sep = HERE / "runs" / "_sep" / "slakh" / song
    pm = pretty_midi.PrettyMIDI(str(src))
    length = int((WIN[1] - WIN[0]) * SR)
    bass = np.zeros(length + SR, np.float32)
    gt_notes, seen = [], set()
    bend_pm = pretty_midi.PrettyMIDI()
    bend_inst = pretty_midi.Instrument(program=33)
    kinds_used = set()
    for inst in pm.instruments:
        if inst.is_drum or inst.program not in BASS_PROGRAMS:
            continue
        kind = forced
        kinds_used.add(kind)
        rng = _rpn_range(inst)
        rt = np.array([r[0] for r in rng])
        rv = np.array([r[1] for r in rng])
        pbs = sorted(inst.pitch_bends, key=lambda b: b.time)
        pt = np.array([b.time for b in pbs])
        pv = np.array([b.pitch for b in pbs], float)

        def bend_fn(t_abs, pt=pt, pv=pv, rt=rt, rv=rv):
            if len(pt) == 0:
                return np.zeros_like(t_abs)
            k = np.searchsorted(pt, t_abs, side="right") - 1
            val = np.where(k >= 0, pv[np.clip(k, 0, None)], 0.0)
            kr = np.searchsorted(rt, t_abs, side="right") - 1
            # ±2반음(±200 cent)으로 제한: 정답 파일(bass_bends.mid)이 ±2 범위라 렌더와 정확히 일치시킨다.
            return np.clip(val / 8192.0 * rv[np.clip(kr, 0, None)] * 100.0, -200.0, 200.0)

        for n in inst.notes:
            if not (WIN[0] <= n.start < WIN[1]):
                continue
            key = (n.pitch, round(n.start, 2))
            if key in seen:
                continue
            seen.add(key)
            end = min(n.end, WIN[1])
            s = pick(bank[kind], n.pitch, n.velocity)
            y = render_note(s, n.pitch, end - n.start, bend_fn, n.start)
            # 벨로시티 → 게인(층이 하나뿐인 경우에도 다이나믹스가 남도록)
            y *= (n.velocity / 127.0) ** 1.2
            i0 = int((n.start - WIN[0]) * SR)
            bass[i0:i0 + len(y)] += y[: len(bass) - i0]
            gt_notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start - WIN[0], end - WIN[0]))
        # 렌더에 실제로 쓴 벤드 곡선을 ±2반음 범위로 다시 기록(창 기준 시각)
        for b in pbs:
            if WIN[0] <= b.time < WIN[1]:
                cents = bend_fn(np.array([b.time]))[0]
                v = int(np.clip(round(cents / 200.0 * 8191), -8192, 8191))
                bend_inst.pitch_bends.append(pretty_midi.PitchBend(v, b.time - WIN[0]))
    bass = bass[:length]
    out = HERE / "data" / set_name / song
    out.mkdir(parents=True, exist_ok=True)
    cur_rms = float(np.sqrt(np.mean(bass ** 2))) or 1.0
    if set_name in SRC_OF:  # R7 전용: 베이스 단독, −20 dBFS RMS
        bass *= 0.1 / cur_rms
        bass = np.clip(bass, -0.99, 0.99)
    else:
        stems = {}
        for name in ("vocals", "drums", "other", "bass"):
            y, sr = sf.read(str(sep / f"{name}.wav"), always_2d=True)
            stems[name] = y[:length]
        ref_rms = float(np.sqrt(np.mean(stems["bass"] ** 2)))
        bass *= ref_rms / cur_rms
        acc = stems["vocals"] + stems["drums"] + stems["other"]
        mix = acc + bass[: len(acc), None]
        peak = np.max(np.abs(mix))
        if peak > 0.99:
            mix *= 0.99 / peak
            bass *= 0.99 / peak
        sf.write(out / "mix.wav", mix.astype(np.float32), SR, subtype="PCM_16")
    sf.write(out / "bass_only.wav", bass.astype(np.float32), SR, subtype="PCM_16")
    gpm = pretty_midi.PrettyMIDI()
    gi = pretty_midi.Instrument(program=33)
    gi.notes = sorted(gt_notes, key=lambda n: (n.start, n.pitch))
    gpm.instruments.append(gi)
    gpm.write(str(out / "bass.mid"))
    bend_inst.notes = list(gi.notes)
    bend_inst.control_changes = [pretty_midi.ControlChange(101, 0, 0.0), pretty_midi.ControlChange(100, 0, 0.0),
                                 pretty_midi.ControlChange(6, 2, 0.0)]
    bend_pm.instruments.append(bend_inst)
    bend_pm.write(str(out / "bass_bends.mid"))
    if set_name not in SRC_OF:
        shutil.copy(HERE / "data" / "slakh" / song / "drums.mid", out / "drums.mid")
    return {"song": song, "set": set_name, "bass_notes": len(gt_notes), "kinds": sorted(kinds_used),
            "bend_events": len(bend_inst.pitch_bends)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="slakh_realbass", choices=sorted(SETS))
    set_name = ap.parse_args().set
    bank = load_samples()
    man_path = HERE / "data" / "manifest.json"
    man = json.loads(man_path.read_text())
    if set_name == "dev_bends_up":
        songs = json.loads((HERE / "data" / "dev_bends.json").read_text())
    elif set_name == "bendtest_up":
        songs = [i["song"] for i in json.loads((HERE / "data" / "bendtest_manifest.json").read_text())]
    else:
        songs = [i["song"] for i in man["slakh"]]
    rows = []
    for song in songs:
        r = build_song(song, bank, set_name)
        rows.append(r)
        print(r, flush=True)
    if set_name in SRC_OF:
        (HERE / "data" / f"{set_name}_manifest.json").write_text(json.dumps(rows, indent=1))
        return 0
    man[set_name] = rows
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
