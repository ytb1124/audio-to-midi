"""박자 격자(opt-in, 2026-10-25): 비트·다운비트 → MIDI 템포 맵 + 선택적 퀀타이즈.

- 템포 맵: 비트 i를 정확히 i번째 4분음표 틱에 놓는 가변 템포 이벤트를 쓴다. 노트의 실제 시각(초)은
  바뀌지 않는다(재생은 그대로, DAW 마디선만 연주에 맞춰진다). 첫 다운비트가 마디 시작에 오도록 앞을 채운다.
- 박자: 다운비트 사이 비트 수의 중앙값(3 또는 4, 그 밖이면 4).
- 퀀타이즈 강도 q(0~1): 노트 시작을 가장 가까운 격자(비트를 subdiv 등분)로 q만큼 당긴다. 길이는 유지.
  q=0이면 노트 시각을 전혀 바꾸지 않는다(피아노 기본값).
- 비트 추적기는 eval/heldout/beats_eval 결과로 고른 것을 격리 venv(.venv-beat)로 부르고, 없으면 librosa로 폴백.

CLI: python beat_grid.py <audio> <in.mid> <out.mid> [--quantize 0.0] [--subdiv 4] [--beats beats.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PPQ = 480
ROOT = Path(__file__).resolve().parents[2]
TRACKER_SCRIPT = Path(__file__).resolve().parent / "beat_track.py"


def track_beats(audio: str, tracker: str | None = None) -> dict:
    """{"beats": [...], "downbeats": [...], "tracker": name}. 격리 venv가 없으면 librosa 폴백."""
    tracker = tracker or os.environ.get("BEAT_TRACKER", "beat_this")
    py = ROOT / ".venv-beat" / "bin" / "python"
    if py.exists() and TRACKER_SCRIPT.exists():
        try:
            cp = subprocess.run([str(py), str(TRACKER_SCRIPT), audio, "--tracker", tracker],
                                check=True, capture_output=True, text=True)
            line = [ln for ln in cp.stdout.splitlines() if ln.strip().startswith("{")][-1]
            out = json.loads(line)
            out["tracker"] = tracker
            return out
        except (subprocess.CalledProcessError, OSError, IndexError, ValueError) as error:
            print(f"{tracker} beat tracking failed ({error}); using librosa.", file=sys.stderr)
    import librosa

    y, sr = librosa.load(audio, sr=22050, mono=True)
    _, frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(frames, sr=sr).tolist()
    return {"beats": beats, "downbeats": beats[::4], "tracker": "librosa"}


def _meter(beats: np.ndarray, downbeats: np.ndarray) -> int:
    if len(downbeats) < 2:
        return 4
    idx = np.searchsorted(beats, downbeats - 1e-3)
    m = int(np.median(np.diff(idx))) if len(idx) > 1 else 4
    return m if m in (2, 3, 4, 6) else 4


def tick_map(beats: np.ndarray, downbeats: np.ndarray, end_time: float):
    """초 → 틱 단조 사상(구간 선형)과 템포 이벤트. 비트 i ↔ 틱 base + i*PPQ."""
    bpb = _meter(beats, downbeats)
    j = int(np.searchsorted(beats, downbeats[0] - 1e-3)) if len(downbeats) else 0
    tick_j = int(np.ceil(j / bpb)) * bpb * PPQ
    if tick_j - j * PPQ <= 0 and beats[0] > 1e-3:
        tick_j += bpb * PPQ
    base = tick_j - j * PPQ
    knots_t = [0.0] + list(beats)
    knots_k = [0] + [base + i * PPQ for i in range(len(beats))]
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    tail_t = max(end_time, beats[-1]) + ibi
    knots_t.append(tail_t)
    knots_k.append(knots_k[-1] + int(round((tail_t - beats[-1]) / ibi * PPQ)))
    kt, kk = np.array(knots_t), np.array(knots_k, float)
    tempos = []  # (tick, microseconds per quarter)
    for a in range(len(kt) - 1):
        dk = kk[a + 1] - kk[a]
        if dk <= 0:
            continue
        us = (kt[a + 1] - kt[a]) / (dk / PPQ) * 1e6
        tempos.append((int(kk[a]), int(round(us))))
    return kt, kk, tempos, bpb, base


def quantize_times(starts: np.ndarray, beats: np.ndarray, q: float, subdiv: int = 4) -> np.ndarray:
    if q <= 0 or len(beats) < 2:
        return starts
    grid = []
    for a, b in zip(beats[:-1], beats[1:]):
        grid += list(a + (b - a) * np.arange(subdiv) / subdiv)
    grid.append(beats[-1])
    g = np.array(grid)
    k = np.clip(np.searchsorted(g, starts), 1, len(g) - 1)
    near = np.where(np.abs(g[k] - starts) < np.abs(g[k - 1] - starts), g[k], g[k - 1])
    # 비트가 덮지 않는 구간(앞·뒤 비트 하나 너머)의 노트는 건드리지 않는다.
    ibi = float(np.median(np.diff(beats)))
    inside = (starts >= beats[0] - ibi / subdiv) & (starts <= beats[-1] + ibi / subdiv)
    return np.where(inside, starts + q * (near - starts), starts)


def apply_grid(midi_in: str, midi_out: str, beats, downbeats, quantize: float = 0.0, subdiv: int = 4) -> dict:
    """MIDI를 템포 맵이 있는 파일로 다시 쓴다. 노트·벤드·CC의 초 단위 시각은 q=0이면 그대로다."""
    import mido
    import pretty_midi

    beats = np.asarray(sorted(beats), float)
    downbeats = np.asarray(sorted(downbeats), float)
    if len(beats) < 2:
        raise ValueError("beat grid needs at least two beats")
    pm = pretty_midi.PrettyMIDI(midi_in)
    end = pm.get_end_time()
    kt, kk, tempos, bpb, base = tick_map(beats, downbeats, end)

    def tick(t: float) -> int:
        return int(round(np.interp(t, kt, kk, right=kk[-1] + (t - kt[-1]) / (kt[-1] - kt[-2]) * (kk[-1] - kk[-2]))))

    mf = mido.MidiFile(ticks_per_beat=PPQ, type=1)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=bpb, denominator=4, time=0))
    ev = [(t, mido.MetaMessage("set_tempo", tempo=us)) for t, us in tempos]
    last = 0
    for t, msg in sorted(ev, key=lambda x: x[0]):
        msg.time = t - last
        last = t
        meta.append(msg)
    mf.tracks.append(meta)
    moved = 0
    for ch_i, inst in enumerate(pm.instruments):
        tr = mido.MidiTrack()
        ch = 9 if inst.is_drum else min(ch_i if ch_i < 9 else ch_i + 1, 15)
        if inst.name:
            tr.append(mido.MetaMessage("track_name", name=inst.name, time=0))
        events = [(0, 0, mido.Message("program_change", program=inst.program, channel=ch))]
        starts = np.array([n.start for n in inst.notes])
        new_starts = quantize_times(starts, beats, quantize, subdiv) if len(starts) else starts
        for n, s2 in zip(inst.notes, new_starts):
            d = s2 - n.start
            moved += abs(d) > 1e-4
            a, b = tick(n.start + d), max(tick(n.end + d), tick(n.start + d) + 1)
            events.append((a, 2, mido.Message("note_on", note=n.pitch, velocity=n.velocity, channel=ch)))
            events.append((b, 1, mido.Message("note_off", note=n.pitch, velocity=0, channel=ch)))
        for c in inst.control_changes:
            events.append((tick(c.time), 0, mido.Message("control_change", control=c.number, value=c.value, channel=ch)))
        for p in inst.pitch_bends:
            events.append((tick(p.time), 0, mido.Message("pitchwheel", pitch=int(p.pitch), channel=ch)))
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for t, _, msg in events:
            msg.time = t - last
            last = t
            tr.append(msg)
        mf.tracks.append(tr)
    mf.save(midi_out)
    return {"bpm": round(60.0 / float(np.median(np.diff(beats))), 2), "timeSignature": f"{bpb}/4",
            "beats": len(beats), "downbeats": len(downbeats), "quantize": quantize, "subdiv": subdiv,
            "notesMoved": int(moved), "firstBeatTick": int(base)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi_in")
    ap.add_argument("midi_out")
    ap.add_argument("--quantize", type=float, default=0.0)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--beats", help="이미 계산한 비트 JSON({beats, downbeats})")
    ap.add_argument("--tracker")
    ap.add_argument("--save-beats", help="비트/다운비트를 이 JSON으로 저장(API 응답용)")
    a = ap.parse_args()
    bt = json.loads(Path(a.beats).read_text()) if a.beats else track_beats(a.audio, a.tracker)
    if a.save_beats:
        Path(a.save_beats).write_text(json.dumps({"beats": bt["beats"], "downbeats": bt["downbeats"]}))
    info = apply_grid(a.midi_in, a.midi_out, bt["beats"], bt["downbeats"], a.quantize, a.subdiv)
    info["tracker"] = bt.get("tracker")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
