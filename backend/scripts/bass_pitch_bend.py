"""단성 베이스 MIDI에 오디오 f0 곡선으로 피치 벤드를 기록한다 (opt-in, 2026-10-25, R7).

원칙: 벤드가 없는 노트에 흔들림을 만들지 않는다. 그래서
  1) 노트마다 f0를 노트 피치 기준 cent로 바꾸고(옥타브 접힘 보정), 중앙값 필터로 평활한다.
  2) "벤드 노트" 판정: 평활 곡선이 min_bend cent 이상 벗어난 구간이 min_dur 이상 이어질 때만.
     (slide_only=False면 판정 없이 모든 노트에 데드밴드 곡선을 쓴다 — 비교용)
  3) 벤드 노트에만 데드밴드(|c| < deadband → 0)를 적용한 곡선을 ±range 반음 범위로 기록한다.
  4) 노트 시작 직전에 0으로 되돌린다(이전 노트 벤드가 새지 않게).

파라미터 선택은 Slakh train(개발셋)에서만 했다 — eval/heldout/pitchbend_eval.py.

CLI: python bass_pitch_bend.py <audio> <in.mid> <out.mid> [--method pyin] [--f0-npz cache.npz]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

BEND_RANGE_SEMITONES = 2  # RPN 0 기본값과 같다. 프론트/DAW 재생이 별도 설정 없이 맞는다.


@dataclass
class BendParams:
    # 기본값 = pYIN의 개발셋(dev_bends_up) 최적값. 테스트(bendtest_up) 1회: F1 0.813, 벤드 곡선 오차
    # 91.9 → 31.7 cent, 무벤드 노트 0.17 cent. 실녹음(Bass_2·Mosquito) 안정 노트 오차 악화 없음 — eval/BEST_RESULTS.md
    smooth_ms: float = 30.0
    deadband: float = 35.0       # cent: 이보다 작은 흔들림은 0으로 둔다
    min_bend: float = 50.0       # cent, 벤드 노트 판정 문턱(slide_only일 때)
    min_dur_ms: float = 40.0     # 문턱 이상이 이어져야 하는 길이
    attack_skip_ms: float = 30.0  # 어택 직후 f0는 불안정하므로 무시
    min_conf: float = 0.05       # pYIN voiced 확률
    min_voiced_frac: float = 0.5
    slide_only: bool = False     # 개발셋에서 데드밴드만으로 무벤드 흔들림 조건(≤2 cent)을 만족했다
    step_cents: float = 5.0      # 이벤트 간소화: 이 이상 바뀔 때만 기록


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 3:
        return x
    k = min(k | 1, len(x) if len(x) % 2 else len(x) - 1)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(win, axis=1)


def note_curve(pitch: int, start: float, end: float, t: np.ndarray, f0: np.ndarray, conf: np.ndarray,
               p: BendParams):
    """노트 하나의 (times, cents) 벤드 곡선. 벤드가 아니라고 판정하면 None."""
    lo = start + p.attack_skip_ms / 1000.0
    m = (t >= lo) & (t < end)
    if m.sum() < 3:
        return None
    tt, ff, cc = t[m], f0[m], conf[m]
    ok = (ff > 0) & (cc >= p.min_conf)
    if ok.mean() < p.min_voiced_frac:
        return None
    cents = np.full(len(tt), np.nan)
    c = 1200 * np.log2(ff[ok] / _hz(pitch))
    cents[ok] = c - 1200 * np.round(c / 1200)  # 옥타브 오판독 접기
    # 무성 프레임은 직전 값으로 채운다(평활 전에)
    idx = np.where(~np.isnan(cents), np.arange(len(cents)), 0)
    np.maximum.accumulate(idx, out=idx)
    cents = cents[idx]
    if np.isnan(cents[0]):
        first = np.flatnonzero(~np.isnan(cents))
        cents[: first[0]] = cents[first[0]]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    sm = _median_smooth(cents, int(round(p.smooth_ms / 1000.0 / dt)))
    if p.slide_only:
        over = np.abs(sm) >= p.min_bend
        run = best = 0
        for o in over:
            run = run + 1 if o else 0
            best = max(best, run)
        if best * dt * 1000.0 < p.min_dur_ms:
            return None
    curve = np.where(np.abs(sm) < p.deadband, 0.0, sm)
    lim = 100.0 * BEND_RANGE_SEMITONES
    curve = np.clip(curve, -lim, lim)
    if not np.any(curve):
        return None
    return tt, curve


def compute_bends(notes, t, f0, conf, p: BendParams):
    """notes: (start, end, pitch) 목록 → 노트 인덱스별 곡선(dict)."""
    out = {}
    for i, (s, e, pitch) in enumerate(notes):
        r = note_curve(pitch, s, e, t, f0, conf, p)
        if r is not None:
            out[i] = r
    return out


def write_bends(pm, inst, notes, bends, p: BendParams):
    """pretty_midi 악기에 벤드 이벤트를 쓴다. 벤드 노트 앞뒤로 0을 넣어 새지 않게 한다."""
    import pretty_midi

    lim = 100.0 * BEND_RANGE_SEMITONES
    events = []
    for i, (tt, curve) in bends.items():
        s, e, _ = notes[i]
        events.append((max(0.0, s - 0.002), 0))
        last = None
        for ti, ci in zip(tt, curve):
            if last is None or abs(ci - last) >= p.step_cents:
                events.append((float(ti), int(round(ci / lim * 8191))))
                last = ci
        events.append((float(e), 0))
    events.sort()
    inst.pitch_bends = [pretty_midi.PitchBend(int(np.clip(v, -8192, 8191)), t) for t, v in events]
    return len(bends)


def apply(audio: str, midi_in: str, midi_out: str, method: str = "pyin", f0_npz: str | None = None,
          params: BendParams | None = None) -> dict:
    import pretty_midi

    p = params or BendParams()
    if f0_npz and Path(f0_npz).exists():
        z = np.load(f0_npz)
        t, f0, conf = z["t"], z["f0"], z["conf"]
    else:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from bass_f0_tracks import extract

        t, f0, conf = extract(audio, method)
    pm = pretty_midi.PrettyMIDI(midi_in)
    inst = next(i for i in pm.instruments if not i.is_drum)
    inst.notes.sort(key=lambda n: n.start)
    notes = [(n.start, n.end, n.pitch) for n in inst.notes]
    bends = compute_bends(notes, t, f0, conf, p)
    n = write_bends(pm, inst, notes, bends, p)
    # RPN 0(벤드 범위)을 명시해 DAW가 ±2반음으로 해석하게 한다.
    inst.control_changes = [c for c in inst.control_changes if c.number not in (100, 101, 6, 38)] + [
        pretty_midi.ControlChange(101, 0, 0.0), pretty_midi.ControlChange(100, 0, 0.0),
        pretty_midi.ControlChange(6, BEND_RANGE_SEMITONES, 0.0), pretty_midi.ControlChange(38, 0, 0.0)]
    pm.write(midi_out)
    return {"bendNotes": n, "notes": len(notes), "method": method, "rangeSemitones": BEND_RANGE_SEMITONES,
            "params": asdict(p)}


def main() -> int:
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi_in")
    ap.add_argument("midi_out")
    ap.add_argument("--method", default="pyin")
    ap.add_argument("--f0-npz")
    a = ap.parse_args()
    print(json.dumps(apply(a.audio, a.midi_in, a.midi_out, a.method, a.f0_npz)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
