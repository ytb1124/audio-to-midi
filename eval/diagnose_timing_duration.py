#!/usr/bin/env python3
"""정답 MIDI가 없는 실전 오디오용 타이밍/듀레이션 진단.

각 노트에 대해:
  - onset_dev: note.start와 오디오 온셋 피크(±80ms 내 최대 온셋 프레임)의 차이.
    음수 = 노트가 실제 어택보다 일찍 시작 (박 앞섬), 양수 = 늦게 시작.
  - sustain_end: note pitch의 CQT 에너지가 어택 피크의 25% 아래로 처음
    떨어지는 시각(다음 노트 시작 전까지). dur_dev = note.end - sustain_end.
    음수 = 서스테인이 잘림(너무 짧음), 양수 = 실제 감쇠보다 길게 유지.

Usage:
    python eval/diagnose_timing_duration.py <audio.wav> <generated.mid> [--csv out.csv]
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import librosa
    import pretty_midi

    print(f"Loading audio: {args.audio}", file=sys.stderr)
    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    hop = 256

    # 온셋 envelope (파이프라인과 동일 설정)
    try:
        y_h, y_p = librosa.effects.hpss(y)
        y_for = 0.65 * y_h + 0.35 * y_p
    except Exception:
        y_for = y
    env = librosa.onset.onset_strength(y=y_for, sr=sr, hop_length=hop, aggregate=np.median, fmax=2200)
    env_n = (env - np.percentile(env, 10)) / max(np.percentile(env, 95) - np.percentile(env, 10), 1e-9)

    # CQT (C1~C7, 12 bins/oct)
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"), n_bins=72, bins_per_octave=12))
    C = np.log1p(10.0 * C)
    base_midi = 24

    def fr(t):
        return int(t * sr / hop)

    pm = pretty_midi.PrettyMIDI(args.midi)
    notes = sorted(
        [n for i in pm.instruments if not i.is_drum for n in i.notes],
        key=lambda n: n.start,
    )
    print(f"Notes: {len(notes)}", file=sys.stderr)

    rows = []
    for idx, n in enumerate(notes):
        # --- onset deviation: ±80ms 창에서 온셋 피크 프레임 찾기 ---
        a = max(0, fr(n.start - 0.08))
        b = min(len(env_n), fr(n.start + 0.08) + 1)
        onset_dev = None
        onset_peak = 0.0
        if b > a:
            seg = env_n[a:b]
            onset_peak = float(seg.max())
            if onset_peak >= 0.25:  # 뚜렷한 어택이 있는 노트만 타이밍 평가
                peak_t = (a + int(np.argmax(seg))) * hop / sr
                onset_dev = float(n.start - peak_t)

        # --- sustain end: note pitch CQT가 어택 피크의 25% 아래로 떨어지는 시각 ---
        pidx = int(n.pitch) - base_midi
        dur_dev = None
        sustain_end = None
        if 0 <= pidx < C.shape[0]:
            next_start = notes[idx + 1].start if idx + 1 < len(notes) else n.end + 2.0
            horizon = min(max(n.end, next_start), n.start + 4.0)
            fa = fr(n.start)
            fbb = min(C.shape[1], fr(horizon) + 1)
            if fbb > fa + 2:
                track = C[pidx, fa:fbb]
                peak_v = float(track[: max(3, fr(n.start + 0.08) - fa)].max())
                if peak_v > 0.3:
                    thresh = 0.25 * peak_v
                    below = np.nonzero(track < thresh)[0]
                    if len(below):
                        sustain_end = (fa + int(below[0])) * hop / sr
                    else:
                        sustain_end = horizon  # 끝까지 유지 (다음 노트 직전까지)
                    dur_dev = float(n.end - sustain_end)

        rows.append(
            {
                "start": float(n.start),
                "end": float(n.end),
                "dur": float(n.end - n.start),
                "pitch": int(n.pitch),
                "onset_peak": round(onset_peak, 2),
                "onset_dev": onset_dev,
                "sustain_end": sustain_end,
                "dur_dev": dur_dev,
            }
        )

    od = np.array([r["onset_dev"] for r in rows if r["onset_dev"] is not None])
    print()
    print(f"=== 타이밍 (뚜렷한 어택 {len(od)}노트) ===")
    print(f"  onset_dev: mean={od.mean()*1000:+.0f}ms median={np.median(od)*1000:+.0f}ms std={od.std()*1000:.0f}ms")
    print(f"  |dev|>30ms: {(np.abs(od)>0.03).sum()}  |dev|>50ms: {(np.abs(od)>0.05).sum()}")
    early = sorted([r for r in rows if r["onset_dev"] is not None and r["onset_dev"] < -0.03], key=lambda r: r["onset_dev"])
    late = sorted([r for r in rows if r["onset_dev"] is not None and r["onset_dev"] > 0.03], key=lambda r: -r["onset_dev"])
    print(f"\n  30ms 이상 빠른 노트 ({len(early)}):")
    for r in early[:15]:
        print(f"    {r['start']:7.2f}s p={r['pitch']} dev={r['onset_dev']*1000:+.0f}ms dur={r['dur']:.3f}")
    print(f"  30ms 이상 늦은 노트 ({len(late)}):")
    for r in late[:15]:
        print(f"    {r['start']:7.2f}s p={r['pitch']} dev={r['onset_dev']*1000:+.0f}ms dur={r['dur']:.3f}")

    dd = np.array([r["dur_dev"] for r in rows if r["dur_dev"] is not None])
    print()
    print(f"=== 듀레이션/서스테인 (측정 가능 {len(dd)}노트) ===")
    print(f"  dur_dev(end - 실제감쇠): mean={dd.mean()*1000:+.0f}ms median={np.median(dd)*1000:+.0f}ms")
    print(f"  100ms 이상 잘림(서스테인 손실): {(dd < -0.1).sum()}   300ms 이상 잘림: {(dd < -0.3).sum()}")
    print(f"  100ms 이상 초과(과지속): {(dd > 0.1).sum()}   300ms 이상 초과: {(dd > 0.3).sum()}")
    cut = sorted([r for r in rows if r["dur_dev"] is not None and r["dur_dev"] < -0.1], key=lambda r: r["dur_dev"])
    over = sorted([r for r in rows if r["dur_dev"] is not None and r["dur_dev"] > 0.1], key=lambda r: -r["dur_dev"])
    print(f"\n  가장 심하게 잘린 노트:")
    for r in cut[:15]:
        print(f"    {r['start']:7.2f}s p={r['pitch']} dur={r['dur']:.3f} 실제감쇠까지 {-r['dur_dev']:.3f}s 더 있음")
    print(f"  가장 심하게 긴 노트:")
    for r in over[:15]:
        print(f"    {r['start']:7.2f}s p={r['pitch']} dur={r['dur']:.3f} 실제감쇠보다 {r['dur_dev']:.3f}s 초과")

    if args.csv:
        import csv as csvmod

        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved CSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
