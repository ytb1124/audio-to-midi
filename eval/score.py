#!/usr/bin/env python3
"""Score a generated MIDI file against a ground-truth MIDI file.

Usage:
    python eval/score.py <generated.mid> <ground_truth.mid>

Reports:
    - mir_eval note transcription precision / recall / F1 (onset-only match)
    - octave-error breakdown via greedy onset matching (diagnostic only)
"""
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

ONSET_TOLERANCE_SEC = 0.05
PITCH_TOLERANCE_CENTS = 50.0

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi_pitch):
    return f"{NOTE_NAMES[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def midi_to_hz(midi_pitch):
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def load_notes(path: Path):
    """Return notes sorted by onset as (onset, offset, pitch) tuples."""
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch))
    notes.sort(key=lambda n: n[0])
    return notes


def mir_eval_metrics(ref_notes, est_notes):
    ref_intervals = np.array([[n[0], n[1]] for n in ref_notes]) if ref_notes else np.zeros((0, 2))
    ref_pitches = np.array([midi_to_hz(n[2]) for n in ref_notes])
    est_intervals = np.array([[n[0], n[1]] for n in est_notes]) if est_notes else np.zeros((0, 2))
    est_pitches = np.array([midi_to_hz(n[2]) for n in est_notes])

    precision, recall, f1, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=ONSET_TOLERANCE_SEC,
        pitch_tolerance=PITCH_TOLERANCE_CENTS,
        offset_ratio=None,
    )
    return precision, recall, f1, avg_overlap_ratio


def greedy_octave_diagnostic(ref_notes, est_notes):
    """Greedy nearest-onset match (ignoring pitch) to classify each ref note as
    correct / octave-error / other-error / missed, plus unmatched est notes (extra)."""
    unused_est = list(range(len(est_notes)))
    results = []

    for ref_idx, (r_on, r_off, r_pitch) in enumerate(ref_notes):
        best_idx = None
        best_dist = None
        for est_idx in unused_est:
            e_on = est_notes[est_idx][0]
            dist = abs(e_on - r_on)
            if dist <= ONSET_TOLERANCE_SEC and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = est_idx

        if best_idx is None:
            results.append(("missed", r_on, r_pitch, None))
            continue

        unused_est.remove(best_idx)
        e_pitch = est_notes[best_idx][2]

        if e_pitch == r_pitch:
            category = "correct"
        elif (e_pitch - r_pitch) % 12 == 0:
            category = "octave_error"
        else:
            category = "other_error"

        results.append((category, r_on, r_pitch, e_pitch))

    extra = [(est_notes[i][0], est_notes[i][2]) for i in unused_est]
    return results, extra


def detect_global_octave_shift(ref_notes, est_notes, tol=ONSET_TOLERANCE_SEC, min_fraction=0.6):
    """onset으로 매칭된 노트 쌍들의 pitch 차이(반음 단위)를 모아서, 옥타브
    배수(±12, ±24, ...)의 지배적인 shift가 있는지 확인한다.

    노트 하나하나가 아니라 파일 전체가 일관되게 한 옥타브 밀려있는 경우
    (익스포트 실수 등으로 정답 MIDI 자체의 옥타브 라벨이 잘못됐을 가능성)를
    감지해서 채점에서 분리하기 위함. min_fraction 이상의 매칭된 쌍이 같은
    옥타브 배수만큼 차이나면 "글로벌 시프트"로 판단한다.
    """
    unused_est = list(range(len(est_notes)))
    diffs = []

    for r_on, r_off, r_pitch in ref_notes:
        best_idx = None
        best_dist = None
        for est_idx in unused_est:
            e_on = est_notes[est_idx][0]
            dist = abs(e_on - r_on)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = est_idx

        if best_idx is None:
            continue

        unused_est.remove(best_idx)
        e_pitch = est_notes[best_idx][2]
        diffs.append(e_pitch - r_pitch)

    if not diffs:
        return None, 0.0, 0

    octave_diffs = [d for d in diffs if d % 12 == 0]
    if not octave_diffs:
        return None, 0.0, len(diffs)

    from collections import Counter

    counts = Counter(octave_diffs)
    shift, shift_count = counts.most_common(1)[0]
    fraction = shift_count / len(diffs)

    if shift != 0 and fraction >= min_fraction:
        return shift, fraction, len(diffs)

    return None, fraction, len(diffs)


def shift_notes(notes, semitones):
    return [(start, end, pitch + semitones) for start, end, pitch in notes]


def detect_global_time_offset(ref_notes, est_notes, max_lag=0.2, bin_sec=0.01):
    """온셋 임펄스 열 교차상관으로 전역 시간 오프셋을 감지한다.

    배경: 정답 MIDI가 오디오와 교차상관으로 정렬된 파일(sample3 등)은
    전역 오프셋이 온셋 허용오차(50ms)를 상당 부분 소진한 상태일 수 있고,
    파이프라인의 타이밍 규약이 바뀌면(온셋 스냅 등) 모든 매칭이 한꺼번에
    깨진다. 옥타브 시프트 보정과 같은 철학으로, 감지된 전역 시간 오프셋을
    보정한 재채점을 함께 보여준다. 반환: 초 단위 lag (est가 ref보다 늦으면 +).
    """
    import numpy as np

    if not ref_notes or not est_notes:
        return 0.0
    t_max = max(max(n[0] for n in ref_notes), max(n[0] for n in est_notes)) + 1.0
    nbins = int(t_max / bin_sec) + 1
    ref_v = np.zeros(nbins)
    est_v = np.zeros(nbins)
    for n in ref_notes:
        ref_v[int(n[0] / bin_sec)] = 1.0
    for n in est_notes:
        est_v[int(n[0] / bin_sec)] = 1.0
    max_bins = int(max_lag / bin_sec)
    best_lag, best_score = 0, -1.0
    for lag in range(-max_bins, max_bins + 1):
        if lag >= 0:
            s = float(np.dot(ref_v[: nbins - lag], est_v[lag:]))
        else:
            s = float(np.dot(ref_v[-lag:], est_v[: nbins + lag]))
        if s > best_score:
            best_score, best_lag = s, lag
    return best_lag * bin_sec


def time_shift_notes(notes, dt):
    return [(start + dt, end + dt, pitch) for start, end, pitch in notes]


def duration_diagnostic(ref_notes, est_notes, tol=ONSET_TOLERANCE_SEC):
    """Onset-matched (pitch-agnostic) pairs, compare durations. Diagnostic only."""
    unused_est = list(range(len(est_notes)))
    pairs = []

    for r_on, r_off, r_pitch in ref_notes:
        best_idx = None
        best_dist = None
        for est_idx in unused_est:
            e_on = est_notes[est_idx][0]
            dist = abs(e_on - r_on)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = est_idx

        if best_idx is None:
            continue

        unused_est.remove(best_idx)
        e_on, e_off, e_pitch = est_notes[best_idx]
        pairs.append(
            {
                "ref_start": r_on,
                "ref_dur": r_off - r_on,
                "est_start": e_on,
                "est_dur": e_off - e_on,
                "diff": (e_off - e_on) - (r_off - r_on),
                "pitch": r_pitch,
            }
        )

    return pairs


def main():
    if len(sys.argv) != 3:
        print("Usage: python eval/score.py <generated.mid> <ground_truth.mid>", file=sys.stderr)
        raise SystemExit(2)

    generated_path = Path(sys.argv[1]).expanduser().resolve()
    ground_truth_path = Path(sys.argv[2]).expanduser().resolve()

    ref_notes = load_notes(ground_truth_path)
    est_notes = load_notes(generated_path)

    print(f"Generated:    {generated_path} ({len(est_notes)} notes)")
    print(f"Ground truth: {ground_truth_path} ({len(ref_notes)} notes)")
    print()

    global_shift, shift_fraction, n_matched = detect_global_octave_shift(ref_notes, est_notes)
    if global_shift is not None:
        direction = "높게" if global_shift > 0 else "낮게"
        print(
            f"⚠ 전체 파일 단위 옥타브 시프트 감지: 매칭된 {n_matched}개 중 {shift_fraction:.1%}가 "
            f"정답보다 {abs(global_shift)//12}옥타브 {direction} 생성됨."
        )
        print("  (사용자 편집으로 일괄 수정 가능한 낮은 우선순위 문제 — 아래 '옥타브 보정 후' 점수가 실질적 채점 기준)")
        print()

    precision, recall, f1, avg_overlap_ratio = mir_eval_metrics(ref_notes, est_notes)
    print("mir_eval note transcription (onset-only, tol={}ms, pitch_tol={}cents):".format(
        int(ONSET_TOLERANCE_SEC * 1000), PITCH_TOLERANCE_CENTS
    ))
    print(f"  precision: {precision:.4f}")
    print(f"  recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print()

    diagnostics, extra = greedy_octave_diagnostic(ref_notes, est_notes)
    counts = {"correct": 0, "octave_error": 0, "other_error": 0, "missed": 0}
    for category, *_ in diagnostics:
        counts[category] += 1

    print("Octave-error diagnostic (greedy onset match, pitch-agnostic):")
    print(f"  correct:      {counts['correct']}")
    print(f"  octave_error: {counts['octave_error']}")
    print(f"  other_error:  {counts['other_error']}")
    print(f"  missed:       {counts['missed']}")
    print(f"  extra (unmatched generated notes): {len(extra)}")
    print()

    octave_errors = [d for d in diagnostics if d[0] == "octave_error"]
    if octave_errors:
        print(f"Octave errors ({len(octave_errors)}):")
        for category, r_on, r_pitch, e_pitch in octave_errors[:50]:
            print(f"  {r_on:8.2f}s  ref {note_name(r_pitch):4s} -> got {note_name(e_pitch):4s}")
        if len(octave_errors) > 50:
            print(f"  ... and {len(octave_errors) - 50} more")
        print()

    dur_pairs = duration_diagnostic(ref_notes, est_notes)
    if dur_pairs:
        diffs = np.array([p["diff"] for p in dur_pairs])
        print(f"Duration diagnostic (onset-matched pairs, pitch-agnostic, n={len(dur_pairs)}):")
        print(f"  mean(gen-ref):   {diffs.mean():+.4f}s")
        print(f"  median(gen-ref): {np.median(diffs):+.4f}s")
        print(f"  std:             {diffs.std():.4f}s")
        print(f"  |diff|>0.1s:     {(np.abs(diffs) > 0.1).sum()} / {len(diffs)}")
        print(f"  |diff|>0.3s:     {(np.abs(diffs) > 0.3).sum()} / {len(diffs)}")

        worst = sorted(dur_pairs, key=lambda p: p["diff"])
        print("\n  worst under-held (generated too short):")
        for p in worst[:5]:
            print(f"    {p['ref_start']:8.2f}s  ref_dur={p['ref_dur']:.3f}s  gen_dur={p['est_dur']:.3f}s  diff={p['diff']:+.3f}s")
        print("  worst over-held (generated too long):")
        for p in worst[-5:]:
            print(f"    {p['ref_start']:8.2f}s  ref_dur={p['ref_dur']:.3f}s  gen_dur={p['est_dur']:.3f}s  diff={p['diff']:+.3f}s")

    time_lag = detect_global_time_offset(ref_notes, est_notes)
    if abs(time_lag) > 0.015:
        print()
        print("=" * 70)
        print(f"시간 오프셋 보정 후 (전역 lag {time_lag*1000:+.0f}ms — 정답 온셋을 이동해 재채점)")
        print("=" * 70)
        ref_t = time_shift_notes(ref_notes, time_lag)
        if global_shift is not None:
            ref_t = shift_notes(ref_t, global_shift)
        p3, r3, f3, _ = mir_eval_metrics(ref_t, est_notes)
        print(f"  precision: {p3:.4f}  recall: {r3:.4f}  F1: {f3:.4f}")
        d3, e3 = greedy_octave_diagnostic(ref_t, est_notes)
        c3 = {"correct": 0, "octave_error": 0, "other_error": 0, "missed": 0}
        for category, *_ in d3:
            c3[category] += 1
        print(
            f"  correct: {c3['correct']}  octave_error: {c3['octave_error']}  "
            f"other_error: {c3['other_error']}  missed: {c3['missed']}  extra: {len(e3)}"
        )

    if global_shift is not None:
        print()
        print("=" * 70)
        print(f"옥타브 보정 후 (정답을 {global_shift:+d}반음 이동, 전체 시프트 무시하고 재채점)")
        print("=" * 70)

        ref_shifted = shift_notes(ref_notes, global_shift)

        precision2, recall2, f1_2, _ = mir_eval_metrics(ref_shifted, est_notes)
        print(f"  precision: {precision2:.4f}  recall: {recall2:.4f}  F1: {f1_2:.4f}")

        diagnostics2, extra2 = greedy_octave_diagnostic(ref_shifted, est_notes)
        counts2 = {"correct": 0, "octave_error": 0, "other_error": 0, "missed": 0}
        for category, *_ in diagnostics2:
            counts2[category] += 1
        print(
            f"  correct: {counts2['correct']}  sporadic_octave_error: {counts2['octave_error']}  "
            f"other_error: {counts2['other_error']}  missed: {counts2['missed']}  extra: {len(extra2)}"
        )

        sporadic = [d for d in diagnostics2 if d[0] == "octave_error"]
        if sporadic:
            print(f"\n  남은 산발적 옥타브 오류 ({len(sporadic)}건, 전체 시프트 보정 후에도 남은 것 — 진짜 문제):")
            for category, r_on, r_pitch, e_pitch in sporadic[:30]:
                print(f"    {r_on:8.2f}s  ref {note_name(r_pitch):4s} -> got {note_name(e_pitch):4s}")
            if len(sporadic) > 30:
                print(f"    ... and {len(sporadic) - 30} more")


if __name__ == "__main__":
    main()
