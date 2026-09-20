#!/usr/bin/env python3
"""
Gated shadow A/B — 2x2 factorial 비교 (baseline / MTL-O / MTL-F / MTL-O+F).
(bass-note-verifier, 2026-07-13 — 사용자 명세)

기존 baseline 최종 MIDI에 octave/offset 리뷰 스테이지를 적용해 4개 버전을
만들고, 정답 있는 파일은 지표로, 없는 실전 파일은 audit으로 비교한다.
(전체 파이프라인 재실행 없이 리뷰 스테이지만 — 효과 격리에 오히려 유리.
 채택 결정 후 실제 삽입 위치로 파이프라인에 연결 + 웹 라운드트립 확인.)
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "scripts"))

import mir_eval  # noqa: E402
import pretty_midi  # noqa: E402
from score import (detect_global_octave_shift, greedy_octave_diagnostic,  # noqa: E402
                   load_notes, midi_to_hz)
import bass_mtl_review as review  # noqa: E402


def f1(ref, est, offset_ratio=None):
    if not ref or not est:
        return 0.0
    ri = np.array([[n[0], n[1]] for n in ref]); rp = np.array([midi_to_hz(n[2]) for n in ref])
    ei = np.array([[n[0], n[1]] for n in est]); ep = np.array([midi_to_hz(n[2]) for n in est])
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=offset_ratio)
    return f


def diag(ref, est):
    res, extra = greedy_octave_diagnostic(ref, est)
    c = {"correct": 0, "octave_error": 0, "other_error": 0, "missed": 0}
    for cat, *_ in res:
        c[cat] += 1
    return c, len(extra)


def offset_mae(ref, est, tol=0.05):
    """onset-매칭된 쌍의 duration 오차 MAE."""
    unused = list(range(len(est)))
    errs = []
    for r_on, r_off, r_p in ref:
        best, bd = None, None
        for k in unused:
            d = abs(est[k][0] - r_on)
            if d <= tol and (bd is None or d < bd):
                bd, best = d, k
        if best is None:
            continue
        unused.remove(best)
        e_on, e_off, e_p = est[best]
        errs.append(abs((e_off - e_on) - (r_off - r_on)))
    return float(np.mean(errs)) if errs else 0.0


def apply_reviews(baseline_mid, wav, out_dir, do_octave, do_offset):
    tmp = Path(tempfile.mkdtemp())
    cur = tmp / "cur.mid"
    shutil.copy(baseline_mid, cur)
    if do_octave:
        review.mtl_octave_review(wav, cur, cur, out_dir)
    if do_offset:
        review.mtl_offset_review(wav, cur, cur, out_dir)
    return cur


def gt_row(label, est_notes, ref_notes, ref_shifted):
    on = f1(ref_shifted, est_notes)
    off = f1(ref_shifted, est_notes, offset_ratio=0.2)
    c, extra = diag(ref_shifted, est_notes)
    mae = offset_mae(ref_shifted, est_notes)
    return (f"  {label:10s} onsetF1={on:.4f} offAwareF1={off:.4f} "
            f"octErr={c['octave_error']} other={c['other_error']} missed={c['missed']} "
            f"extra={extra} offMAE={mae*1000:.0f}ms")


def eval_gt(song, wav, gt_mid, baseline_dir):
    print(f"\n===== {song} (정답 있음) =====")
    baseline = Path(baseline_dir) / "bass_hybrid.mid"
    ref = load_notes(gt_mid)
    est_base = load_notes(baseline)
    shift, _, _ = detect_global_octave_shift(ref, est_base)
    ref_shifted = [(s, e, p + shift) for s, e, p in ref] if shift else ref
    if shift:
        print(f"  (전역 옥타브 시프트 {shift:+d} 보정 후 채점)")

    variants = [
        ("Baseline", False, False), ("MTL-O", True, False),
        ("MTL-F", False, True), ("MTL-O+F", True, True),
    ]
    for label, do_o, do_f in variants:
        if not do_o and not do_f:
            est = est_base
        else:
            m = apply_reviews(baseline, wav, baseline_dir, do_o, do_f)
            est = load_notes(m)
        print(gt_row(label, est, ref, ref_shifted))


def eval_real(song, wav, baseline_dir):
    print(f"\n===== {song} (정답 없음 — audit) =====")
    baseline = Path(baseline_dir) / "bass_hybrid.mid"
    base_notes = load_notes(baseline)
    base_map = {round(n[0], 3): n[2] for n in base_notes}

    for label, do_o, do_f in [("MTL-O", True, False), ("MTL-F", False, True), ("MTL-O+F", True, True)]:
        m = apply_reviews(baseline, wav, baseline_dir, do_o, do_f)
        est = load_notes(m)
        oct_changes = 0
        deltas = []
        short = sum(1 for n in est if (n[1] - n[0]) < 0.045)
        long_ = sum(1 for n in est if (n[1] - n[0]) > 1.5)
        for n in est:
            k = round(n[0], 3)
            if k in base_map and base_map[k] != n[2]:
                oct_changes += 1
        # offset delta 분포
        base_end = {round(n[0], 3): n[1] for n in base_notes}
        for n in est:
            k = round(n[0], 3)
            if k in base_end:
                deltas.append((n[1] - base_end[k]) * 1000)
        darr = np.array(deltas) if deltas else np.array([0.0])
        base_short = sum(1 for n in base_notes if (n[1] - n[0]) < 0.045)
        base_long = sum(1 for n in base_notes if (n[1] - n[0]) > 1.5)
        print(f"  {label:8s} 노트 {len(est)} (base {len(base_notes)}) | octave변경 {oct_changes} | "
              f"offset delta median {np.median(darr):+.0f}ms p10 {np.percentile(darr,10):+.0f} "
              f"p90 {np.percentile(darr,90):+.0f} | 초단음<45ms {short}(base {base_short}) "
              f"장음>1.5s {long_}(base {base_long})")


def main():
    gt_files = [
        ("bass_test", "eval/ground_truth/bass_test.wav", "eval/ground_truth/bass_test.mid", "eval/runs/snap3_bass_test"),
        ("bass_sample2", "eval/ground_truth/bass_sample2.wav", "eval/ground_truth/bass_sample2.mid", "eval/runs/snap3_bass_sample2"),
        ("bass_sample3", "eval/ground_truth/bass_sample3.wav", "eval/ground_truth/bass_sample3_fullcorrected.mid", "eval/runs/snap3_bass_sample3"),
        ("bass_sample4", "eval/ground_truth/bass_sample4.wav", "eval/ground_truth/bass_sample4.mid", "eval/runs/snap3_bass_sample4"),
        ("bass_sample5", "eval/ground_truth/bass_sample5.wav", "eval/ground_truth/bass_sample5.mid", "eval/runs/snap3_bass_sample5"),
    ]
    real_files = [
        ("Bass_2", "/Users/taebin/Desktop/Bass_2.wav", "eval/runs/snap3_bass2"),
        ("Mosquito", "/Users/taebin/Desktop/전심/전공레슨/Mosquito_Multi-Track/Bass_D.I.wav", "eval/runs/snap3_mosquito"),
    ]
    for song, wav, gt, d in gt_files:
        if Path(d, "bass_hybrid.mid").exists():
            eval_gt(song, wav, gt, d)
    for song, wav, d in real_files:
        if Path(d, "bass_hybrid.mid").exists():
            eval_real(song, wav, d)


if __name__ == "__main__":
    main()
