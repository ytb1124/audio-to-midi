#!/usr/bin/env python3
"""
노트별 피치 안정성 분석 + sustain 기준 shadow 보정 + confidence metadata.
(사용자 명세 2~5단계, 2026-07-13 — 전부 shadow: 실제 MIDI 미적용, audit만)

각 노트마다 계산:
  onset_pitch          기존 배정 pitch
  sustain_pitch        sustain 구간 pYIN dominant 반음
  voiced_ratio         sustain voiced 프레임 비율
  pitch_std            sustain pYIN 표준편차
  dominant_occupancy   sustain voiced 중 dominant 반음이 차지하는 비율
  pitch_slope          노트 내 pYIN 선형기울기(semitone/sec) — slide/bend 감지
  harm_continuity      어택 대비 중간부 배음 지속비

분류(task 2): agree / attack_unstable / slide_bend / transition /
              muted_ghost / segmentation / misrecognition

shadow 보정 결정(task 3): sustain pitch가 기존과 ±1~2반음 다르고 안정적이며
  slide가 아니고 sustain confidence가 더 높을 때만 'apply'(후보). 미적용.
  hold_slide / hold_unvoiced / hold_low_confidence / hold_large_change

confidence metadata(task 5): validity/pitch/octave/voiced/articulation/reason.

사용법: python eval/diagnose_pitch_stability.py <audio.wav> <generated.mid>
        [--shadow-csv ...] [--meta-csv ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 임계값 (사용자 명세의 조건 — 초기값, audit으로 관찰 후 조정)
SLIDE_SLOPE = 8.0        # |semitone/sec| 이상이면 slide/bend
MIN_VOICED = 0.5         # sustain voiced ratio 하한
MAX_PITCH_STD = 0.5      # sustain pitch 안정성
MIN_OCCUPANCY = 0.6      # dominant 반음 점유율
MIN_HARM_CONT = 0.5      # 배음 지속
MIN_NOTE_DUR = 0.12      # 노트 길이 하한(보정 대상)
UNVOICED_RATIO = 0.3


def nm(p):
    p = int(round(p))
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def load_audio(p):
    import librosa
    y, sr = librosa.load(p, sr=22050, mono=True)
    return y / (np.max(np.abs(y)) + 1e-9), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("midi")
    ap.add_argument("--shadow-csv", default=None)
    ap.add_argument("--meta-csv", default=None)
    args = ap.parse_args()

    import librosa
    import pretty_midi

    y, sr = load_audio(args.audio)
    hop = 256
    print("pYIN...", file=sys.stderr)
    f0, _, vprob = librosa.pyin(y, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C5"),
                               sr=sr, frame_length=2048, hop_length=hop, fill_na=np.nan)
    midi = np.full(len(f0), np.nan)
    v = np.isfinite(f0)
    midi[v] = 69.0 + 12.0 * np.log2(f0[v] / 440.0)
    vprob = np.nan_to_num(vprob, nan=0.0)

    print("CQT...", file=sys.stderr)
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
                           n_bins=72, bins_per_octave=12))
    C = np.log1p(10.0 * C)
    base = 24
    y_h, y_p = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=0.65 * y_h + 0.35 * y_p, sr=sr, hop_length=hop,
                                       aggregate=np.median, fmax=2200)
    env_lo, env_hi = np.percentile(env, 10), np.percentile(env, 95)
    env_sc = max(env_hi - env_lo, 1e-9)

    def frames(t0, t1):
        a = max(0, int(t0 * sr / hop)); b = min(len(midi), int(t1 * sr / hop) + 1)
        return a, b

    def cqt_e(pitch, t0, t1):
        idx = int(round(pitch - base))
        if idx < 0 or idx >= C.shape[0]:
            return 0.0
        a, b = frames(t0, t1)
        return float(np.median(C[idx, a:b])) if b > a else 0.0

    def cqt_harm(pitch, t0, t1):
        return (cqt_e(pitch, t0, t1) + 0.42 * cqt_e(pitch + 12, t0, t1)
                + 0.22 * cqt_e(pitch + 19, t0, t1) + 0.12 * cqt_e(pitch + 24, t0, t1))

    pm = pretty_midi.PrettyMIDI(args.midi)
    notes = sorted([n for i in pm.instruments if not i.is_drum for n in i.notes], key=lambda n: n.start)
    print(f"notes: {len(notes)}", file=sys.stderr)

    shadow_rows = []
    meta_rows = []
    from collections import Counter
    cls_count = Counter()
    dec_count = Counter()

    for k, note in enumerate(notes):
        t, end, pitch = float(note.start), float(note.end), int(note.pitch)
        dur = end - t

        # onset 구간 (어택) vs sustain 구간
        oa, ob = frames(t, t + 0.06)
        sa, sb = frames(t + 0.08, min(end, t + 0.40))
        onset_seg = midi[oa:ob]; onset_v = np.isfinite(onset_seg)
        sus_seg = midi[sa:sb]; sus_v = np.isfinite(sus_seg)

        onset_pitch_read = float(np.nanmedian(onset_seg[onset_v])) if onset_v.any() else None
        voiced_ratio = float(sus_v.sum()) / max(len(sus_seg), 1)
        if sus_v.any():
            sv = sus_seg[sus_v]
            sustain_pitch = float(np.nanmedian(sv))
            pitch_std = float(np.std(sv)) if len(sv) >= 2 else 0.0
            sus_rounded = np.round(sv).astype(int)
            dom = Counter(sus_rounded).most_common(1)[0][0]
            occupancy = float((sus_rounded == dom).sum()) / len(sus_rounded)
            # pitch slope: 시간 대비 선형 회귀 (semitone/sec)
            if len(sv) >= 4:
                ts = np.arange(len(sv)) * hop / sr
                slope = float(np.polyfit(ts, sv, 1)[0])
                # 방향성 있는 이동인지: 앞1/3 vs 뒤1/3 median 차 (노이즈면 ~0)
                third = max(1, len(sv) // 3)
                span = abs(float(np.median(sv[-third:])) - float(np.median(sv[:third])))
            else:
                slope = 0.0; span = 0.0
        else:
            sustain_pitch = None; pitch_std = 99.0; occupancy = 0.0; slope = 0.0; dom = None; span = 0.0

        harm_a = cqt_harm(pitch, t + 0.02, t + 0.08)
        harm_m = cqt_harm(pitch, t + 0.15, t + 0.28)
        harm_cont = harm_m / harm_a if harm_a > 1e-6 else 0.0
        att_peak = (float(np.max(env[max(1, oa - 1):ob])) - env_lo) / env_sc if ob > oa else 0.0

        sus_dist = abs(sustain_pitch - pitch) if sustain_pitch is not None else 99.0

        # 진짜 슬라이드: 높은 slope + 방향성 있는 이동(span>1.5). span 작으면
        # 저음 pYIN 노이즈(occupancy 낮고 std 큼)지 슬라이드 아님.
        is_slide = abs(slope) >= SLIDE_SLOPE and span >= 1.5

        # --- 분류 (task 2) ---
        if voiced_ratio < UNVOICED_RATIO:
            cls = "muted_ghost"
        elif is_slide:
            cls = "slide_bend"
        elif pitch_std > 1.0 and occupancy < 0.5:
            cls = "pyin_noise_lowconf"  # pYIN 자체가 불안정 (주로 저음)
        elif sus_dist <= 0.6:
            cls = "agree"
        elif onset_pitch_read is not None and abs(onset_pitch_read - pitch) <= 0.6 and sus_dist > 0.6:
            # 어택은 배정 pitch와 맞는데 sustain이 다름 → 전환/세그멘테이션
            cls = "transition" if abs(slope) >= 3.0 else "attack_stable_sustain_moved"
        elif onset_pitch_read is not None and abs(onset_pitch_read - sustain_pitch) > 1.5:
            cls = "attack_unstable"  # 어택이 sustain과 크게 다름
        elif 0.6 < sus_dist <= 2.5:
            cls = "misrecognition"   # sustain이 안정적으로 다른 반음
        else:
            cls = "segmentation_or_other"
        cls_count[cls] += 1

        # --- shadow 보정 결정 (task 3) ---
        # 핵심 수정: '기존 신뢰'는 어택 세기가 아니라, sustain 구간에서 배정
        # pitch가 오디오에 얼마나 지지받는가(CQT 하모닉). dom과 CQT로 교차검증해
        # pYIN 단독(저음 서브옥타브/오프셋)으로 잘못 정정하는 걸 막는다.
        decision = "keep"
        cqt_assigned = cqt_harm(pitch, t + 0.08, min(end, t + 0.40)) if dom is not None else 0.0
        cqt_dom = cqt_harm(dom, t + 0.08, min(end, t + 0.40)) if dom is not None else 0.0
        if sustain_pitch is not None and dom is not None and dom != pitch:
            diff = abs(dom - pitch)
            if voiced_ratio < MIN_VOICED:
                decision = "hold_unvoiced"
            elif diff > 2:
                decision = "hold_large_change"
            elif is_slide:
                decision = "hold_slide"
            elif not (1 <= diff <= 2):
                decision = "keep"
            elif (pitch_std <= MAX_PITCH_STD and occupancy >= MIN_OCCUPANCY
                  and harm_cont >= MIN_HARM_CONT and dur >= MIN_NOTE_DUR
                  and cqt_dom > cqt_assigned * 1.1):   # CQT도 dom을 더 지지
                decision = "apply"
            else:
                decision = "hold_low_confidence"
        dec_count[decision] += 1

        # --- confidence metadata (task 5) ---
        pitch_confidence = round(voiced_ratio * occupancy * (1.0 - min(sus_dist, 2.0) / 2.0), 3)
        e0 = cqt_e(pitch, t + 0.02, min(end, t + 0.3))
        e_up = cqt_e(pitch + 12, t + 0.02, min(end, t + 0.3))
        e_dn = cqt_e(pitch - 12, t + 0.02, min(end, t + 0.3))
        octave_confidence = round(e0 / (e0 + max(e_up, e_dn) + 1e-6), 3)
        if voiced_ratio < UNVOICED_RATIO:
            artic = "mute_ghost"
        elif is_slide:
            artic = "slide_up" if slope > 0 else "slide_down"
        elif att_peak >= 0.6:
            artic = "pluck"
        else:
            artic = "legato"
        review_reason = ""
        if decision == "apply":
            review_reason = f"pitch->{nm(dom)}?"
        elif cls == "muted_ghost":
            review_reason = "unvoiced/ghost"
        elif cls == "slide_bend":
            review_reason = "slide/bend"
        elif sus_dist > 0.6:
            review_reason = "pitch_uncertain"

        shadow_rows.append({
            "start": round(t, 4), "dur_ms": round(dur * 1000), "onset_pitch": pitch,
            "onset_note": nm(pitch),
            "onset_read": round(onset_pitch_read, 2) if onset_pitch_read is not None else "",
            "sustain_pitch": round(sustain_pitch, 2) if sustain_pitch is not None else "",
            "sustain_dom": nm(dom) if dom is not None else "",
            "voiced_ratio": round(voiced_ratio, 2), "pitch_std": round(pitch_std, 2),
            "occupancy": round(occupancy, 2), "pitch_slope": round(slope, 1),
            "harm_continuity": round(harm_cont, 2),
            "class": cls, "shadow_decision": decision,
        })
        meta_rows.append({
            "start": round(t, 4), "pitch": pitch, "note": nm(pitch),
            "pitch_confidence": pitch_confidence, "octave_confidence": octave_confidence,
            "voiced_ratio": round(voiced_ratio, 2), "articulation_hint": artic,
            "review_reason": review_reason,
        })

    print("\n=== task2 분류 분포 ===")
    for c, n in cls_count.most_common():
        print(f"  {c:28s} {n}")
    print("\n=== task3 shadow 보정 결정 분포 (미적용) ===")
    for d, n in dec_count.most_common():
        print(f"  {d:22s} {n}")
    apply_rows = [r for r in shadow_rows if r["shadow_decision"] == "apply"]
    print(f"\n=== 'apply' 후보 (sustain+CQT 교차검증 통과, 미적용) {len(apply_rows)}건 ===")
    for r in apply_rows[:30]:
        print(f"  {r['start']:7.2f}s {r['onset_note']}->{r['sustain_dom']} "
              f"voiced={r['voiced_ratio']} occ={r['occupancy']} slope={r['pitch_slope']} std={r['pitch_std']} "
              f"harm={r['harm_continuity']}")

    if args.shadow_csv:
        with open(args.shadow_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(shadow_rows[0].keys())); w.writeheader(); w.writerows(shadow_rows)
        print(f"\nSaved shadow audit: {args.shadow_csv}", file=sys.stderr)
    if args.meta_csv:
        with open(args.meta_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys())); w.writeheader(); w.writerows(meta_rows)
        print(f"Saved confidence metadata: {args.meta_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
