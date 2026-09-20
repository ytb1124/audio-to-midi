#!/usr/bin/env python3
"""
Gated shadow 통합: 기존 파이프라인 결과를 MTL octave/offset 헤드로 '재검토'.
(bass-note-verifier 브랜치, 2026-07-13 — 사용자 명세 반영)

핵심 원칙:
- 기존 스테이지를 대체하지 않고 그 뒤에 삽입해 second-opinion으로만 동작.
- worst-case 회귀 방지 우선. confidence 게이트 + 위험 케이스 보류.
- octave: MTL이 기존과 다른 옥타브를 높은 confidence로 제안하고 위험하지
  않을 때만 변경. ±24/음역밖/강한겹침/초단음/pYIN lock 충돌은 보류.
- offset: 절대 덮어쓰기 금지. 기존 offset에 대한 delta로만 적용,
  clamp(-150ms,+250ms), 다음 온셋-10ms 상한, 최소 30ms.
- 모든 변경을 audit CSV로 기록 (적용/보류 사유 포함).

피처는 학습 스크립트(extract_octave/proposal_features.py)와 동일하게 계산.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bass_audio_cache import cqt_c1_72, load_audio_22050, onset_env_hpss, pyin_track  # noqa: E402
from bass_mtl_infer import load_model, softmax  # noqa: E402

NEIGHBOR_WINDOW_SEC = 2.5
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 모델/게이트 버전 (결과 추적용 — 사용자 명세). 결과 MIDI가 어느 모델과
# 게이트로 만들어졌는지 audit에 남긴다.
MTL_VERSION = {
    "model": "bass_note_verifier",
    "checkpoint_version": "offset_v1",
    "gate_version": "offset_gate_v1",
    "git_commit": "d165b9a7",
}


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def write_mtl_status(output_dir, **kw):
    """job audit: MTL 적용/폴백 상태 + 모델 버전 기록.
    운영 환경에서 결과 추적/디버깅 및 fallback 가시성 확보."""
    import json
    rec = dict(MTL_VERSION)
    rec.update(kw)
    try:
        (Path(output_dir) / "bass_mtl_status.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
    except Exception as exc:
        eprint(f"Warning: could not write mtl status: {exc}")


def nm(p):
    return f"{NAMES[p % 12]}{p // 12 - 1}"


# ---- 공통 피처 헬퍼 (extract 스크립트와 동일 계산) ----

def cqt_energy(C, hop, sr, base, pitch, start, end):
    idx = int(round(pitch - base))
    if idx < 0 or idx >= C.shape[0]:
        return 0.0
    a = max(0, int((start + 0.015) * sr / hop))
    b = min(C.shape[1], int(min(end, start + 0.30) * sr / hop) + 1)
    return float(np.median(C[idx, a:b])) if b > a else 0.0


def cqt_harmonic_win(C, hop, sr, base, pitch, t0, t1):
    def e(p):
        idx = int(round(p - base))
        if idx < 0 or idx >= C.shape[0]:
            return 0.0
        a = max(0, int(t0 * sr / hop)); b = min(C.shape[1], int(t1 * sr / hop) + 1)
        return float(np.median(C[idx, a:b])) if b > a else 0.0
    return e(pitch) + 0.42 * e(pitch + 12) + 0.22 * e(pitch + 19) + 0.12 * e(pitch + 24)


def read_notes(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    inst = next((i for i in pm.instruments if not i.is_drum), None)
    if inst is None:
        return pm, None, []
    inst.notes.sort(key=lambda n: (n.start, n.pitch))
    return pm, inst, inst.notes


# ================= OCTAVE REVIEW =================

def mtl_octave_review(audio_path, midi_in, midi_out, output_dir,
                      min_midi=28, max_midi=60, conf_threshold=0.9, audit_only=False):
    """audit_only=True: 제안/audit만 기록, 실제 pitch는 변경 안 함(shadow).
    사용자 명세: octave head는 현 파이프라인과 중복이라 shadow/audit only 유지."""
    output_dir = Path(output_dir)
    model = load_model()
    y, sr = load_audio_22050(audio_path, output_dir)
    C, chop, base = cqt_c1_72(y, sr, output_dir)
    pyin_midi, pyin_prob, phop = pyin_track(y, sr, "E1", "C5", output_dir, "e1c5")

    # 파일 투표(trust_pyin) — distortion/subharmonic 위험 신호로 사용
    trust_pyin = False
    vote_path = output_dir / "bass_octave_file_vote.json"
    if vote_path.exists():
        try:
            import json
            trust_pyin = json.loads(vote_path.read_text()).get("trust_pyin") is True
        except Exception:
            pass

    pm, inst, notes = read_notes(midi_in)
    if inst is None or not notes:
        eprint("MTL octave: no notes")
        return
    note_list = [(float(n.start), float(n.end), int(n.pitch)) for n in notes]

    def pyin_stats(start, end):
        a = max(0, int(start * sr / phop)); b = min(len(pyin_midi), int(min(end, start + 0.35) * sr / phop) + 1)
        if b <= a:
            return None, 0.0
        seg = pyin_midi[a:b]; prob = pyin_prob[a:b]; v = np.isfinite(seg)
        if not v.any():
            return None, float(np.median(prob)) if len(prob) else 0.0
        return float(np.nanmedian(seg[v])), float(np.nanmedian(prob[v]))

    def raw_harm(pitch, s, e):
        return (cqt_energy(C, chop, sr, base, pitch, s, e)
                + 0.42 * cqt_energy(C, chop, sr, base, pitch + 12, s, e)
                + 0.22 * cqt_energy(C, chop, sr, base, pitch + 19, s, e)
                + 0.12 * cqt_energy(C, chop, sr, base, pitch + 24, s, e))

    audit = []
    changed = 0
    for i, note in enumerate(notes):
        start, end = float(note.start), float(note.end)
        old = int(note.pitch); pc = old % 12
        cands = [p for p in range(min_midi, max_midi + 1) if p % 12 == pc]
        if len(cands) < 2:
            continue
        py_med, py_conf = pyin_stats(start, end)
        neigh = [(ns, ne, np_) for j, (ns, ne, np_) in enumerate(note_list)
                 if j != i and np_ % 12 == pc and abs(ns - start) <= NEIGHBOR_WINDOW_SEC]

        feats = []
        for cand in cands:
            e0 = cqt_energy(C, chop, sr, base, cand, start, end)
            e12 = cqt_energy(C, chop, sr, base, cand + 12, start, end)
            em12 = cqt_energy(C, chop, sr, base, cand - 12, start, end)
            e19 = cqt_energy(C, chop, sr, base, cand + 19, start, end)
            e24 = cqt_energy(C, chop, sr, base, cand + 24, start, end)
            harm = e0 + 0.42 * e12 + 0.22 * e19 + 0.12 * e24
            nsup = (sum(raw_harm(cand, ns, ne) for ns, ne, _ in neigh) / len(neigh)) if neigh else 0.0
            feats.append({
                "cqt_e0": e0, "cqt_e12": e12, "cqt_e_minus12": em12, "cqt_e19": e19, "cqt_e24": e24,
                "cqt_harmonic": harm, "pyin_dist": abs(py_med - cand) if py_med is not None else 99.0,
                "pyin_conf": py_conf, "note_duration": end - start,
                "neighbor_support_mean": nsup, "neighbor_count": len(neigh),
            })
        logits = model.octave_scores(feats)
        probs = softmax(logits)
        best = int(np.argmax(logits))
        mtl_pitch = cands[best]
        mtl_conf = float(probs[best])

        applied = False
        reason = "keep"
        if mtl_pitch != old:
            # 위험 케이스 게이트
            if abs(mtl_pitch - old) >= 24:
                reason = "hold_pm24"
            elif not (min_midi <= mtl_pitch <= max_midi):
                reason = "hold_range"
            elif (end - start) < 0.045:
                reason = "hold_short"
            elif mtl_conf < conf_threshold:
                reason = "hold_lowconf"
            elif trust_pyin and py_med is not None and abs(py_med - mtl_pitch) > abs(py_med - old):
                # 실녹음(trust_pyin) 파일에서 pYIN lock과 더 멀어지는 변경 보류
                reason = "hold_pyin_conflict"
            else:
                # 직전 노트와 강한 시간 겹침이 있는지 (모노포닉 침범)
                overlap = False
                if i > 0:
                    ps, pe, _ = note_list[i - 1]
                    overlap = ps < end - 0.02 and pe > start + 0.02
                if overlap:
                    reason = "hold_overlap"
                elif audit_only:
                    reason = "shadow_audit_only"
                else:
                    note.pitch = mtl_pitch
                    applied = True
                    changed += 1
                    reason = "apply"
        audit.append({
            "start": round(start, 4), "existing_pitch": old, "existing_note": nm(old),
            "mtl_pitch": mtl_pitch, "mtl_note": nm(mtl_pitch),
            "mtl_confidence": round(mtl_conf, 4), "applied": int(applied), "reason": reason,
        })

    pm.write(str(midi_out))
    with (output_dir / "bass_mtl_octave_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start", "existing_pitch", "existing_note", "mtl_pitch",
                                          "mtl_note", "mtl_confidence", "applied", "reason"])
        w.writeheader()
        for r in audit:
            w.writerow(r)
    n_prop = sum(1 for r in audit if r["mtl_pitch"] != r["existing_pitch"])
    eprint(f"MTL octave review: {changed} applied / {n_prop} proposed / {len(audit)} notes")


# ================= OFFSET REVIEW =================

def _win(arr, hop, sr, t0, t1, fn, default=0.0):
    a = max(0, int(t0 * sr / hop)); b = min(len(arr), int(t1 * sr / hop) + 1)
    return float(fn(arr[a:b])) if b > a else default


def _decay_time(row, hop, sr, t0, t_cap, ratio=0.25):
    if row is None:
        return 0.0
    a = int(t0 * sr / hop); b = min(len(row), int(t_cap * sr / hop) + 1)
    if b <= a + 2:
        return 0.0
    seg = row[a:b]; peak = float(seg[: max(3, int(0.08 * sr / hop))].max())
    if peak < 0.3:
        return 0.0
    below = np.nonzero(seg < ratio * peak)[0]
    return float(below[0]) * hop / sr if len(below) else (b - a) * hop / sr


def mtl_offset_review(audio_path, midi_in, midi_out, output_dir,
                      clamp_lo=-0.15, clamp_hi=0.25, next_gap=0.01, min_dur=0.06,
                      min_apply_delta=0.06, only_shorten_oversustain=True):
    """게이트 (2026-07-13 factorial 후 추가): 이미 정확한 노트를 건드려
    앵커(bass_test 0.97->0.67)를 무너뜨리는 문제 방지.
    - min_apply_delta: |delta| 이만큼 미만이면 미적용 (작은 조정은 노이즈)
    - min_dur: 결과 노트 최소 길이(60ms) — 초단음 양산 방지
    - only_shorten_oversustain: 노트 끝이 자기 음향 감쇠점(cqt_decay)보다
      명백히 길 때(과지속)만 단축 허용. 감쇠 안에 이미 들어온 노트는 보존."""
    import librosa
    output_dir = Path(output_dir)

    # 모델 로드 실패 fallback (사용자 명세): checkpoint/dependency 문제가 있어도
    # 전체 변환이 실패하면 안 됨. baseline offset을 그대로 두고 status만 남긴다.
    try:
        model = load_model()
    except Exception as exc:
        eprint(f"MTL offset: model load failed ({exc}) — baseline offsets 유지")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_mtl_status(output_dir, mtl_enabled=True, mtl_applied=False,
                         fallback="baseline", reason="checkpoint_load_failed",
                         detail=str(exc)[:200])
        # 파이프라인은 in-place(midi_in==midi_out)라 baseline 그대로. standalone
        # 호출로 경로가 다르면 baseline을 복사해 출력 보장.
        if str(Path(midi_in)) != str(Path(midi_out)):
            import shutil as _sh
            _sh.copy2(midi_in, midi_out)
        return

    y, sr = load_audio_22050(audio_path, output_dir)
    C, chop, base = cqt_c1_72(y, sr, output_dir)
    onset_env, ehop = onset_env_hpss(y, sr, output_dir)
    pyin_c1, pyin_c1_prob, phop = pyin_track(y, sr, "C1", "C5", output_dir, "c1c5")
    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=1024)[0]
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf = S[freqs >= 800, :]
    flux = np.concatenate([[0.0], np.sqrt(np.sum(np.maximum(np.diff(hf, axis=1), 0) ** 2, axis=0))])

    env_lo, env_hi = np.percentile(onset_env, 10), np.percentile(onset_env, 95)
    env_sc = max(env_hi - env_lo, 1e-9)
    flux_hi = max(np.percentile(flux, 95), 1e-9)

    pm, inst, notes = read_notes(midi_in)
    if inst is None or not notes:
        eprint("MTL offset: no notes")
        return
    starts = np.array([float(n.start) for n in notes])

    def pyin_win(t0, t1, pitch):
        a = max(0, int(t0 * sr / phop)); b = min(len(pyin_c1), int(t1 * sr / phop) + 1)
        if b <= a:
            return 0.0, 99.0, 0.0
        seg = pyin_c1[a:b]; v = np.isfinite(seg)
        vf = float(v.sum()) / len(seg)
        if not v.any():
            return vf, 99.0, 0.0
        vv = seg[v]; med = float(np.nanmedian(vv))
        return vf, abs(med - pitch), (float(np.std(vv)) if len(vv) >= 2 else 0.0)

    # 1패스: 모든 노트의 피처 수집 (batch inference 준비 — 노트별 개별 모델
    # 호출 금지, 사용자 명세). cqt_decay는 게이트에서 재사용하려 따로 보관.
    feats = []
    ctx = []  # (t, pitch, old_end, next_start, cqt_decay)
    for i, note in enumerate(notes):
        t = float(note.start); pitch = int(note.pitch); old_end = float(note.end)
        next_start = float(notes[i + 1].start) if i + 1 < len(notes) else t + 1.5
        t_cap = min(t + 1.5, next_start)

        att_peak = (_win(onset_env, ehop, sr, t - 0.02, t + 0.04, np.max) - env_lo) / env_sc
        pre_min = _win(rms, hop, sr, t - 0.05, t - 0.005, np.min, 1e-6)
        post_max = _win(rms, hop, sr, t + 0.005, t + 0.05, np.max)
        rms_ratio = post_max / pre_min if pre_min > 1e-6 else (10.0 if post_max > 1e-6 else 1.0)
        flux_peak = _win(flux, hop, sr, t - 0.01, t + 0.04, np.max) / flux_hi
        vf, pdist, _ = pyin_win(t + 0.02, t + 0.17, pitch)
        harm = cqt_harmonic_win(C, chop, sr, base, pitch, t + 0.01, t + 0.15)
        sv, sdist, pstd = pyin_win(t + 0.08, t + 0.25, pitch)
        harm_a = cqt_harmonic_win(C, chop, sr, base, pitch, t + 0.02, t + 0.08)
        harm_m = cqt_harmonic_win(C, chop, sr, base, pitch, t + 0.15, t + 0.28)
        harm_cont = harm_m / harm_a if harm_a > 1e-6 else 0.0
        row_pitch = C[int(round(pitch - base))] if 0 <= int(round(pitch - base)) < C.shape[0] else None
        cqt_decay = _decay_time(row_pitch, chop, sr, t, t_cap)
        rms_decay = _decay_time(rms, hop, sr, t, t_cap)
        gap_prev = t - starts[i - 1] if i > 0 else 9.9
        gap_next = next_start - t

        feats.append({
            "att_peak": att_peak, "rms_ratio": rms_ratio, "flux_peak": flux_peak,
            "pyin_voiced": vf, "pyin_dist": pdist, "cqt_harmonic": harm,
            "gap_prev": min(gap_prev, 9.9), "gap_next": min(gap_next, 9.9),
            "audio_strength": att_peak, "src_audio": 1, "src_raw": 1,
            "dur": old_end - t,
            "sustain_voiced": sv, "sustain_dist": sdist, "pitch_std": pstd,
            "harm_continuity": harm_cont, "cqt_decay": cqt_decay, "rms_decay": rms_decay,
        })
        ctx.append((t, pitch, old_end, next_start, cqt_decay))

    # 배치 추론 1회
    pred_durs = model.offset_durations(feats) if feats else np.array([])

    # 2패스: 게이트 + 적용
    audit = []
    changed = 0
    for note, (t, pitch, old_end, next_start, cqt_decay), pred_dur in zip(notes, ctx, pred_durs):
        pred_dur = float(pred_dur)
        delta = (t + pred_dur) - old_end
        delta = max(clamp_lo, min(clamp_hi, delta))
        new_end = old_end + delta
        new_end = min(new_end, next_start - next_gap)
        new_end = max(new_end, t + min_dur)

        gate_reason = ""
        if abs(new_end - old_end) < min_apply_delta:
            gate_reason = "skip_small"
        elif only_shorten_oversustain and new_end < old_end:
            decay_end = t + cqt_decay if cqt_decay > 0 else old_end
            if old_end <= decay_end + 0.06:
                gate_reason = "skip_within_decay"

        applied = (gate_reason == "") and abs(new_end - old_end) > 0.005
        if applied:
            note.end = new_end
            changed += 1
        else:
            new_end = old_end
        audit.append({
            "start": round(t, 4), "pitch": pitch, "note": nm(pitch),
            "old_end": round(old_end, 4), "pred_dur": round(pred_dur, 4),
            "new_end": round(new_end, 4), "delta_ms": round((new_end - old_end) * 1000),
            "applied": int(applied),
        })

    pm.write(str(midi_out))
    with (output_dir / "bass_mtl_offset_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start", "pitch", "note", "old_end", "pred_dur",
                                          "new_end", "delta_ms", "applied"])
        w.writeheader()
        for r in audit:
            w.writerow(r)
    deltas = [r["delta_ms"] for r in audit if r["applied"]]
    med = int(np.median(deltas)) if deltas else 0
    eprint(f"MTL offset review: {changed} adjusted / {len(audit)} notes (median |delta| {med}ms)")

    # validity confidence metadata (UI용, 자동 삭제 아님 — 사용자 명세):
    # 최종 노트별 P(real)을 계산해 저신뢰 노트를 UI에서 표시할 수 있게 저장.
    n_lowconf = _write_validity_metadata(model, notes, y, sr, C, chop, base,
                                         onset_env, ehop, rms, hop, flux, flux_hi,
                                         env_lo, env_sc, pyin_c1, phop, output_dir)

    write_mtl_status(output_dir, mtl_enabled=True, mtl_applied=True,
                     fallback=None, reason=None,
                     offset_adjusted=changed, notes=len(audit),
                     validity_lowconf=n_lowconf)


def _write_validity_metadata(model, notes, y, sr, C, chop, base, onset_env, ehop,
                             rms, hop, flux, flux_hi, env_lo, env_sc, pyin_c1, phop,
                             output_dir):
    """최종 노트별 validity(P(real)) confidence를 계산해 CSV로 저장.
    낮을수록 '진짜 노트인지 불확실' — UI에서 검토 표시용(자동 삭제 안 함)."""
    def pyin_win(t0, t1, pitch):
        a = max(0, int(t0 * sr / phop)); b = min(len(pyin_c1), int(t1 * sr / phop) + 1)
        if b <= a:
            return 0.0, 99.0, 0.0
        seg = pyin_c1[a:b]; v = np.isfinite(seg)
        vf = float(v.sum()) / len(seg)
        if not v.any():
            return vf, 99.0, 0.0
        vv = seg[v]; med = float(np.nanmedian(vv))
        return vf, abs(med - pitch), (float(np.std(vv)) if len(vv) >= 2 else 0.0)

    starts = np.array([float(n.start) for n in notes])
    feats = []
    for i, note in enumerate(notes):
        t = float(note.start); pitch = int(note.pitch); end = float(note.end)
        next_start = float(notes[i + 1].start) if i + 1 < len(notes) else t + 1.5
        att = (_win(onset_env, ehop, sr, t - 0.02, t + 0.04, np.max) - env_lo) / env_sc
        pre = _win(rms, hop, sr, t - 0.05, t - 0.005, np.min, 1e-6)
        post = _win(rms, hop, sr, t + 0.005, t + 0.05, np.max)
        rr = post / pre if pre > 1e-6 else (10.0 if post > 1e-6 else 1.0)
        fp = _win(flux, hop, sr, t - 0.01, t + 0.04, np.max) / flux_hi
        vf, pd, _ = pyin_win(t + 0.02, t + 0.17, pitch)
        harm = cqt_harmonic_win(C, chop, sr, base, pitch, t + 0.01, t + 0.15)
        sv, sd, ps = pyin_win(t + 0.08, t + 0.25, pitch)
        ha = cqt_harmonic_win(C, chop, sr, base, pitch, t + 0.02, t + 0.08)
        hm = cqt_harmonic_win(C, chop, sr, base, pitch, t + 0.15, t + 0.28)
        hc = hm / ha if ha > 1e-6 else 0.0
        feats.append({
            "att_peak": att, "rms_ratio": rr, "flux_peak": fp, "pyin_voiced": vf,
            "pyin_dist": pd, "cqt_harmonic": harm, "gap_prev": min(t - starts[i - 1] if i > 0 else 9.9, 9.9),
            "gap_next": min(next_start - t, 9.9), "audio_strength": att, "src_audio": 1, "src_raw": 1,
            "dur": end - t, "sustain_voiced": sv, "sustain_dist": sd, "pitch_std": ps,
            "harm_continuity": hc, "cqt_decay": 0.0, "rms_decay": 0.0,
        })
    if not feats:
        return 0
    probs = model.validity_probs(feats)
    n_low = 0
    with (output_dir / "bass_validity_metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start", "pitch", "note", "validity_confidence", "flag"])
        w.writeheader()
        for note, p in zip(notes, probs):
            # UI 표시용 3단계 (자동 삭제 아님): 낮을수록 검토 필요
            flag = "ok" if p >= 0.8 else ("review" if p >= 0.5 else "uncertain")
            if p < 0.8:
                n_low += 1
            w.writerow({"start": round(float(note.start), 4), "pitch": int(note.pitch),
                        "note": nm(int(note.pitch)), "validity_confidence": round(float(p), 4),
                        "flag": flag})
    eprint(f"Validity metadata: {n_low}/{len(notes)} notes flagged for review (<0.8)")
    return n_low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["octave", "offset"])
    ap.add_argument("audio"); ap.add_argument("midi_in"); ap.add_argument("midi_out")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--min-midi", type=int, default=28)
    ap.add_argument("--max-midi", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.output_dir) if args.output_dir else Path(args.midi_out).parent
    if args.mode == "octave":
        mtl_octave_review(args.audio, args.midi_in, args.midi_out, out, args.min_midi, args.max_midi)
    else:
        mtl_offset_review(args.audio, args.midi_in, args.midi_out, out)


if __name__ == "__main__":
    main()
