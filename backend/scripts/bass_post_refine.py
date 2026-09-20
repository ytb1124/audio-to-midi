#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


def midi_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi_note % 12]}{midi_note // 12 - 1}"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def load_audio(audio_path):
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if len(y) == 0:
        raise RuntimeError("Empty audio")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    return y, sr


def build_cqt(y, sr):
    import librosa

    hop = 256
    fmin = librosa.note_to_hz("C1")
    base_midi = 24
    n_bins = 72

    C = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=12,
        )
    )

    C = np.log1p(10.0 * C)
    return C, hop, base_midi


def cqt_energy(C, hop, sr, base_midi, pitch, start, end):
    idx = int(round(pitch - base_midi))
    if idx < 0 or idx >= C.shape[0]:
        return 0.0

    a = max(0, int((start + 0.010) * sr / hop))
    b = min(C.shape[1], int(min(end, start + 0.300) * sr / hop) + 1)

    if b <= a:
        return 0.0

    return float(np.median(C[idx, a:b]))


def bass_harmonic_score(C, hop, sr, base_midi, pitch, start, end, enable_low_octave_bias=False):
    """
    기존 validator는 위쪽 배음도 fundamental처럼 보는 문제가 있었다.
    여기서는 낮은 octave가 비슷하게 그럴듯하면 낮은 쪽을 더 선호한다.
    """
    e0 = cqt_energy(C, hop, sr, base_midi, pitch, start, end)
    e12 = cqt_energy(C, hop, sr, base_midi, pitch + 12, start, end)
    e19 = cqt_energy(C, hop, sr, base_midi, pitch + 19, start, end)
    e24 = cqt_energy(C, hop, sr, base_midi, pitch + 24, start, end)

    # harmonic support
    score = e0 + 0.48 * e12 + 0.25 * e19 + 0.14 * e24

    # 너무 높은 음역은 기본적으로 덜 믿는다.
    # 실제 고음은 가능하지만, 이 파일에서는 E2/E3가 배음 오인으로 많이 생긴다.
    # eval/ground_truth/bass_test 하니스 실측: 이 고정 페널티가 실제로 맞는 높은 음
    # (E3/C3/B3/A3 등)을 한 옥타브 아래로 깎는 회귀를 유발함. 기본 비활성화.
    if enable_low_octave_bias:
        if pitch >= 40:
            score -= 0.08 * (pitch - 39)
        if pitch >= 48:
            score -= 0.20 * (pitch - 47)

    return float(score)


def read_midi_notes(midi_path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []

    for inst_idx, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue

        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        for note_idx, n in enumerate(inst.notes):
            notes.append(
                {
                    "inst_idx": inst_idx,
                    "note_idx": note_idx,
                    "start": float(n.start),
                    "end": float(n.end),
                    "pitch": int(n.pitch),
                    "velocity": int(n.velocity),
                }
            )

    notes.sort(key=lambda x: (x["start"], x["pitch"]))
    return pm, notes


def nearest_candidate_row(candidates_df, t, max_dist=0.030):
    if candidates_df is None or len(candidates_df) == 0:
        return None

    idx = (candidates_df["time"] - t).abs().idxmin()
    row = candidates_df.loc[idx]

    if abs(float(row["time"]) - t) <= max_dist:
        return row

    return None


def parse_nearby_raw(row):
    if row is None:
        return []

    raw = row.get("nearby_raw", "[]")
    if not isinstance(raw, str):
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def candidate_pitches_from_raw(current_pitch, raw_notes, min_midi, max_midi):
    candidates = set()

    # 현재 pitch도 후보에 넣는다.
    for q in [current_pitch - 24, current_pitch - 12, current_pitch, current_pitch + 12]:
        if min_midi <= q <= max_midi:
            candidates.add(q)

    # raw pitch는 그대로 믿지 말고, 아래 octave 후보를 강하게 검토한다.
    for rn in raw_notes:
        try:
            p = int(rn["pitch"])
        except Exception:
            continue

        for q in [p - 36, p - 24, p - 12, p]:
            if min_midi <= q <= max_midi:
                candidates.add(q)

    return sorted(candidates)


def choose_best_pitch(
    current_pitch,
    start,
    end,
    velocity,
    row,
    C,
    hop,
    sr,
    base_midi,
    min_midi,
    max_midi,
    prev_pitch=None,
    enable_low_octave_bias=False,
):
    raw_notes = parse_nearby_raw(row)
    candidates = candidate_pitches_from_raw(current_pitch, raw_notes, min_midi, max_midi)

    if not candidates:
        return current_pitch, [], 0.0, "no_candidates"

    audio_strength = 0.0
    sources = ""
    if row is not None:
        try:
            audio_strength = float(row.get("audio_strength", 0.0) or 0.0)
        except Exception:
            audio_strength = 0.0
        sources = str(row.get("sources", ""))

    scored = []

    for p in candidates:
        s = bass_harmonic_score(C, hop, sr, base_midi, p, start, end, enable_low_octave_bias=enable_low_octave_bias)

        # raw 후보 보너스.
        # raw가 p보다 한두 옥타브 위를 잡은 경우도 p를 후보로 인정한다.
        for rn in raw_notes:
            try:
                rp = int(rn["pitch"])
                rv = int(rn.get("velocity", velocity)) / 127.0
            except Exception:
                continue

            if p == rp:
                s += 0.08 * rv
            elif p == rp - 12:
                s += 0.25 * rv
            elif p == rp - 24:
                s += 0.22 * rv
            elif p == rp - 36:
                s += 0.12 * rv

        # 현재 pitch 유지 보너스는 아주 작게.
        # 기존에는 이게 커서 잘못된 pitch가 고정됐다.
        if p == current_pitch:
            s += 0.035

        # 낮은 octave 우선.
        # 베이스에서는 위쪽 배음이 더 크게 보일 수 있으므로 낮은 쪽을 조금 유리하게 둔다.
        # eval 하니스 실측: 이 보너스도 실제 고음 노트를 깎는 방향으로 작용해 기본 비활성화.
        if enable_low_octave_bias:
            s += max(0, 43 - p) * 0.012

        # 이전 pitch와 너무 큰 점프는 penalty.
        if prev_pitch is not None:
            jump = abs(p - prev_pitch)
            if jump > 12:
                s -= 0.18 * (jump - 12)
            elif jump > 7:
                s -= 0.06 * (jump - 7)

        scored.append((float(s), int(p)))

    scored.sort(reverse=True)

    best_score, best_pitch = scored[0]
    current_score = None
    for s, p in scored:
        if p == current_pitch:
            current_score = s
            break

    if current_score is None:
        current_score = -9999.0

    margin = best_score - current_score

    # 같은 pitch class에서 낮은 octave가 현재와 비슷하면 낮은 octave 선택
    # eval 하니스 실측: 이 override가 이미 올바른 고음 후보를 낮은 옥타브로
    # 바꿔치기하는 사례가 있어 기본 비활성화.
    if enable_low_octave_bias:
        same_class_lower = [
            (s, p) for s, p in scored
            if p < current_pitch and p % 12 == current_pitch % 12
        ]

        if same_class_lower:
            lower_score, lower_pitch = same_class_lower[0]
            if lower_score >= current_score - 0.18:
                best_score, best_pitch = lower_score, lower_pitch
                margin = best_score - current_score

    reason = f"best_margin_{margin:.4f}_sources_{sources}_audio_{audio_strength:.3f}"
    return best_pitch, scored[:8], best_score, reason


def should_drop_note(
    start,
    end,
    old_pitch,
    new_pitch,
    velocity,
    best_score,
    row,
    prev_pitch=None,
    next_pitch=None,
):
    dur = float(end - start)

    audio_strength = 0.0
    sources = ""
    raw_notes = []

    if row is not None:
        sources = str(row.get("sources", ""))
        try:
            audio_strength = float(row.get("audio_strength", 0.0) or 0.0)
        except Exception:
            audio_strength = 0.0
        raw_notes = parse_nearby_raw(row)

    max_raw_velocity = 0
    for rn in raw_notes:
        try:
            max_raw_velocity = max(max_raw_velocity, int(rn.get("velocity", 0)))
        except Exception:
            pass

    # 1) 거의 무음인데 raw note 하나 때문에 생긴 note 제거
    if best_score < 0.35 and audio_strength < 0.15 and max_raw_velocity < 60:
        return True, "drop_low_audio_score_raw_weak"

    # 2) 짧고 약한 raw-only ghost 제거
    if dur < 0.075 and audio_strength < 0.18 and velocity < 58 and best_score < 1.25:
        return True, "drop_short_weak_raw_only"

    # 3) 높은 octave ghost 제거
    if new_pitch >= 48 and audio_strength < 0.22 and best_score < 2.0:
        return True, "drop_high_octave_ghost"

    # 4) 순간적으로 튀었다가 바로 돌아오는 passing ghost 제거
    if prev_pitch is not None and next_pitch is not None:
        if dur < 0.105:
            jump_in = abs(new_pitch - prev_pitch)
            jump_out = abs(next_pitch - new_pitch)
            same_neighbors = abs(prev_pitch - next_pitch) <= 2

            if jump_in >= 7 and jump_out >= 7 and same_neighbors and audio_strength < 0.35:
                return True, "drop_spike_between_same_neighbors"

    return False, ""


def write_notes_to_midi(pm, notes, midi_out):
    # 기존 non-drum instrument 하나만 사용
    target_inst = None
    for inst in pm.instruments:
        if not inst.is_drum:
            target_inst = inst
            break

    if target_inst is None:
        return

    import pretty_midi

    target_inst.notes = []
    for n in notes:
        target_inst.notes.append(
            pretty_midi.Note(
                velocity=int(n["velocity"]),
                pitch=int(n["pitch"]),
                start=float(n["start"]),
                end=float(n["end"]),
            )
        )

    pm.write(str(midi_out))


def save_final_csv(notes, csv_path):
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["start", "end", "duration", "pitch", "note", "velocity"],
        )
        writer.writeheader()

        for n in notes:
            writer.writerow(
                {
                    "start": round(float(n["start"]), 6),
                    "end": round(float(n["end"]), 6),
                    "duration": round(float(n["end"] - n["start"]), 6),
                    "pitch": int(n["pitch"]),
                    "note": midi_to_name(int(n["pitch"])),
                    "velocity": int(n["velocity"]),
                }
            )


def make_overlay(audio_path, midi_path, out_png):
    try:
        import librosa
        import matplotlib.pyplot as plt
        import pretty_midi
    except Exception as exc:
        eprint("Overlay skipped:", exc)
        return

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    C = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=256,
            fmin=librosa.note_to_hz("C1"),
            n_bins=48,
            bins_per_octave=12,
        )
    )
    C = librosa.amplitude_to_db(C, ref=np.max)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=256)

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    plt.figure(figsize=(18, 7))
    plt.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 24, 72],
        cmap="magma",
        vmin=-70,
        vmax=0,
    )

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            plt.plot([n.start, n.end], [n.pitch, n.pitch], linewidth=1.2)

    plt.title("Bass post-refine overlay")
    plt.xlabel("time seconds")
    plt.ylabel("MIDI pitch")
    plt.ylim(24, 60)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=140)
    plt.close()


def post_refine(audio_path, midi_in, midi_out, output_dir, min_midi=28, max_midi=52, enable_low_octave_bias=False):
    import pandas as pd

    audio_path = Path(audio_path)
    midi_in = Path(midi_in)
    midi_out = Path(midi_out)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y, sr = load_audio(audio_path)
    C, hop, base_midi = build_cqt(y, sr)

    pm, notes = read_midi_notes(midi_in)

    candidates_csv = output_dir / "bass_candidates.csv"
    candidates_df = None
    if candidates_csv.exists():
        candidates_df = pd.read_csv(candidates_csv)

    # 1차 pitch 선택
    proposed = []
    audit_rows = []

    prev_pitch = None

    for i, n in enumerate(notes):
        row = nearest_candidate_row(candidates_df, float(n["start"])) if candidates_df is not None else None

        new_pitch, scored, best_score, reason = choose_best_pitch(
            current_pitch=int(n["pitch"]),
            start=float(n["start"]),
            end=float(n["end"]),
            velocity=int(n["velocity"]),
            row=row,
            C=C,
            hop=hop,
            sr=sr,
            base_midi=base_midi,
            min_midi=min_midi,
            max_midi=max_midi,
            prev_pitch=prev_pitch,
            enable_low_octave_bias=enable_low_octave_bias,
        )

        nn = dict(n)
        nn["pitch"] = int(new_pitch)
        nn["_best_score"] = float(best_score)
        nn["_row"] = row
        nn["_scored"] = scored
        nn["_reason"] = reason

        proposed.append(nn)
        prev_pitch = int(new_pitch)

    # 2차 ghost pruning
    kept = []
    drop_rows = []

    for i, n in enumerate(proposed):
        prev_pitch = proposed[i - 1]["pitch"] if i > 0 else None
        next_pitch = proposed[i + 1]["pitch"] if i + 1 < len(proposed) else None

        drop, drop_reason = should_drop_note(
            start=float(n["start"]),
            end=float(n["end"]),
            old_pitch=int(notes[i]["pitch"]),
            new_pitch=int(n["pitch"]),
            velocity=int(n["velocity"]),
            best_score=float(n["_best_score"]),
            row=n["_row"],
            prev_pitch=prev_pitch,
            next_pitch=next_pitch,
        )

        row_data = {
            "index": i,
            "start": round(float(n["start"]), 6),
            "end": round(float(n["end"]), 6),
            "duration": round(float(n["end"] - n["start"]), 6),
            "old_pitch": int(notes[i]["pitch"]),
            "old_note": midi_to_name(int(notes[i]["pitch"])),
            "new_pitch": int(n["pitch"]),
            "new_note": midi_to_name(int(n["pitch"])),
            "velocity": int(n["velocity"]),
            "best_score": round(float(n["_best_score"]), 6),
            "changed": int(int(notes[i]["pitch"]) != int(n["pitch"])),
            "dropped": int(drop),
            "reason": drop_reason or n["_reason"],
            "scores": json.dumps(
                [
                    {
                        "score": round(float(s), 6),
                        "pitch": int(p),
                        "note": midi_to_name(int(p)),
                    }
                    for s, p in n["_scored"]
                ],
                ensure_ascii=False,
            ),
        }

        audit_rows.append(row_data)

        if drop:
            drop_rows.append(row_data)
            continue

        clean_note = {
            "start": float(n["start"]),
            "end": float(n["end"]),
            "pitch": int(n["pitch"]),
            "velocity": int(n["velocity"]),
        }
        kept.append(clean_note)

    # 겹침 정리
    kept.sort(key=lambda x: x["start"])
    for i in range(len(kept) - 1):
        if kept[i]["end"] > kept[i + 1]["start"] - 0.008:
            kept[i]["end"] = max(kept[i]["start"] + 0.045, kept[i + 1]["start"] - 0.008)

    kept = [n for n in kept if n["end"] - n["start"] >= 0.045]

    write_notes_to_midi(pm, kept, midi_out)

    audit_csv = output_dir / "bass_post_refine_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "start",
                "end",
                "duration",
                "old_pitch",
                "old_note",
                "new_pitch",
                "new_note",
                "velocity",
                "best_score",
                "changed",
                "dropped",
                "reason",
                "scores",
            ],
        )
        writer.writeheader()
        for r in audit_rows:
            writer.writerow(r)

    drops_csv = output_dir / "bass_post_refine_drops.csv"
    with drops_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "start",
                "end",
                "duration",
                "old_pitch",
                "old_note",
                "new_pitch",
                "new_note",
                "velocity",
                "best_score",
                "changed",
                "dropped",
                "reason",
                "scores",
            ],
        )
        writer.writeheader()
        for r in drop_rows:
            writer.writerow(r)

    final_csv = output_dir / "bass_final_notes_post_refined.csv"
    save_final_csv(kept, final_csv)

    # 오버레이 PNG는 디버깅용 — matplotlib 렌더링이 긴 곡에서 수 초를 잡아먹어
    # 기본 비활성 (BASS_DEBUG_OVERLAY=1 환경변수로 켬)
    import os as _os

    overlay_png = output_dir / "bass_post_refine_overlay.png"
    if _os.environ.get("BASS_DEBUG_OVERLAY") == "1":
        make_overlay(audio_path, midi_out, overlay_png)

    old_count = len(notes)
    new_count = len(kept)
    changed_count = sum(1 for r in audit_rows if int(r["changed"]) == 1)
    dropped_count = len(drop_rows)

    cnt = Counter(int(n["pitch"]) for n in kept)

    eprint(f"Post refine old notes: {old_count}")
    eprint(f"Post refine new notes: {new_count}")
    eprint(f"Post refine pitch changes: {changed_count}")
    eprint(f"Post refine drops: {dropped_count}")
    eprint("Pitch distribution:")
    for p in sorted(cnt):
        eprint(f"  {p:3d} {midi_to_name(p):4s} {cnt[p]}")

    eprint(f"Saved post refine audit: {audit_csv}")
    eprint(f"Saved post refine drops: {drops_csv}")
    eprint(f"Saved post refined notes: {final_csv}")
    eprint(f"Saved post refine overlay: {overlay_png}")

    return {
        "old_count": old_count,
        "new_count": new_count,
        "changed_count": changed_count,
        "dropped_count": dropped_count,
        "audit_csv": str(audit_csv),
        "drops_csv": str(drops_csv),
        "final_csv": str(final_csv),
        "overlay_png": str(overlay_png),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("midi_in")
    parser.add_argument("midi_out")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-midi", type=int, default=28)
    parser.add_argument("--max-midi", type=int, default=52)
    parser.add_argument("--enable-low-octave-bias", action="store_true")
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    midi_in = Path(args.midi_in).expanduser().resolve()
    midi_out = Path(args.midi_out).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else midi_out.parent

    post_refine(
        audio_path=audio,
        midi_in=midi_in,
        midi_out=midi_out,
        output_dir=output_dir,
        min_midi=args.min_midi,
        max_midi=args.max_midi,
        enable_low_octave_bias=args.enable_low_octave_bias,
    )


if __name__ == "__main__":
    main()
