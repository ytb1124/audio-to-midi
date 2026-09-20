#!/usr/bin/env python3
"""
각 정답 곡에 대해, 우리 파이프라인 생성 결과와 대조해서 전체 파일 단위
옥타브 시프트를 감지하고, 감지되면 보정한 새 정답 MIDI를 저장한다.

목적: bass_sample/2/3/4/5의 "서브옥타브 우세" 문제가 실제 음향 현상이
아니라 정답 MIDI 익스포트 과정의 옥타브 라벨 실수일 가능성이 높다는
사용자 지적에 따라, 학습 데이터 생성 전에 라벨을 보정한다.

사용법: python eval/apply_octave_shift_correction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import load_notes, detect_global_octave_shift

import pretty_midi


FILES = [
    ("bass_sample", "eval/runs/norm_sample/bass_hybrid.mid", "eval/ground_truth/bass_sample.mid"),
    ("bass_sample2", "eval/runs/norm_sample2/bass_hybrid.mid", "eval/ground_truth/bass_sample2.mid"),
    ("bass_sample3", "eval/runs/norm_sample3/bass_hybrid.mid", "eval/ground_truth/bass_sample3.mid"),
    ("bass_sample4", "eval/runs/norm_sample4/bass_hybrid.mid", "eval/ground_truth/bass_sample4.mid"),
    ("bass_sample5", "eval/runs/norm_sample5/bass_hybrid.mid", "eval/ground_truth/bass_sample5.mid"),
]


def write_shifted_midi(ref_notes_original, shift, original_path, out_path):
    """원본 MIDI의 velocity 등 메타는 유지하면서 pitch만 shift만큼 이동해서 새로 쓴다.
    인덱스 순서에 의존하지 않고 각 note 객체를 직접 이동시킨다."""
    pm = pretty_midi.PrettyMIDI(str(original_path))
    inst = None
    for candidate in pm.instruments:
        if not candidate.is_drum:
            inst = candidate
            break

    for note in inst.notes:
        note.pitch = int(note.pitch) + shift

    pm.write(str(out_path))


def main():
    for song_id, gen_path, ref_path in FILES:
        gen_path = Path(gen_path)
        ref_path = Path(ref_path)

        if not gen_path.exists():
            print(f"{song_id}: skip (generated file not found: {gen_path})", file=sys.stderr)
            continue

        ref_notes = load_notes(ref_path)
        est_notes = load_notes(gen_path)

        shift, fraction, n_matched = detect_global_octave_shift(ref_notes, est_notes)

        if shift is None:
            print(f"{song_id}: no global shift detected (fraction={fraction:.2%}, n={n_matched}) — 원본 그대로 사용", file=sys.stderr)
            continue

        # est가 ref보다 shift만큼 낮게(-12) 나왔다면, ref를 -12만큼 보정해서
        # "정답이 실제로는 이 옥타브였다"고 재해석한다.
        correction = shift  # est - ref = shift 이므로, ref_corrected = ref + shift

        out_path = ref_path.with_name(ref_path.stem + "_octcorrected.mid")
        write_shifted_midi(ref_notes, correction, ref_path, out_path)

        print(
            f"{song_id}: shift={correction:+d} semitones ({correction//12:+d} octave), "
            f"fraction={fraction:.2%} of {n_matched} matched -> saved {out_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
