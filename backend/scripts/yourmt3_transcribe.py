"""YourMT3+ (MLSP2024) 전사 래퍼 — 격리 venv(.venv-yourmt3)에서 실행된다.

모델/코드 출처: 공식 HuggingFace Space `mimbres/YourMT3` (저자 Sungkyun Chang).
체크포인트: YPTF.MoE+Multi (noPS) — Space 데모의 기본 모델.

메인 venv의 torch/numpy 스택과 의존성이 충돌하지 않도록 별도 인터프리터로
subprocess 호출한다(transformers==4.45.1, numpy==1.26.4 고정).

사용법:
    python yourmt3_transcribe.py input.wav output.mid --instrument bass|all
마지막 줄에 JSON 한 줄({"notes": N, "programs": {...}, "fallback": bool})을 출력한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YMT3_ROOT = Path(os.environ.get("YOURMT3_DIR", ROOT / "third_party" / "yourmt3"))
AMT_SRC = YMT3_ROOT / "amt" / "src"

CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
MODEL_ARGS = [
    CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
    "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe",
    "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
    "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32",
]
BASS_PROGRAMS = set(range(32, 40))
ONSET_GROUP_SEC = 0.03


def load_model(device: str):
    os.chdir(YMT3_ROOT)  # config의 상대경로(amt/logs)를 그대로 쓰기 위함
    sys.path.insert(0, str(AMT_SRC))
    sys.path.insert(0, str(YMT3_ROOT))
    from model_helper import load_model_checkpoint

    return load_model_checkpoint(args=MODEL_ARGS, device=device)


def infer_notes(model, wav: Path, device: str):
    import torch
    import torchaudio
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    import soundfile as sf

    # torchaudio 2.9+의 load는 torchcodec을 요구하므로 soundfile로 읽는다.
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    frames = model.audio_cfg["input_frames"]
    segs = slice_padded_array(audio, frames, frames)
    segs = torch.from_numpy(segs.astype("float32")).to(device).unsqueeze(1)

    with torch.inference_mode():
        pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=segs)

    starts = [frames * i / model.audio_cfg["sample_rate"] for i in range(segs.shape[0])]
    per_channel = []
    for ch in range(model.task_manager.num_decoding_channels):
        arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped, _events, _err = model.task_manager.detokenize_list_batches(
            arr_ch, starts, return_events=True
        )
        notes_ch, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
        per_channel.append(notes_ch)
    return mix_notes(per_channel)


def write_midi(notes, out: Path, inverse_vocab) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    tracks: dict[tuple[int, bool], pretty_midi.Instrument] = {}
    for n in notes:
        program = n.program
        if not n.is_drum and inverse_vocab is not None:
            program = inverse_vocab.get(program, [program])[0]
        key = (0 if n.is_drum else int(program), bool(n.is_drum))
        if key not in tracks:
            tracks[key] = pretty_midi.Instrument(program=key[0], is_drum=key[1])
        end = max(float(n.offset), float(n.onset) + 0.01)
        tracks[key].notes.append(
            # MT3 계열 토크나이저는 velocity를 on/off(1/0)로만 낸다 → 고정값으로 정규화.
            pretty_midi.Note(velocity=int(n.velocity) if n.velocity and n.velocity > 1 else 100,
                             pitch=int(n.pitch), start=float(n.onset), end=end)
        )
    for inst in tracks.values():
        inst.notes.sort(key=lambda x: x.start)
        pm.instruments.append(inst)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out))


def select_bass(all_midi: Path, out: Path) -> int:
    """전체 악기 결과에서 베이스 라인을 단성으로 고른다.

    베이스 단독 오디오도 YourMT3+는 노트를 piano/guitar/bass 등으로 흩어 놓는다
    (bass_sample2 실측: bass 162 / piano 268 / guitar 154). 베이스 프로그램만 고르면
    노트 대부분을 버리므로, 드럼 제외 전체에서 동시(30ms) 온셋 묶음마다 하나만 남긴다 —
    베이스 프로그램 우선, 그다음 최저음. (eval/yourmt3_select_policy.py로 비교)
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(all_midi))
    items = sorted(
        ((n, int(i.program)) for i in pm.instruments if not i.is_drum for n in i.notes),
        key=lambda x: x[0].start,
    )
    picked, group = [], []
    for it in items + [None]:
        if it is not None and group and it[0].start - group[0][0].start <= ONSET_GROUP_SEC:
            group.append(it)
            continue
        if group:
            picked.append(min(group, key=lambda x: (0 if x[1] in BASS_PROGRAMS else 1, x[0].pitch)))
        group = [it] if it is not None else []

    out_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=33, name="Bass (YourMT3+)")
    inst.notes = [n for n, _ in picked]
    for n in inst.notes:
        if n.velocity <= 1:
            n.velocity = 100
    out_pm.instruments.append(inst)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_pm.write(str(out))
    return len(inst.notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--instrument", choices=("bass", "all"), default="all")
    ap.add_argument("--device", default=os.environ.get("YOURMT3_DEVICE", "cpu"))
    ap.add_argument("--reselect", action="store_true",
                    help="추론 없이 기존 <output>_all.mid에서 베이스만 다시 고른다")
    args = ap.parse_args()
    wav = args.input.resolve()
    out = args.output.resolve()
    all_midi = out if args.instrument == "all" else out.with_name(out.stem + "_all.mid")

    programs: dict = {}
    if not (args.reselect and all_midi.exists()):
        model = load_model(args.device)
        notes = infer_notes(model, wav, args.device)
        inverse_vocab = model.midi_output_inverse_vocab

        def gm_program(n):
            p = n.program
            return inverse_vocab.get(p, [p])[0] if inverse_vocab is not None else p

        programs = dict(Counter("drum" if n.is_drum else str(gm_program(n)) for n in notes))
        write_midi(notes, all_midi, inverse_vocab)

    n_out = select_bass(all_midi, out) if args.instrument == "bass" else None
    print(json.dumps({"notes": n_out, "programs": programs, "instrument": args.instrument}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
