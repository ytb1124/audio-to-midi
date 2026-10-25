"""Round 2 N3 — ADT_STR(github.com/pier-maker92/ADT_STR, HF Pierfrancesco/adt-str) 추론 워커.

.venv-adtstr/bin/python으로 실행한다. 모델 코드와 가중치는 HF 스냅샷(eval/heldout/runs/_r2/adtstr, gitignore)에서 읽는다.
모델 출력 pitch는 GM이 아니라 저장소 고유의 26클래스 코드(utils/mapping_utils.py의
GM_standard_midi_to_Gm_custom_Mapping 값)다. 여기서 대표 GM 번호로 되돌려 쓴다 — 채점은 eval/score_drums.py의
CLASS_MAP(GM)으로 하므로, 이 변환이 없으면 예컨대 custom 44(오픈 하이햇)가 GM 44(페달 하이햇)로 잘못 읽힌다.

입력: --jobs <json>  [[audio, out_mid], ...]   (곡마다 모델을 다시 올리지 않으려고 목록으로 받는다)
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "runs" / "_r2" / "adtstr"

# 저장소 custom 코드 → 대표 GM 번호(같은 custom으로 합쳐진 GM 중 우리 CLASS_MAP에서 같은 클래스인 것)
CUSTOM_TO_GM = {
    35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40,
    41: 43,  # floor tom(41·43·45 합침) → tom
    42: 42,  # closed hi-hat
    43: 44,  # pedal hi-hat → hihat_closed
    44: 46,  # open hi-hat
    45: 47,  # mid tom(47·48) → tom
    46: 49,  # crash(49·57) → crash
    47: 50,  # high tom → tom
    48: 51,  # ride(51·53·59) → ride
    49: 52,  # chinese → crash
    50: 54,  # tambourine → 채점 클래스 밖
    51: 55,  # splash → crash
    52: 56, 53: 58, 54: 63, 55: 70, 56: 72, 57: 74, 58: 75, 59: 79, 60: 81,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="setting-tau-0.8")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    import pretty_midi
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(REPO))
    from adt_transcriber import ADTTranscriber

    tr = ADTTranscriber.from_pretrained(str(REPO), variant=a.variant, device=a.device)
    for audio, out in json.loads(Path(a.jobs).read_text()):
        t = time.time()
        wav = tr._load_audio(audio)
        notes = tr._transcribe_waveform(wav)
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        raw = notes.tolist() if hasattr(notes, "tolist") else list(notes)
        unknown = 0
        for row in raw:
            on, off, pitch, vel = row[:4]
            gm = CUSTOM_TO_GM.get(int(pitch))
            if gm is None:
                unknown += 1
                continue
            inst.notes.append(pretty_midi.Note(int(max(1, min(127, vel))), gm, float(on), float(off)))
        pm.instruments.append(inst)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        pm.write(out)
        print(json.dumps({"out": out, "notes": len(inst.notes), "unknown": unknown, "seconds": round(time.time() - t, 1)}), flush=True)


if __name__ == "__main__":
    main()
