"""Round 2 N1 — MuScriptor(github.com/muscriptor/muscriptor) 추론 워커. .venv-muscriptor/bin/python으로 실행한다.

가중치는 HF 게이트 저장소(MuScriptor/muscriptor-<size>, CC BY-NC 4.0)라 사용자가 `hf auth login` + 라이선스 동의를 해야 받힌다.
템포 검출·퀀타이즈 없이 원시 노트를 MIDI로 쓴다(전체 악기; 채점은 is_drum 트랙만 본다). 디코딩은 기본값(greedy, cfg 1.0).

입력: --jobs <json> [[audio, out_mid, instruments(list|null)], ...]
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    from muscriptor.transcription_model import TranscriptionModel

    torch.set_num_threads(2)
    tm = TranscriptionModel.load_model(a.model, device=a.device)
    for audio, out, instruments in json.loads(Path(a.jobs).read_text()):
        t = time.time()
        events = list(tm.transcribe(audio, instruments=instruments))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(tm.events_to_midi_bytes(iter(events)))
        print(json.dumps({"out": out, "seconds": round(time.time() - t, 1), "device": str(tm._device)}), flush=True)


if __name__ == "__main__":
    main()
