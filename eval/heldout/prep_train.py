"""Slakh train 부분집합(data/raw/slakh_train, 120곡)을 테스트와 같은 조건으로 준비한다.

30~90초 창 → eval/heldout/data/slakh_train/<Track>/{mix.wav, bass.mid, drums.mid}, 그리고
htdemucs 분리 → runs/_sep/slakh_train/<Track>/. R7(벤드 파라미터)·R6(벨로시티)·R4(드럼 파인튜닝)의
개발·학습 데이터이며 테스트 곡과 겹치지 않는다(train ID 00001~00138, test 01876~01906).
"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from fetch_heldout import cut_wav, window_notes, write_midi, WIN_START, WIN_END, BASS_PROGRAMS  # noqa: E402
import pretty_midi

RAW = ROOT / "data" / "raw" / "slakh_train"
man = json.loads((RAW / "manifest.json").read_text())
tracks = man["tracks"] if isinstance(man, dict) and "tracks" in man else man
names = [t["song"] if isinstance(t, dict) and "song" in t else (t.get("track") if isinstance(t, dict) else t) for t in tracks]
out_items = []
for song in names:
    out = HERE / "data" / "slakh_train" / song
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "mix.wav").exists():
        cut_wav(RAW / song / "mix.flac", out / "mix.wav", WIN_START, WIN_END)
        pm = pretty_midi.PrettyMIDI(str(RAW / song / "all_src.mid"))
        bass = [x for i in pm.instruments if not i.is_drum and i.program in BASS_PROGRAMS for x in i.notes]
        drums = [x for i in pm.instruments if i.is_drum for x in i.notes]
        write_midi(out / "bass.mid", window_notes(bass, WIN_START, WIN_END), False, 33)
        write_midi(out / "drums.mid", window_notes(drums, WIN_START, WIN_END), True)
    sep = HERE / "runs" / "_sep" / "slakh_train" / song
    if not (sep / "stems.json").exists():
        sep.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run([sys.executable, str(ROOT / "backend/scripts/demucs_separate.py"), str(out / "mix.wav"),
                             str(sep), "--model", "htdemucs", "--device", os.environ.get("DEMUCS_DEVICE", "mps")],
                            check=True, capture_output=True, text=True)
        info = json.loads([l for l in cp.stdout.splitlines() if l.strip().startswith("{")][-1])
        (sep / "stems.json").write_text(json.dumps({k: str(sep / v) for k, v in info["stems"].items()}))
    out_items.append({"song": song})
    print(song, flush=True)
(HERE / "data" / "slakh_train_manifest.json").write_text(json.dumps(out_items, indent=1))
