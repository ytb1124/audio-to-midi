"""R7 전용 테스트셋: Slakh test split 151곡 중 30~90초에 베이스 벤드 노트가 2개 이상인 곡(기본 20곡 제외).
기본 홀드아웃 20곡에는 벤드 노트가 22개뿐이라 검정력이 약하다. 같은 미러, test split만 쓴다(학습·튜닝 미사용).
출력: data/slakh_bendtest/<Track>/mix.wav, data/_raw/slakh/<Track>/all_src.mid, runs/_sep/slakh_bendtest/<Track>/"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from fetch_heldout import fetch, cut_wav, slakh_test_tracks, SLAKH_HF, RAW
from pitchbend_eval import gt_items, is_bend
base = {i["song"] for i in json.loads((HERE / "data/manifest.json").read_text())["slakh"]}
picked = []
for tr in slakh_test_tracks():
    if tr in base: continue
    mid = fetch(f"{SLAKH_HF}/resolve/main/data/test/{tr}/all_src.mid", RAW / "slakh" / tr / "all_src.mid")
    try: nb = sum(is_bend(d["gt"]) for d in gt_items(mid))
    except Exception: continue
    if nb < 2: continue
    flac = fetch(f"{SLAKH_HF}/resolve/main/data/test/{tr}/mix.flac", RAW / "slakh" / tr / "mix.flac")
    out = HERE / "data" / "slakh_bendtest" / tr; out.mkdir(parents=True, exist_ok=True)
    if not (out / "mix.wav").exists(): cut_wav(flac, out / "mix.wav", 30.0, 90.0)
    sep = HERE / "runs/_sep/slakh_bendtest" / tr
    if not (sep / "stems.json").exists():
        sep.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run([sys.executable, str(ROOT / "backend/scripts/demucs_separate.py"), str(out / "mix.wav"), str(sep),
                             "--model", "htdemucs", "--device", "mps"], check=True, capture_output=True, text=True)
        info = json.loads([l for l in cp.stdout.splitlines() if l.strip().startswith("{")][-1])
        (sep / "stems.json").write_text(json.dumps({k: str(sep / v) for k, v in info["stems"].items()}))
    picked.append({"song": tr, "bend_notes": nb}); print(tr, nb, flush=True)
(HERE / "data/bendtest_manifest.json").write_text(json.dumps(picked, indent=1))
print(len(picked), sum(p["bend_notes"] for p in picked))
