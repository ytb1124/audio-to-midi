"""Alternative beat/downbeat trackers. Run with .venv-beat/bin/python.
Output: eval/heldout/runs/_beats/<tracker>/<song>.json {beats, downbeats, seconds}
"""
import argparse, json, tempfile, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def make_beat_this():
    from beat_this.inference import File2Beats
    f = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    def run(p):
        b, d = f(str(p))
        return list(map(float, b)), list(map(float, d))
    return run


def make_madmom():
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    rdb = RNNDownBeatProcessor()
    ddb = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    rb = RNNBeatProcessor()
    db = DBNBeatTrackingProcessor(fps=100)
    def run(p):
        arr = ddb(rdb(str(p)))
        downs = [float(t) for t, k in arr if int(k) == 1]
        beats = [float(x) for x in db(rb(str(p)))]
        return beats, downs
    return run


def make_allin1():
    import allin1
    tmp = Path(tempfile.mkdtemp(prefix="allin1_"))
    def run(p):
        r = allin1.analyze(str(p), device="cpu", demix_dir=tmp / "demix",
                           spec_dir=tmp / "spec", keep_byproducts=False)
        return list(map(float, r.beats)), list(map(float, r.downbeats))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, choices=["beat_this", "madmom", "allin1"])
    ap.add_argument("--input-root", default=None)
    a = ap.parse_args()
    if a.input_root:
        root = Path(a.input_root)
        songs = sorted(d.name for d in root.iterdir() if (d / "mix.wav").exists())
    else:
        root = HERE / "data" / "slakh"
        man = json.load(open(HERE / "data" / "manifest.json"))["slakh"]
        songs = [m["song"] if isinstance(m, dict) else m for m in man]
    out = HERE / "runs" / "_beats" / a.tracker
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in songs if not (out / f"{s}.json").exists()]
    if not todo:
        print("nothing to do"); return
    run = {"beat_this": make_beat_this, "madmom": make_madmom, "allin1": make_allin1}[a.tracker]()
    for s in todo:
        t = time.time()
        try:
            b, d = run(root / s / "mix.wav")
        except Exception as e:
            print(f"FAIL {s}: {type(e).__name__}: {e}"); continue
        sec = time.time() - t
        json.dump({"beats": b, "downbeats": d, "seconds": sec}, open(out / f"{s}.json", "w"))
        print(f"{s}: {len(b)} beats {len(d)} downbeats {sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
