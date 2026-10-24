#!/usr/bin/env python
"""Alternative separators -> bass.wav / drums.wav. Run with .venv-sep/bin/python.
  --model {htdemucs_ft,scnet_xl,bs_roformer,mdx23c}  --sets slakh mdb [--songs a b]
  --input-root DIR --set-name NAME   (DIR/<song>/mix.wav, for sets not in manifest)
Output: eval/heldout/runs/_sep_alt/<model>/<set>/<song>/{bass,drums}.wav + stems.json
"""
import argparse, json, os, sys, time, shutil, tempfile, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELD = ROOT / "eval" / "heldout"
MSST = ROOT / "third_party" / "MSST"
W = MSST / "weights"
MODELS = {
    "htdemucs_ft": dict(backend="audio_separator", file="htdemucs_ft.yaml"),
    "scnet_xl": dict(backend="msst", type="scnet", cfg="config_musdb18_scnet_xl.yaml",
                     ckpt="model_scnet_ep_54_sdr_9.8051.ckpt"),
    "bs_roformer": dict(backend="msst", type="bs_roformer", cfg="config_bs_roformer_384_8_2_485100.yaml",
                        ckpt="model_bs_roformer_ep_17_sdr_9.6568.ckpt"),
    "mdx23c": dict(backend="msst", type="mdx23c", cfg="config_musdb18_mdx23c.yaml",
                   ckpt="model_mdx23c_ep_168_sdr_7.0207.ckpt"),
}


def pick_device(force=None):
    import torch
    if force:
        return force
    return "mps" if torch.backends.mps.is_available() else "cpu"


class MSSTSep:
    def __init__(self, spec, device):
        sys.path.insert(0, str(MSST))
        import torch
        from utils.settings import get_model_from_config
        self.torch = torch
        self.spec = spec
        self.model, self.config = get_model_from_config(spec["type"], str(W / spec["cfg"]))
        sd = torch.load(str(W / spec["ckpt"]), map_location="cpu", weights_only=False)
        for k in ("state", "state_dict", "model_state_dict"):
            if k in sd:
                sd = sd[k]
        self.model.load_state_dict(sd)
        # keep memory modest (SCNet XL at batch 4 blew past 26GB on MPS)
        self.config.inference.batch_size = 1
        self.model.eval()
        self.set_device(device)

    def set_device(self, device):
        self.device = device
        self.model = self.model.to(device)

    def run(self, mix_path):
        import librosa, numpy as np
        from utils.model_utils import demix
        from utils.audio_utils import normalize_audio, denormalize_audio
        sr = 44100
        mix, _ = librosa.load(mix_path, sr=sr, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        norm = None
        if self.config.inference.get("normalize", False):
            mix_n, norm = normalize_audio(mix)
        else:
            mix_n = mix
        with self.torch.no_grad():
            out = demix(self.config, self.model, mix_n, self.torch.device(self.device),
                        model_type=self.spec["type"], pbar=False)
        res = {}
        for k in ("bass", "drums"):
            e = out[k]
            if norm is not None:
                e = denormalize_audio(e, norm)
            res[k] = np.asarray(e).T
        return res, sr


class DemucsSep:
    def __init__(self, spec, device):
        from audio_separator.separator import Separator
        self.tmp = tempfile.mkdtemp(prefix="sepalt_")
        self.sep = Separator(output_dir=self.tmp, model_file_dir=str(ROOT / "third_party" / "audio-separator-models"),
                             output_format="WAV", log_level=30)
        self.sep.load_model(spec["file"])
        self.device = str(self.sep.torch_device)

    def set_device(self, device):
        pass

    def run(self, mix_path):
        import soundfile as sf
        for f in Path(self.tmp).glob("*"):
            f.unlink()
        files = self.sep.separate(mix_path, custom_output_names={"Bass": "bass", "Drums": "drums"})
        res = {}
        for k in ("bass", "drums"):
            p = Path(self.tmp) / f"{k}.wav"
            if not p.exists():
                cand = [Path(self.tmp) / f for f in files if k in Path(f).name.lower()]
                p = cand[0]
            a, sr = sf.read(str(p), always_2d=True)
            res[k] = a
        return res, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--input-root", default=None)
    ap.add_argument("--set-name", default=None)
    ap.add_argument("--device", default=None, help="mps|cpu (default mps if available, cpu fallback on error)")
    ap.add_argument("--out-root", default=str(HELD / "runs" / "_sep_alt"))
    a = ap.parse_args()

    jobs = []  # (set, song, mix_path)
    manifest = json.load(open(HELD / "data" / "manifest.json"))
    for s in a.sets:
        if s in manifest:
            for e in manifest[s]:
                jobs.append((s, e["song"], HELD / "data" / s / e["song"] / "mix.wav"))
        else:
            d = HELD / "data" / s
            for p in sorted(d.glob("*/mix.wav")):
                jobs.append((s, p.parent.name, p))
    if a.input_root:
        name = a.set_name or Path(a.input_root).name
        for p in sorted(Path(a.input_root).glob("*/mix.wav")):
            jobs.append((name, p.parent.name, p))
    if a.songs:
        jobs = [j for j in jobs if j[1] in a.songs]

    out_base = Path(a.out_root) / a.model
    todo = [j for j in jobs if not (out_base / j[0] / j[1] / "stems.json").exists()]
    print(f"[{a.model}] {len(jobs)} songs, {len(todo)} to do", flush=True)
    if not todo:
        return
    import soundfile as sf
    spec = MODELS[a.model]
    device = pick_device(a.device)
    cls = MSSTSep if spec["backend"] == "msst" else DemucsSep
    sep = cls(spec, device)
    print(f"[{a.model}] device={getattr(sep,'device',device)}", flush=True)
    ok = fail = 0
    for s, song, mix in todo:
        t0 = time.time()
        try:
            try:
                res, sr = sep.run(str(mix))
            except Exception as e:
                if sep.device == "cpu":
                    raise
                print(f"  {song}: {sep.device} failed ({e!r}); retry on cpu", flush=True)
                sep.set_device("cpu")
                res, sr = sep.run(str(mix))
            od = out_base / s / song
            od.mkdir(parents=True, exist_ok=True)
            paths = {}
            for k in ("bass", "drums"):
                sf.write(str(od / f"{k}.wav"), res[k], sr, subtype="PCM_16")
                paths[k] = str(od / f"{k}.wav")
            json.dump(paths, open(od / "stems.json", "w"))
            ok += 1
            print(f"{s}/{song}: {time.time()-t0:.1f}s device={sep.device}", flush=True)
        except Exception:
            fail += 1
            print(f"{s}/{song}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[{a.model}] done ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
