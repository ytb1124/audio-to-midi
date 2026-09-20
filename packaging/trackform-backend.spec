# Apple Silicon development sidecar for Trackform.
# Build with:
#   .venv-backend/bin/python -m PyInstaller packaging/trackform-backend.spec

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPECPATH).resolve().parent
BACKEND = ROOT / "backend"

datas = [
    (str(BACKEND / "scripts"), "backend/scripts"),
]
binaries = []
hiddenimports = [
    "backend.cli",
    "backend.pipeline",
    "transkun.transcribe",
    "demucs.separate",
    "basic_pitch.predict",
]

# These packages use dynamic imports and/or ship native libraries and model data.
# collect_all keeps the spec resilient to small upstream package layout changes.
for package in (
    "torch",
    "torchaudio",
    "demucs",
    "transkun",
    "basic_pitch",
    "librosa",
    "scipy",
    "sklearn",
    "joblib",
    "soundfile",
    "soxr",
    "resampy",
    "pretty_midi",
    "mido",
    "numba",
    "llvmlite",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

# A few optional backends are loaded by name at runtime.
for package in ("torch", "torchaudio", "demucs", "transkun", "basic_pitch"):
    hiddenimports.extend(collect_submodules(package))

# Demucs resolves HT-Demucs through the Hugging Face cache. Include the cache
# files as regular files so the sidecar can run offline without relying on a
# user's home-directory cache. The cache is optional during development; if it
# is absent, Demucs keeps its existing download behavior.
demucs_cache = Path.home() / ".cache/huggingface/hub/models--adefossez--HTDemucs"
demucs_snapshot = "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"
if demucs_cache.exists():
    snapshot_dir = demucs_cache / "snapshots" / demucs_snapshot
    for file_name in ("htdemucs.yaml", "955717e8.safetensors"):
        source = snapshot_dir / file_name
        if source.exists():
            datas.append(
                (
                    str(source),
                    f"models/huggingface/hub/models--adefossez--HTDemucs/snapshots/{demucs_snapshot}",
                )
            )
    refs_main = demucs_cache / "refs" / "main"
    if refs_main.exists():
        datas.append(
            (
                str(refs_main),
                "models/huggingface/hub/models--adefossez--HTDemucs/refs",
            )
        )

# Avoid duplicate entries introduced by collect_all + collect_submodules.
datas = list(dict.fromkeys(datas))
binaries = list(dict.fromkeys(binaries))
hiddenimports = sorted(set(hiddenimports))

analysis = Analysis(
    [str(BACKEND / "cli.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    name="trackform-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    exclude_binaries=True,
)

COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="trackform-backend",
)
