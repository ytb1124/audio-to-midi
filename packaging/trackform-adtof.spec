# ADTOF drum transcription worker for the desktop app (2026-10-25).
# TensorFlow + madmom conflict with the main sidecar's torch stack, so ADTOF ships as its own
# PyInstaller onedir next to dist/trackform-backend. The frozen pipeline calls it when the
# isolated .venv-adtof is not available. ADTOF is CC BY-NC-SA 4.0 (non-commercial).
# Build with:
#   .venv-adtof/bin/python -m PyInstaller packaging/trackform-adtof.spec --noconfirm --clean

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent

datas, binaries, hiddenimports = [], [], []
for package in ("adtof", "madmom", "tensorflow", "tf_keras", "pretty_midi", "mido", "librosa", "soundfile"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

a = Analysis(
    [str(ROOT / "backend" / "scripts" / "adtof_transcribe.py")],
    pathex=[str(ROOT / "backend" / "scripts")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "torchaudio", "IPython", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="trackform-adtof", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="trackform-adtof")
