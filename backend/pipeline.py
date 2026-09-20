"""Shared audio-to-MIDI pipelines for the web API and desktop CLI.

This module deliberately contains no FastAPI or HTTP-specific code.  Every
pipeline receives local paths and returns local paths so callers can decide
whether to expose those results over HTTP or through a desktop sidecar.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


if getattr(sys, "frozen", False):
    # PyInstaller onedir places Python modules and bundled data under _MEIPASS.
    BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
_VENV_BIN = BASE_DIR / ".venv-backend" / "bin"

ProgressCallback = Callable[[int, str], None]

SCRIPT_PATH = BASE_DIR / "backend" / "scripts" / "transkun_audio_sustain.py"
BASS_HYBRID_SCRIPT_PATH = (
    BASE_DIR / "backend" / "scripts" / "bass_basicpitch_audio_hybrid.py"
)
DEMUCS_SCRIPT_PATH = BASE_DIR / "backend" / "scripts" / "demucs_separate.py"
DRUM_SCRIPT_PATH = BASE_DIR / "backend" / "scripts" / "drum_audio_to_midi.py"


def _py() -> str:
    """Return the project Python interpreter when the venv is available."""
    venv_py = _VENV_BIN / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _venv_exe(name: str) -> str:
    """Resolve a console script without relying on the caller's PATH."""
    for candidate in (_VENV_BIN / name, Path(_py()).parent / name):
        if candidate.exists():
            return str(candidate)
    return shutil.which(name) or name


def _script_cmd(script: Path, *args: str) -> list[str]:
    """Run a pipeline helper through the current interpreter."""
    return [_py(), str(script), *args]


def _tool_cmd(name: str, *args: str) -> list[str]:
    """Run a console-script dependency in both venv and frozen modes."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--trackform-tool", name, *args]
    return [_venv_exe(name), *args]


def _basic_pitch_cmd(*args: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--trackform-tool", "basic_pitch", *args]
    exe = _venv_exe("basic-pitch")
    if Path(exe).exists() or shutil.which("basic-pitch"):
        return [exe, *args]
    return [_py(), "-m", "basic_pitch", *args]


def _emit(progress: ProgressCallback | None, value: int, status: str) -> None:
    if progress is not None:
        progress(value, status)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a worker command and keep its normal logs off the JSON stdout."""
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr, end="")
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise
    if completed.stdout:
        print(completed.stdout, file=sys.stderr, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed


def _find_midi_in_dir(directory: Path) -> Path | None:
    midi_files = list(directory.glob("*.mid")) + list(directory.glob("*.midi"))
    if not midi_files:
        return None
    midi_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return midi_files[0]


def _last_json_line(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {}


def run_piano_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    raw_midi = output_dir / "piano_transkun_raw.mid"
    sustain_midi = output_dir / "piano_transkun_sustain.mid"

    _emit(progress, 10, "Uploaded audio.")
    _emit(progress, 25, "Running Transkun piano transcription...")
    _run(_tool_cmd("transkun", str(input_audio), str(raw_midi)))

    _emit(progress, 75, "Repairing sustain with original audio...")
    _run(_script_cmd(SCRIPT_PATH, str(input_audio), str(raw_midi), str(sustain_midi)))

    if not sustain_midi.exists():
        raise RuntimeError(f"Piano MIDI was not created: {sustain_midi}")

    return {
        "midiPath": str(sustain_midi.resolve()),
        "midiFileName": sustain_midi.name,
        "mode": "piano",
    }


def run_bass_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    _emit(progress, 10, "Uploaded audio.")

    basic_pitch_dir = output_dir / "basic_pitch_bass_raw"
    basic_pitch_dir.mkdir(exist_ok=True)

    _emit(progress, 25, "Running Basic Pitch for bass pitch candidates...")
    _run(_basic_pitch_cmd(str(basic_pitch_dir), str(input_audio)))

    raw_midi = _find_midi_in_dir(basic_pitch_dir)
    if raw_midi is None:
        raise RuntimeError("No MIDI file generated by Basic Pitch.")

    final_midi = output_dir / "bass_hybrid.mid"
    _emit(progress, 70, "Matching bass pitch with original audio timing...")

    bass_command = _script_cmd(
        BASS_HYBRID_SCRIPT_PATH,
        str(input_audio),
        str(raw_midi),
        str(final_midi),
    )
    if os.environ.get("BASS_MTL") == "1":
        bass_command += ["--mtl-offset", "--mtl-octave-audit"]
    _run(bass_command)

    if not final_midi.exists():
        raise RuntimeError(f"Bass MIDI was not created: {final_midi}")

    return {
        "midiPath": str(final_midi.resolve()),
        "midiFileName": final_midi.name,
        "mode": "bass",
    }


def run_general_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    _emit(progress, 10, "Uploaded audio.")
    _emit(progress, 25, "Running Basic Pitch general transcription...")
    _run(_basic_pitch_cmd(str(output_dir), str(input_audio)))

    _emit(progress, 85, "Finding generated MIDI...")
    midi_file = _find_midi_in_dir(output_dir)
    if midi_file is None:
        raise RuntimeError("No MIDI file generated by Basic Pitch.")

    final_midi = output_dir / "general_basic_pitch.mid"
    if midi_file.resolve() != final_midi.resolve():
        shutil.copyfile(midi_file, final_midi)

    return {
        "midiPath": str(final_midi.resolve()),
        "midiFileName": final_midi.name,
        "mode": "general",
    }


def run_drums_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    separate: bool = False,
) -> dict:
    _emit(progress, 10, "Uploaded audio.")
    adt_input = input_audio
    drums_stem_path: Path | None = None
    device = os.environ.get("DEMUCS_DEVICE", "cpu")

    if separate:
        _emit(progress, 25, "Separating drums stem with HT-Demucs...")
        completed = _run(
            _script_cmd(
                DEMUCS_SCRIPT_PATH,
                str(input_audio),
                str(output_dir),
                "--model",
                "htdemucs",
                "--device",
                device,
            )
        )
        stems_info = _last_json_line(completed.stdout)
        drums_name = stems_info.get("stems", {}).get("drums")
        if not drums_name:
            raise RuntimeError("Demucs output did not contain a drums stem.")
        drums_stem_path = output_dir / drums_name
        adt_input = drums_stem_path

    _emit(progress, 65, "Transcribing drums (kick/snare/hihat)...")
    drum_midi = output_dir / "drums.mid"
    audit_csv = output_dir / "drum_adt_audit.csv"
    completed = _run(
        _script_cmd(
            DRUM_SCRIPT_PATH,
            str(adt_input),
            str(drum_midi),
            "--audit",
            str(audit_csv),
        )
    )
    counts = _last_json_line(completed.stdout).get("counts", {})

    if not drum_midi.exists():
        raise RuntimeError(f"Drum MIDI was not created: {drum_midi}")

    result = {
        "midiPath": str(drum_midi.resolve()),
        "midiFileName": drum_midi.name,
        "mode": "drums",
        "counts": counts,
    }
    if drums_stem_path is not None:
        result["drumsStemPath"] = str(drums_stem_path.resolve())
    return result


def transcribe(
    mode: str,
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    """Run one of the four supported MIDI conversion modes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audio = input_audio.resolve()
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")

    if mode == "piano":
        result = run_piano_mode(input_audio, output_dir, progress)
    elif mode == "bass":
        result = run_bass_mode(input_audio, output_dir, progress)
    elif mode == "drums":
        result = run_drums_mode(input_audio, output_dir, progress, separate=False)
    elif mode == "general":
        result = run_general_mode(input_audio, output_dir, progress)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _emit(progress, 100, "Done.")
    return result


def separate(
    input_audio: Path,
    output_dir: Path,
    model: str = "htdemucs",
    progress: ProgressCallback | None = None,
) -> dict:
    """Run HT-Demucs separation and return local stem paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audio = input_audio.resolve()
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")

    _emit(progress, 10, "Uploaded audio.")
    _emit(progress, 25, "Running HT-Demucs stem separation...")
    device = os.environ.get("DEMUCS_DEVICE", "cpu")
    completed = _run(
        _script_cmd(
            DEMUCS_SCRIPT_PATH,
            str(input_audio),
            str(output_dir),
            "--model",
            model,
            "--device",
            device,
        )
    )
    stems_info = _last_json_line(completed.stdout)
    stems = stems_info.get("stems", {})
    if not stems:
        raise RuntimeError("Demucs output did not contain any stems.")

    _emit(progress, 90, "Packaging stems...")
    result = {
        "stems": {
            stem_name: str((output_dir / file_name).resolve())
            for stem_name, file_name in stems.items()
        },
        "model": stems_info.get("model", model),
        "device": stems_info.get("device", device),
        "mode": "separate",
    }
    _emit(progress, 100, "Done.")
    return result
