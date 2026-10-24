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
YOURMT3_SCRIPT_PATH = BASE_DIR / "backend" / "scripts" / "yourmt3_transcribe.py"
ADTOF_SCRIPT_PATH = BASE_DIR / "backend" / "scripts" / "adtof_transcribe.py"


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


def _isolated_py(venv_name: str) -> str | None:
    """YourMT3/ADTOF처럼 의존성이 충돌하는 모델용 격리 venv 인터프리터."""
    candidate = BASE_DIR / venv_name / "bin" / "python"
    return str(candidate) if candidate.exists() else None


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


def _try_yourmt3(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None,
    mode: str,
) -> dict | None:
    """YourMT3+로 전사한다. 격리 venv/모델이 없거나 실패하면 None(→ Basic Pitch 폴백)."""
    py = _isolated_py(".venv-yourmt3")
    if py is None or not YOURMT3_SCRIPT_PATH.exists():
        print("YourMT3+ venv not found; using Basic Pitch.", file=sys.stderr)
        return None
    final_midi = output_dir / f"{mode}_yourmt3.mid"
    instrument = "bass" if mode == "bass" else "all"
    _emit(progress, 25, "Running YourMT3+ transcription...")
    try:
        completed = _run(
            [py, str(YOURMT3_SCRIPT_PATH), str(input_audio), str(final_midi),
             "--instrument", instrument]
        )
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"YourMT3+ failed ({error}); using Basic Pitch.", file=sys.stderr)
        return None
    if not final_midi.exists():
        return None
    info = _last_json_line(completed.stdout)
    return {
        "midiPath": str(final_midi.resolve()),
        "midiFileName": final_midi.name,
        "mode": mode,
        "backend": "yourmt3",
        "yourmt3": info,
    }


def run_bass_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    backend: str = "basicpitch",
) -> dict:
    _emit(progress, 10, "Uploaded audio.")

    raw_midi: Path | None = None
    if backend in ("yourmt3", "yourmt3_hybrid"):
        result = _try_yourmt3(input_audio, output_dir, progress, "bass")
        if result is not None and backend == "yourmt3":
            return result
        if result is not None:
            raw_midi = Path(result["midiPath"])
        else:
            _emit(progress, 20, "YourMT3+ unavailable; falling back to Basic Pitch.")

    if raw_midi is None:
        basic_pitch_dir = output_dir / "basic_pitch_bass_raw"
        basic_pitch_dir.mkdir(exist_ok=True)

        _emit(progress, 25, "Running Basic Pitch for bass pitch candidates...")
        _run(_basic_pitch_cmd(str(basic_pitch_dir), str(input_audio)))

        raw_midi = _find_midi_in_dir(basic_pitch_dir)
        if raw_midi is None:
            raise RuntimeError("No MIDI file generated by Basic Pitch.")
        candidate_backend = "basicpitch"
    else:
        candidate_backend = "yourmt3_hybrid"

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
    if candidate_backend == "yourmt3_hybrid":
        bass_command.append("--use-given-raw")
    _run(bass_command)

    if not final_midi.exists():
        raise RuntimeError(f"Bass MIDI was not created: {final_midi}")

    return {
        "midiPath": str(final_midi.resolve()),
        "midiFileName": final_midi.name,
        "mode": "bass",
        "backend": candidate_backend,
    }


def run_general_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    backend: str = "basicpitch",
) -> dict:
    _emit(progress, 10, "Uploaded audio.")
    if backend in ("yourmt3", "yourmt3_hybrid"):  # hybrid 후처리는 베이스 전용
        result = _try_yourmt3(input_audio, output_dir, progress, "general")
        if result is not None:
            return result
        _emit(progress, 20, "YourMT3+ unavailable; falling back to Basic Pitch.")
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


def _try_adtof(
    adt_input: Path,
    output_dir: Path,
    progress: ProgressCallback | None,
) -> dict | None:
    """ADTOF(Frame_RNN)로 드럼을 전사한다. 격리 venv가 없거나 실패하면 None(→ 기존 분류기)."""
    py = _isolated_py(".venv-adtof")
    if py is None or not ADTOF_SCRIPT_PATH.exists():
        print("ADTOF venv not found; using E-GMD classifier.", file=sys.stderr)
        return None
    drum_midi = output_dir / "drums_adtof.mid"
    _emit(progress, 65, "Transcribing drums with ADTOF...")
    try:
        completed = _run([py, str(ADTOF_SCRIPT_PATH), str(adt_input), str(drum_midi)])
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"ADTOF failed ({error}); using E-GMD classifier.", file=sys.stderr)
        return None
    if not drum_midi.exists():
        return None
    return {
        "midiPath": str(drum_midi.resolve()),
        "midiFileName": drum_midi.name,
        "mode": "drums",
        "backend": "adtof",
        "counts": _last_json_line(completed.stdout).get("counts", {}),
    }


def run_drums_mode(
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    separate: bool = False,
    backend: str = "egmd",
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

    if backend == "adtof":
        result = _try_adtof(adt_input, output_dir, progress)
        if result is not None:
            if drums_stem_path is not None:
                result["drumsStemPath"] = str(drums_stem_path.resolve())
            return result
        _emit(progress, 60, "ADTOF unavailable; falling back to E-GMD classifier.")

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


# 피아노 외 모드의 전사 백엔드 선택지. 첫 값이 기본값이며 환경변수로 바꿀 수 있다.
# yourmt3_hybrid: YourMT3+ 원출력을 베이스 후처리(onset union/옥타브/fragment)의 후보로 사용(베이스 전용).
NOTE_BACKENDS = ("basicpitch", "yourmt3", "yourmt3_hybrid")
DRUM_BACKENDS = ("egmd", "adtof")


def resolve_backend(mode: str, backend: str | None) -> str | None:
    """요청값 → 환경변수 → 기본값 순으로 모드별 백엔드를 정한다."""
    if mode in ("bass", "general"):
        choices, env = NOTE_BACKENDS, f"{mode.upper()}_BACKEND"
    elif mode == "drums":
        choices, env = DRUM_BACKENDS, "DRUMS_BACKEND"
    else:
        return None
    value = (backend or os.environ.get(env) or choices[0]).strip().lower()
    if value not in choices:
        raise ValueError(f"Unknown {mode} backend: {value} (choices: {', '.join(choices)})")
    return value


def transcribe(
    mode: str,
    input_audio: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
    backend: str | None = None,
) -> dict:
    """Run one of the four supported MIDI conversion modes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audio = input_audio.resolve()
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")

    backend = resolve_backend(mode, backend)
    if mode == "piano":
        result = run_piano_mode(input_audio, output_dir, progress)
    elif mode == "bass":
        result = run_bass_mode(input_audio, output_dir, progress, backend=backend)
    elif mode == "drums":
        result = run_drums_mode(
            input_audio, output_dir, progress, separate=False, backend=backend
        )
    elif mode == "general":
        result = run_general_mode(input_audio, output_dir, progress, backend=backend)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if backend is not None:
        # 새 백엔드 경로는 성공 시 직접 "backend"를 채운다. 비어 있으면 폴백 경로가 돈 것.
        result.setdefault("backend", NOTE_BACKENDS[0] if mode != "drums" else DRUM_BACKENDS[0])
        result["requestedBackend"] = backend

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
