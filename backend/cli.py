"""JSONL command-line entry point for Trackform audio processing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import runpy
from pathlib import Path
from typing import Any

try:
    from . import pipeline
except ImportError:  # Support: python backend/cli.py ...
    import pipeline  # type: ignore[no-redef]


def emit(payload: dict[str, Any]) -> None:
    """Write exactly one machine-readable event to stdout."""
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def progress_event(progress: int, status: str) -> None:
    emit({"type": "progress", "progress": progress, "status": status})


def _dispatch_frozen_worker() -> bool:
    """Run pipeline helper scripts inside the frozen executable.

    The normal CLI remains unchanged.  PyInstaller cannot spawn a separate
    Python interpreter, so pipeline subprocesses re-enter this executable and
    dispatch to the original script/module in the frozen bundle.
    """
    if not getattr(sys, "frozen", False):
        return False

    argv = sys.argv[1:]
    if argv and argv[0].endswith(".py"):
        script_path = argv[0]
        script_dir = str(Path(script_path).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        sys.argv = [script_path, *argv[1:]]
        runpy.run_path(script_path, run_name="__main__")
        return True

    if len(argv) >= 2 and argv[0] == "--trackform-tool":
        tool = argv[1]
        sys.argv = [tool, *argv[2:]]
        if tool == "transkun":
            from transkun.transcribe import main as transkun_main

            transkun_main()
            return True
        if tool == "basic_pitch":
            from basic_pitch.predict import main as basic_pitch_main

            basic_pitch_main()
            return True

    if len(argv) >= 2 and argv[0] == "-m":
        module = argv[1]
        sys.argv = [module, *argv[2:]]
        if module == "demucs":
            from demucs.separate import main as demucs_main

            demucs_main()
            return True
        if module == "basic_pitch":
            from basic_pitch.predict import main as basic_pitch_main

            basic_pitch_main()
            return True

    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trackform audio processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Convert audio to MIDI"
    )
    transcribe_parser.add_argument(
        "--mode",
        choices=("piano", "bass", "drums", "general"),
        required=True,
    )
    transcribe_parser.add_argument("--input", required=True, type=Path)
    transcribe_parser.add_argument("--output-dir", required=True, type=Path)

    separate_parser = subparsers.add_parser(
        "separate", help="Separate audio into Demucs stems"
    )
    separate_parser.add_argument("--input", required=True, type=Path)
    separate_parser.add_argument("--output-dir", required=True, type=Path)
    separate_parser.add_argument("--model", default="htdemucs")

    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_audio = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.command == "transcribe":
        return pipeline.transcribe(
            mode=args.mode,
            input_audio=input_audio,
            output_dir=output_dir,
            progress=progress_event,
        )

    return pipeline.separate(
        input_audio=input_audio,
        output_dir=output_dir,
        model=args.model,
        progress=progress_event,
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
        if args.command == "separate":
            emit(
                {
                    "type": "completed",
                    "mode": "separate",
                    "stems": result["stems"],
                    "model": result.get("model"),
                    "device": result.get("device"),
                }
            )
        else:
            completed = {
                "type": "completed",
                "mode": result["mode"],
                "midiPath": result["midiPath"],
                "midiFileName": result["midiFileName"],
            }
            if "counts" in result:
                completed["counts"] = result["counts"]
            if "drumsStemPath" in result:
                completed["drumsStemPath"] = result["drumsStemPath"]
            emit(completed)
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        emit({"type": "error", "error": "Interrupted"})
        return 130
    except Exception as error:
        print("Trackform CLI failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "error": str(error)})
        return 1


if __name__ == "__main__":
    # PyInstaller sidecars use the same executable for multiprocessing helper
    # processes spawned by librosa/numba/torch.
    from multiprocessing import freeze_support

    freeze_support()

    if getattr(sys, "frozen", False):
        # Keep bundled model caches offline when they are present.
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        bundled_hf_home = bundle_root / "models" / "huggingface"
        if bundled_hf_home.exists():
            os.environ.setdefault("HF_HOME", str(bundled_hf_home))

        if _dispatch_frozen_worker():
            raise SystemExit(0)

    raise SystemExit(main())
