"""FastAPI compatibility layer for the shared Trackform pipelines.

The transformation implementation lives in :mod:`backend.pipeline`.  This
module keeps the existing web API available while the desktop CLI uses the
same functions directly.
"""

from pathlib import Path
from uuid import uuid4
import os
import shutil
import subprocess
import threading

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_ALLOWED_ORIGINS = ["http://localhost:3000"]


def _parse_allowed_origins(raw_origins: str | None) -> list[str]:
    """Parse comma-separated CORS origins without permitting wildcard access."""
    if not raw_origins:
        return DEFAULT_ALLOWED_ORIGINS.copy()

    origins = []
    for value in raw_origins.split(","):
        origin = value.strip()
        if origin and origin != "*" and origin not in origins:
            origins.append(origin)

    return origins or DEFAULT_ALLOWED_ORIGINS.copy()


ALLOWED_ORIGINS = _parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

JOBS: dict[str, dict] = {}


def output_url(job_id: str, file_name: str) -> str:
    return f"{PUBLIC_BASE_URL}/outputs/{job_id}/{file_name}"


async def save_upload(upload: UploadFile, dest: Path):
    def _write():
        with dest.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

    await run_in_threadpool(_write)


def update_job(job_id: str, progress=None, status=None, error=None, result=None):
    job = JOBS[job_id]
    if progress is not None:
        job["progress"] = progress
    if status is not None:
        job["status"] = status
    if error is not None:
        job["error"] = error
    if result is not None:
        job["result"] = result


def _progress_callback(job_id: str):
    def callback(progress: int, status: str):
        update_job(job_id, progress=progress, status=status)

    return callback


def _subprocess_error(error: subprocess.CalledProcessError) -> str:
    detail = (error.stderr or "").strip().splitlines()[-5:]
    return " | ".join(detail) or str(error)


def _web_midi_result(job_id: str, result: dict) -> dict:
    web_result = dict(result)
    midi_path = Path(web_result.pop("midiPath"))
    web_result["midiUrl"] = output_url(job_id, midi_path.name)
    return web_result


def _web_stem_result(job_id: str, result: dict) -> dict:
    web_result = dict(result)
    stem_paths = web_result.pop("stems")
    web_result["stems"] = {
        stem_name: output_url(job_id, Path(path).name)
        for stem_name, path in stem_paths.items()
    }
    return web_result


def _web_drum_result(job_id: str, result: dict) -> dict:
    web_result = _web_midi_result(job_id, result)
    drums_stem_path = web_result.pop("drumsStemPath", None)
    if drums_stem_path:
        web_result["drumsStemUrl"] = output_url(job_id, Path(drums_stem_path).name)
    return web_result


def process_job(job_id: str, input_audio: Path, mode: str):
    job_output_dir = OUTPUT_DIR / job_id
    try:
        result = pipeline.transcribe(
            mode=mode,
            input_audio=input_audio,
            output_dir=job_output_dir,
            progress=_progress_callback(job_id),
        )
        update_job(
            job_id,
            progress=100,
            status="Done.",
            result=_web_midi_result(job_id, result),
        )
    except subprocess.CalledProcessError as error:
        update_job(
            job_id,
            progress=100,
            status="Error.",
            error=_subprocess_error(error),
        )
    except Exception as error:
        update_job(job_id, progress=100, status="Error.", error=str(error))


def process_separate_job(job_id: str, input_audio: Path, model: str):
    job_output_dir = OUTPUT_DIR / job_id
    try:
        result = pipeline.separate(
            input_audio=input_audio,
            output_dir=job_output_dir,
            model=model,
            progress=_progress_callback(job_id),
        )
        update_job(
            job_id,
            progress=100,
            status="Done.",
            result=_web_stem_result(job_id, result),
        )
    except subprocess.CalledProcessError as error:
        update_job(
            job_id,
            progress=100,
            status="Error.",
            error="Demucs 실행 실패: " + _subprocess_error(error),
        )
    except Exception as error:
        update_job(job_id, progress=100, status="Error.", error=str(error))


def process_drums_job(job_id: str, input_audio: Path, separate: bool):
    job_output_dir = OUTPUT_DIR / job_id
    try:
        result = pipeline.run_drums_mode(
            input_audio=input_audio,
            output_dir=job_output_dir,
            progress=_progress_callback(job_id),
            separate=separate,
        )
        update_job(
            job_id,
            progress=100,
            status="Done.",
            result=_web_drum_result(job_id, result),
        )
    except subprocess.CalledProcessError as error:
        update_job(
            job_id,
            progress=100,
            status="Error.",
            error="Drum ADT 실행 실패: " + _subprocess_error(error),
        )
    except Exception as error:
        update_job(job_id, progress=100, status="Error.", error=str(error))


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    mode: str = Form("piano"),
):
    job_id = str(uuid4())
    original_suffix = Path(file.filename).suffix or ".mp3"
    input_audio = UPLOAD_DIR / f"{job_id}{original_suffix}"
    await save_upload(file, input_audio)

    JOBS[job_id] = {
        "jobId": job_id,
        "mode": mode,
        "progress": 0,
        "status": "Queued.",
        "error": None,
        "result": None,
    }
    threading.Thread(
        target=process_job,
        args=(job_id, input_audio, mode),
        daemon=True,
    ).start()
    return {"jobId": job_id}


@app.post("/api/separate")
async def separate(
    file: UploadFile = File(...),
    model: str = Form("htdemucs"),
):
    job_id = str(uuid4())
    original_suffix = Path(file.filename).suffix or ".mp3"
    input_audio = UPLOAD_DIR / f"{job_id}{original_suffix}"
    await save_upload(file, input_audio)

    JOBS[job_id] = {
        "jobId": job_id,
        "mode": "separate",
        "progress": 0,
        "status": "Queued.",
        "error": None,
        "result": None,
    }
    threading.Thread(
        target=process_separate_job,
        args=(job_id, input_audio, model),
        daemon=True,
    ).start()
    return {"jobId": job_id}


@app.post("/api/drums")
async def drums(
    file: UploadFile = File(...),
    separate: str = Form("true"),
):
    job_id = str(uuid4())
    original_suffix = Path(file.filename).suffix or ".mp3"
    input_audio = UPLOAD_DIR / f"{job_id}{original_suffix}"
    await save_upload(file, input_audio)

    do_separate = str(separate).lower() not in ("false", "0", "no")
    JOBS[job_id] = {
        "jobId": job_id,
        "mode": "drums",
        "progress": 0,
        "status": "Queued.",
        "error": None,
        "result": None,
    }
    threading.Thread(
        target=process_drums_job,
        args=(job_id, input_audio, do_separate),
        daemon=True,
    ).start()
    return {"jobId": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in JOBS:
        return {"error": "Job not found"}
    return JOBS[job_id]
