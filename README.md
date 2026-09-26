# Trackform

An audio-to-MIDI workspace for turning recorded musical ideas into editable performance material.

Trackform combines automatic music transcription, stem separation, MIDI editing, and browser playback in one workflow. The browser interface runs as a Next.js application; the processing layer is a FastAPI service that calls specialist transcription models. The same processing pipeline can be packaged as a local Tauri desktop application for offline-oriented work.

## What it does

- Upload an audio file and choose a transcription mode.
- Piano transcription with Transkun and an audio-guided sustain repair pass.
- General transcription with Basic Pitch.
- Bass transcription with a Basic Pitch and audio-timing hybrid pass.
- Drum transcription with optional stem separation and drum-focused processing.
- Full stem separation through Demucs, with selectable model and device reporting.
- Preview and edit the generated MIDI in a piano-roll workspace.
- Move, resize, select, and audition MIDI notes with Tone.js instruments.
- Export generated MIDI files for continued work in a DAW or notation program.

## Project structure

```text
app/                    Next.js interface and MIDI editor
backend/main.py         FastAPI upload, job, and progress API
backend/pipeline.py     Shared transcription and separation pipeline
eval/                   Evaluation and diagnostic scripts
packaging/              PyInstaller sidecar configuration
src-tauri/              Tauri desktop shell and local backend bridge
```

The web API exposes three jobs:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/transcribe` | Piano, bass, drums, or general MIDI conversion |
| `POST /api/separate` | Demucs stem separation |
| `POST /api/drums` | Drum-focused conversion with optional separation |
| `GET /api/jobs/{job_id}` | Progress and result polling |

## Local development

### Web interface

```bash
npm install
npm run dev
```

The interface runs at `http://localhost:3000`.

### Python backend

Create a Python environment, install the pinned working dependencies, and start FastAPI:

```bash
python -m venv .venv-backend
source .venv-backend/bin/activate
pip install -r backend/requirements-working.txt
uvicorn backend.main:app --reload --port 8000
```

The browser expects the API at `http://localhost:8000` by default. Set `NEXT_PUBLIC_API_BASE_URL` when the backend runs elsewhere. For a non-local deployment, set `ALLOWED_ORIGINS` and `PUBLIC_BASE_URL` explicitly.

### Desktop build

The Tauri build packages the Python service as a sidecar. Build that sidecar first, then run the desktop shell:

```bash
npm run sidecar:build
npm run tauri:dev
```

The desktop bridge reports transcription progress through Tauri events and keeps generated files local to the job.

## Design decisions

Piano mode uses Transkun because the current workflow preserves piano sustain and timing more reliably than the general-purpose path. General mode remains based on Basic Pitch, while bass and drum paths add instrument-specific processing. These paths are intentionally separate so that a change to one instrument does not silently change the others.

The editor is deliberately usable after transcription: a generated MIDI file is loaded into the piano roll, where notes can be inspected, corrected, auditioned, and exported. This keeps the system assistive rather than treating the first model output as a finished arrangement.

## Current scope

This is an active research and production prototype. Transcription quality depends on the source recording, instrument mix, model, and local compute environment. The evaluation scripts under `eval/` document ongoing diagnostics; they are not a claim of uniform accuracy across every genre or arrangement.

## License

No license has been selected yet. Until one is added, the repository should be treated as all rights reserved.
