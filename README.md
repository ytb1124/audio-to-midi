# Trackform

An audio-to-MIDI workspace for turning recorded musical ideas into editable performance material.

Trackform combines automatic music transcription, stem separation, MIDI editing, and browser playback in one workflow. The browser interface runs as a Next.js application; the processing layer is a FastAPI service that calls specialist transcription models. The same processing pipeline can be packaged as a local Tauri desktop application for offline-oriented work.

## What it does

- Upload an audio file and choose a transcription mode.
- Piano transcription with Transkun and an audio-guided sustain repair pass.
- General transcription with Basic Pitch, or YourMT3+ as an optional backend.
- Bass transcription with a Basic Pitch and audio-timing hybrid pass, or YourMT3+ as an optional backend.
- Drum transcription on a separated drum stem with ADTOF, with velocity taken from attack loudness. The in-house E-GMD classifier remains as a fallback and as an option.
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
eval/heldout/           Held-out benchmark on public datasets (real and rendered mixes)
packaging/              PyInstaller sidecar configuration
src-tauri/              Tauri desktop shell and local backend bridge
```

The web API exposes these jobs:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/transcribe` | Piano, bass, drums, or general MIDI conversion |
| `POST /api/separate` | Demucs stem separation |
| `POST /api/drums` | Drum-focused conversion with optional separation |
| `GET /api/jobs/{job_id}` | Progress and result polling |

`/api/transcribe` and `/api/drums` accept an optional `backend` field:

| Mode | Backends | Default |
| --- | --- | --- |
| `bass` | `basicpitch`, `yourmt3`, `yourmt3_hybrid` | `basicpitch` |
| `general` | `basicpitch`, `yourmt3` | `basicpitch` |
| `drums` | `adtof`, `egmd` | `adtof` |

The result reports the engine that actually ran as `backend`. If an optional model is not installed, the job falls back to the built-in engine (`basicpitch` or `egmd`) instead of failing.

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

### Optional model backends

YourMT3+ and ADTOF need dependency stacks that conflict with the main backend environment (different PyTorch, NumPy and TensorFlow versions), so each runs from its own virtual environment. The setup script downloads code and checkpoints from the official sources (HuggingFace Space `mimbres/YourMT3`, GitHub `MZehren/ADTOF`):

```bash
bash backend/scripts/setup_isolated_models.sh
```

This creates `.venv-yourmt3/`, `.venv-adtof/` and `third_party/`, all ignored by git. Without them, drums fall back to the E-GMD classifier and the note modes use Basic Pitch.

### Desktop build

The Tauri build packages the Python service as a sidecar. Build that sidecar first, then run or bundle the desktop shell:

```bash
npm run sidecar:build
npm run tauri:dev     # development window
npm run tauri:build   # .app and .dmg under src-tauri/target/release/bundle/
```

The desktop bridge reports transcription progress through Tauri events and keeps generated files local to the job. The sidecar bundles Transkun, Basic Pitch, HT-Demucs and the in-house classifiers. It does not bundle the optional YourMT3+ and ADTOF environments, so **the desktop app transcribes drums with the E-GMD classifier**.

Prebuilt macOS (Apple Silicon) builds are attached to [GitHub Releases](https://github.com/ytb1124/audio-to-midi/releases). They are not code-signed. On first launch, right-click the app and choose **Open**.

## Evaluation

Model choices are made on a held-out benchmark. The earlier development set (6 bass and 8 drum recordings in `eval/ground_truth/`) was also used to tune the current bass and drum pipelines, so scores on it were optimistic: bass 0.886 and drums 0.776 there, against 0.57 and 0.55 on unseen music.

The held-out benchmark (`eval/heldout/`) uses public data that none of the compared systems were trained on:

- Slakh2100 test (20 rendered mixes): bass and drums
- MDB Drums (23 real recordings): drums

Every system is measured on the same path: HT-Demucs separation, then transcription. Scores are per-song mean F1 at 50 ms onset tolerance. Brackets show the 95% bootstrap interval of the per-song difference from the previous default.

| Task | Previous default | Best tested | Change |
| --- | --- | --- | --- |
| Drums, real recordings (MDB) | E-GMD classifier 0.546 | **ADTOF on drum stem 0.859** | +0.31 [+0.25, +0.38] |
| Drums, rendered (Slakh) | E-GMD classifier 0.663 | YourMT3+ on full mix 0.806 | +0.14 [+0.06, +0.22] |
| Bass, rendered (Slakh, pitch-class F1) | Basic Pitch + hybrid 0.574 | YourMT3+ on full mix 0.850 | +0.28 [+0.20, +0.36] |

Drums now default to ADTOF because its lead holds on real recordings; cymbal F1 rose from 0.00 to 0.68. YourMT3+ leads on rendered audio but falls to 0.636 on real drum recordings. Bass keeps its current default until a real-recording bass test set exists. YourMT3+ was trained on Slakh's train split, which favours it on the Slakh test songs.

Full tables, scoring rules and reproduction commands are in [`eval/heldout/RESULTS_HELDOUT.md`](eval/heldout/RESULTS_HELDOUT.md). The next drum experiment is planned in [`eval/ADTOF_FINETUNE_PLAN.md`](eval/ADTOF_FINETUNE_PLAN.md).

## Design decisions

Piano mode uses Transkun because the current workflow preserves piano sustain and timing more reliably than the general-purpose path. General mode remains based on Basic Pitch, while bass and drum paths add instrument-specific processing. These paths are intentionally separate so that a change to one instrument does not silently change the others. Piano output is locked by a regression check: a fixed 30-second fixture must produce a MIDI file with an unchanged SHA-256.

The editor is deliberately usable after transcription: a generated MIDI file is loaded into the piano roll, where notes can be inspected, corrected, auditioned, and exported. This keeps the system assistive rather than treating the first model output as a finished arrangement.

## Current scope

This is an active research prototype. Transcription quality depends on the source recording, instrument mix, model, and local compute environment. The benchmark above covers a limited set of public recordings; it is not a claim of uniform accuracy across every genre or arrangement. Known gaps:

- There is no real-recording bass test set yet.
- ADTOF outputs five drum classes, so it does not separate open and closed hi-hat, or crash and ride.
- Velocity is mapped from attack loudness but has not been validated against reference velocities.
- Pitch bends are not exported.

## License

No license has been selected for this repository yet. Until one is added, it should be treated as all rights reserved.

Third-party models keep their own terms. **ADTOF is CC BY-NC-SA 4.0 (non-commercial)**, and several checkpoints (Transkun, YourMT3+, HT-Demucs) were trained on datasets with non-commercial terms. The project currently uses them for research. A commercial build would need to switch drums back to `egmd` (`DRUMS_BACKEND=egmd`) and review the remaining models.
