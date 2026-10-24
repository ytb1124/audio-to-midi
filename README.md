# Trackform

An audio-to-MIDI workspace for turning recorded musical ideas into editable performance material.

Trackform combines automatic music transcription, stem separation, MIDI editing, and browser playback in one workflow. The browser interface runs as a Next.js application; the processing layer is a FastAPI service that calls specialist transcription models. The same processing pipeline can be packaged as a local Tauri desktop application for offline-oriented work.

## What it does

- Upload an audio file and choose a transcription mode.
- Piano transcription with Transkun and an audio-guided sustain repair pass.
- General transcription with Basic Pitch, or YourMT3+ as an optional backend.
- Bass transcription with a Basic Pitch and audio-timing hybrid pass, or YourMT3+ as an optional backend.
- Drum transcription on a separated drum stem with ADTOF, with velocity taken from each drum class's own frequency band. The in-house E-GMD classifier remains as a fallback and as an option.
- Optional beat grid: Beat This! tracks beats and downbeats, the MIDI file gets a per-beat tempo map and time signature, and notes can be pulled toward a 16th-note grid with an adjustable strength (default 0, so note times stay as played).
- Optional bass pitch bends: slides and bends are written as MIDI pitch bend (±2 semitones) from the pYIN pitch curve, only where the pitch actually moves.
- Full stem separation through Demucs, with selectable model and device reporting.
- Preview and edit the generated MIDI in a piano-roll workspace.
- Move, resize, select, and audition MIDI notes with built-in sampled instruments that work offline: grand piano (4 velocity layers), electric and upright bass, acoustic drum kit, electric piano, strings and pad. Pitch bends are drawn on the notes and played back.
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

Optional post-processing fields (omit them and the output is unchanged):

| Field | Values | Applies to |
| --- | --- | --- |
| `beatGrid` | `true` / `false` (default) | `/api/transcribe`, `/api/drums` |
| `quantize` | `0`–`1`, default `0` | with `beatGrid` |
| `pitchBend` | `true` / `false` (default) | `/api/transcribe` with `mode=bass` |

The full contract, including the result fields, is in `CLAUDE.md`.

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

The desktop bridge reports transcription progress through Tauri events and keeps generated files local to the job. The sidecar bundles Transkun, Basic Pitch, HT-Demucs, Beat This! and the in-house classifiers. ADTOF ships as a second bundled worker (`packaging/trackform-adtof.spec`, built from `.venv-adtof`), so the desktop app uses the same default drum model as the web backend. YourMT3+ is not bundled; it is optional and not a default.

```bash
.venv-adtof/bin/python -m PyInstaller packaging/trackform-adtof.spec --noconfirm --clean --distpath dist --workpath packaging/build
```

Prebuilt macOS (Apple Silicon) builds are attached to [GitHub Releases](https://github.com/ytb1124/audio-to-midi/releases). They are not code-signed. On first launch, right-click the app and choose **Open**.

## Evaluation

Model choices are made on a held-out benchmark. The earlier development set (6 bass and 8 drum recordings in `eval/ground_truth/`) was also used to tune the current bass and drum pipelines, so scores on it were optimistic: bass 0.886 and drums 0.776 there, against 0.57 and 0.55 on unseen music.

The held-out benchmark (`eval/heldout/`) uses public data that none of the compared systems were trained on:

- Slakh2100 test (20 rendered mixes): bass and drums
- MDB Drums (23 real recordings): drums

Every system is measured on the same path: HT-Demucs separation, then transcription. Scores are per-song mean F1 at 50 ms onset tolerance. Brackets show the 95% bootstrap interval of the per-song difference from the previous default.

Version 0.3.0 adds a real-bass test: the same Slakh bass parts re-rendered with recorded electric and upright bass samples (CC0), mixed with the original accompaniment.

| Task | Default | Best alternative tested | Result |
| --- | --- | --- | --- |
| Drums, real recordings (MDB) | ADTOF on HT-Demucs stem 0.859 | SCNet-XL stem → ADTOF 0.870 | +0.011 [−0.002, +0.026], kept HT-Demucs |
| Bass, real-bass renders (electric / upright) | Basic Pitch + hybrid 0.287 / 0.480 | YourMT3+ on the stem 0.101 / 0.252 | −0.19 / −0.23, kept the hybrid |
| Bass, Slakh rendering | Basic Pitch + hybrid 0.582 | YourMT3+ on the stem 0.733 | +0.15, but the lead reverses on real bass |
| Beats (Slakh, 18 songs) | — (librosa fallback 0.809) | **Beat This! 0.927** | +0.118 [+0.033, +0.220] |
| Drum velocity (Spearman ρ) | attack loudness 0.330 | **per-class band energy 0.620** | +0.290 [+0.152, +0.429] |
| Bass pitch bends (bent-note F1) | none | **pYIN 0.813** | CREPE 0.748, RMVPE 0.526 |

YourMT3+ was trained on Slakh's training split. Fed the separated bass stem, its lead on Slakh rendering did not carry over to recorded bass; fed the full mix, it led on all three bass sets (+0.17 to +0.27), which points to a separate full-song bass route. The bass hybrid's post-processing, which looked neutral on Slakh, gains +0.11 to +0.14 over raw Basic Pitch on recorded bass.

Every candidate, its confidence interval, the comparison with published results, and why the search stopped are in [`eval/BEST_RESULTS.md`](eval/BEST_RESULTS.md). The earlier tables are in [`eval/heldout/RESULTS_HELDOUT.md`](eval/heldout/RESULTS_HELDOUT.md).

## Design decisions

Piano mode uses Transkun because the current workflow preserves piano sustain and timing more reliably than the general-purpose path. General mode remains based on Basic Pitch, while bass and drum paths add instrument-specific processing. These paths are intentionally separate so that a change to one instrument does not silently change the others. Piano output is locked by a regression check: a fixed 30-second fixture must produce a MIDI file with an unchanged SHA-256.

The editor is deliberately usable after transcription: a generated MIDI file is loaded into the piano roll, where notes can be inspected, corrected, auditioned, and exported. This keeps the system assistive rather than treating the first model output as a finished arrangement.

## Current scope

This is an active research prototype. Transcription quality depends on the source recording, instrument mix, model, and local compute environment. The benchmark above covers a limited set of public recordings; it is not a claim of uniform accuracy across every genre or arrangement. Known gaps:

- The bass test uses real recorded single notes re-rendered from MIDI, not full real performances; a MIDI-pickup recording set does not exist yet.
- ADTOF outputs five drum classes, so it does not separate open and closed hi-hat, or crash and ride.
- The ADTOF fine-tuning experiment (`eval/ADTOF_FINETUNE_PLAN.md`) was not run: its rehearsal dataset is available on request only.
- Editor sample sources and licenses are listed in `public/samples/LICENSES.md`.

## License

No license has been selected for this repository yet. Until one is added, it should be treated as all rights reserved.

Third-party models keep their own terms. **ADTOF is CC BY-NC-SA 4.0 (non-commercial)**, and several checkpoints (Transkun, YourMT3+, HT-Demucs) were trained on datasets with non-commercial terms. The project currently uses them for research. A commercial build would need to switch drums back to `egmd` (`DRUMS_BACKEND=egmd`) and review the remaining models.
