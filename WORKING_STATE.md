# Working State Archive

## Date
2026-07-05

## Status
Piano MIDI conversion is working very well.

## Pipeline
Audio upload from web
→ Piano Mode
→ Transkun transcription
→ Audio-guided sustain repair
→ MIDI auto-load in web
→ Playback with selectable instruments

## Important
This state should be preserved before adding more editing features.

## Working features
- Piano Mode = Transkun + sustain repair
- General Mode = Basic Pitch
- MIDI playback
- Instrument selection
- Grand Piano Sample
- Soft Synth
- Pad
- Basic Synth variants
- Piano roll preview
- Play / Pause / Stop
- Timeline

## Known caution
Do not aggressively change pitch correction or quantization.
The current piano transcription quality is good because:
- Transkun is used for piano
- sustain repair uses original audio
- pitch/start timing are preserved
