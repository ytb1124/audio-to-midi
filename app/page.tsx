"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Midi } from "@tonejs/midi";
import * as Tone from "tone";
import {
  desktopAssetUrl,
  isDesktopRuntime,
  runDesktopSeparation,
  runDesktopTranscription,
} from "./lib/desktop-backend";

type ModeType = "piano" | "bass" | "drums" | "general";
type Language = "en" | "ko";

type InstrumentType = "grand_piano" | "double_bass" | "electric_bass" | "drum_kit";
type StemName = "vocals" | "drums" | "bass" | "other";

type NoteEvent = {
  id: string;
  name: string;
  midi: number;
  time: number;
  duration: number;
  velocity: number;
};

type PlaybackInstrument = {
  triggerAttackRelease: (
    noteName: string,
    duration: number,
    time?: Tone.Unit.Time,
    velocity?: number
  ) => void;
  dispose: () => void;
  _extraDisposables?: Tone.ToneAudioNode[];
};

type JobResult = {
  midiUrl: string;
  midiFileName: string;
  mode: ModeType;
};

type StemJobResult = {
  stems: Partial<Record<StemName, string>>;
  model?: string;
  device?: string;
  mode: "separate";
};

type NoteDragMode = "move" | "resize-end";

type NoteDragState = {
  mode: NoteDragMode;
  pointerId: number;
  startClientX: number;
  startClientY: number;
  targetNoteIds: Set<string>;
  originalNotes: NoteEvent[];
  latestNotes: NoteEvent[];
  didChange: boolean;
};

type MarqueeSelectionState = {
  pointerId: number;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
  initialSelectedNoteIds: Set<string>;
  appendToSelection: boolean;
  didDrag: boolean;
};

type MarqueeBox = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type TimeSignature = {
  numerator: number;
  denominator: number;
};

type MeterPreset = TimeSignature & {
  label: string;
};

type StemTrackState = Record<StemName, boolean>;

type StemMidiConversionState = {
  isProcessing: boolean;
  progress: number;
  status: string;
  result: JobResult | null;
};

type StemWaveformPeak = {
  min: number;
  max: number;
};

type StemWaveformPeaks = Partial<Record<StemName, StemWaveformPeak[]>>;

const BASE_PIXELS_PER_BEAT = 48;
const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
const PIANO_KEY_WIDTH = 124;
const BASE_NOTE_ROW_HEIGHT = 18;
const RULER_HEIGHT = 28;
const MIN_NOTE_DURATION = 0.05;
const MIN_PIANO_MIDI = 21;
const MAX_PIANO_MIDI = 108;
const STEM_ESTIMATE_STORAGE_KEY = "trackform-stem-estimate-factor";
const DEFAULT_STEM_ESTIMATE_SECONDS = 4 * 60;
const MIN_STEM_ESTIMATE_SECONDS = 90;
const MAX_STEM_ESTIMATE_SECONDS = 15 * 60;
const DEFAULT_STEM_SECONDS_PER_AUDIO_SECOND = 0.72;
const STEM_ESTIMATE_OVERHEAD_SECONDS = 28;
const MUSICAL_METER_PRESETS: MeterPreset[] = [
  { numerator: 2, denominator: 2, label: "2/2 · Cut time" },
  { numerator: 2, denominator: 4, label: "2/4" },
  { numerator: 3, denominator: 4, label: "3/4 · Waltz" },
  { numerator: 4, denominator: 4, label: "4/4 · Common time" },
  { numerator: 5, denominator: 4, label: "5/4" },
  { numerator: 6, denominator: 4, label: "6/4" },
  { numerator: 7, denominator: 4, label: "7/4" },
  { numerator: 3, denominator: 8, label: "3/8" },
  { numerator: 5, denominator: 8, label: "5/8" },
  { numerator: 6, denominator: 8, label: "6/8" },
  { numerator: 7, denominator: 8, label: "7/8" },
  { numerator: 9, denominator: 8, label: "9/8" },
  { numerator: 11, denominator: 8, label: "11/8" },
  { numerator: 12, denominator: 8, label: "12/8" },
  { numerator: 13, denominator: 8, label: "13/8" },
  { numerator: 5, denominator: 16, label: "5/16" },
  { numerator: 7, denominator: 16, label: "7/16" },
  { numerator: 11, denominator: 16, label: "11/16" },
];
const STEM_ORDER: StemName[] = ["vocals", "drums", "bass", "other"];
const STEM_LABELS: Record<StemName, string> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  other: "Other",
};
const GM_DRUM_LABELS: Record<number, string> = {
  35: "Acoustic Kick",
  36: "Kick",
  37: "Side Stick",
  38: "Snare",
  39: "Clap",
  40: "Electric Snare",
  41: "Low Floor Tom",
  42: "Closed Hi-hat",
  43: "High Floor Tom",
  44: "Pedal Hi-hat",
  45: "Low Tom",
  46: "Open Hi-hat",
  47: "Low-Mid Tom",
  48: "High-Mid Tom",
  49: "Crash",
  50: "High Tom",
  51: "Ride",
  52: "China",
  53: "Ride Bell",
  54: "Tambourine",
  55: "Splash",
  56: "Cowbell",
  57: "Crash 2",
  59: "Ride 2",
};
const MIDI_CONVERTIBLE_STEMS: StemName[] = ["drums", "bass"];
const STEM_WAVEFORM_PEAK_COUNT = 3600;
const DRUM_SAMPLE_BASE_URL = "https://oramics.github.io/sampled/DM/CR-78/samples/";
const DRUM_SAMPLE_FILE_BY_MIDI: Record<number, string> = {
  24: "tamb-short.wav",
  25: "tamb-long.wav",
  26: "guiro-short.wav",
  27: "guiro-long.wav",
  28: "bongo-l.wav",
  29: "bongo-h.wav",
  30: "conga-l.wav",
  31: "rim.wav",
  32: "rim.wav",
  33: "rim.wav",
  34: "rim.wav",
  35: "kick.wav",
  36: "kick-accent.wav",
  37: "rim.wav",
  38: "snare.wav",
  39: "snare-accent.wav",
  40: "snare-accent.wav",
  41: "conga-l.wav",
  42: "hihat.wav",
  43: "bongo-l.wav",
  44: "hihat-accent.wav",
  45: "bongo-l.wav",
  46: "hihat-metal.wav",
  47: "bongo-h.wav",
  48: "bongo-h.wav",
  49: "cymbal.wav",
  50: "bongo-h.wav",
  51: "cymbal.wav",
  52: "cymbal.wav",
  53: "cymbal.wav",
  54: "tamb-short.wav",
  55: "cymbal.wav",
  56: "cowbell.wav",
  57: "cymbal.wav",
  59: "cymbal.wav",
};

function createInitialStemMidiConversions(): Record<StemName, StemMidiConversionState> {
  return {
    vocals: { isProcessing: false, progress: 0, status: "", result: null },
    drums: { isProcessing: false, progress: 0, status: "", result: null },
    bass: { isProcessing: false, progress: 0, status: "", result: null },
    other: { isProcessing: false, progress: 0, status: "", result: null },
  };
}

function isBlackKey(midi: number) {
  return [1, 3, 6, 8, 10].includes(midi % 12);
}

function calculateEditorPitchRange(notes: NoteEvent[]) {
  if (notes.length === 0) {
    return { min: MIN_PIANO_MIDI, max: MAX_PIANO_MIDI };
  }

  const pitches = notes.map((note) => note.midi);

  return {
    min: Math.max(MIN_PIANO_MIDI, Math.min(...pitches) - 12),
    max: Math.min(MAX_PIANO_MIDI, Math.max(...pitches) + 12),
  };
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";

  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);

  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatCountdown(seconds: number) {
  const safeSeconds = Math.max(0, Math.ceil(seconds));
  const m = Math.floor(safeSeconds / 60);
  const s = safeSeconds % 60;

  return `${m}:${s.toString().padStart(2, "0")}`;
}

function estimateStemSeparationSeconds(audioDurationSeconds: number | null) {
  if (!audioDurationSeconds || !Number.isFinite(audioDurationSeconds)) {
    return DEFAULT_STEM_ESTIMATE_SECONDS;
  }

  const savedFactor =
    typeof window !== "undefined"
      ? Number(window.localStorage.getItem(STEM_ESTIMATE_STORAGE_KEY))
      : NaN;
  const factor =
    Number.isFinite(savedFactor) && savedFactor > 0.1 && savedFactor < 6
      ? savedFactor
      : DEFAULT_STEM_SECONDS_PER_AUDIO_SECOND;

  return Math.min(
    MAX_STEM_ESTIMATE_SECONDS,
    Math.max(
      MIN_STEM_ESTIMATE_SECONDS,
      Math.round(audioDurationSeconds * factor + STEM_ESTIMATE_OVERHEAD_SECONDS)
    )
  );
}

function saveStemEstimateCalibration(
  audioDurationSeconds: number | null,
  elapsedSeconds: number
) {
  if (
    typeof window === "undefined" ||
    !audioDurationSeconds ||
    !Number.isFinite(audioDurationSeconds) ||
    audioDurationSeconds < 10 ||
    elapsedSeconds < 10
  ) {
    return;
  }

  const measuredFactor = elapsedSeconds / audioDurationSeconds;
  const previousFactor = Number(
    window.localStorage.getItem(STEM_ESTIMATE_STORAGE_KEY)
  );
  const nextFactor =
    Number.isFinite(previousFactor) && previousFactor > 0.1 && previousFactor < 6
      ? previousFactor * 0.65 + measuredFactor * 0.35
      : measuredFactor;

  window.localStorage.setItem(
    STEM_ESTIMATE_STORAGE_KEY,
    String(Math.min(6, Math.max(0.1, nextFactor)))
  );
}

function readAudioMetadataDuration(file: File) {
  return new Promise<number | null>((resolve) => {
    const audio = document.createElement("audio");
    const objectUrl = URL.createObjectURL(file);
    let didResolve = false;

    const cleanup = () => {
      URL.revokeObjectURL(objectUrl);
      audio.removeAttribute("src");
      audio.load();
    };

    const finish = (duration: number | null) => {
      if (didResolve) return;
      didResolve = true;
      cleanup();
      resolve(duration);
    };

    const timeoutId = window.setTimeout(() => finish(null), 2500);

    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      window.clearTimeout(timeoutId);
      finish(Number.isFinite(audio.duration) ? audio.duration : null);
    };
    audio.onerror = () => {
      window.clearTimeout(timeoutId);
      finish(null);
    };
    audio.src = objectUrl;
  });
}

function isPowerOfTwo(value: number) {
  return value > 0 && (value & (value - 1)) === 0;
}

function makeNoteId() {
  return `edited-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function getCurrentTimeMs() {
  return Date.now();
}

function createGrandPianoSampler() {
  const reverb = new Tone.Reverb({
    decay: 2.2,
    wet: 0.18,
  }).toDestination();

  const compressor = new Tone.Compressor({
    threshold: -18,
    ratio: 3,
    attack: 0.02,
    release: 0.25,
  }).connect(reverb);

  const sampler = new Tone.Sampler({
    urls: {
      A0: "A0.mp3",
      C1: "C1.mp3",
      "D#1": "Ds1.mp3",
      "F#1": "Fs1.mp3",
      A1: "A1.mp3",
      C2: "C2.mp3",
      "D#2": "Ds2.mp3",
      "F#2": "Fs2.mp3",
      A2: "A2.mp3",
      C3: "C3.mp3",
      "D#3": "Ds3.mp3",
      "F#3": "Fs3.mp3",
      A3: "A3.mp3",
      C4: "C4.mp3",
      "D#4": "Ds4.mp3",
      "F#4": "Fs4.mp3",
      A4: "A4.mp3",
      C5: "C5.mp3",
      "D#5": "Ds5.mp3",
      "F#5": "Fs5.mp3",
      A5: "A5.mp3",
      C6: "C6.mp3",
      "D#6": "Ds6.mp3",
      "F#6": "Fs6.mp3",
      A6: "A6.mp3",
      C7: "C7.mp3",
      "D#7": "Ds7.mp3",
      "F#7": "Fs7.mp3",
      A7: "A7.mp3",
      C8: "C8.mp3",
    },
    release: 1.2,
    baseUrl: "https://tonejs.github.io/audio/salamander/",
  }).connect(compressor);

  const instrument = sampler as PlaybackInstrument;
  instrument._extraDisposables = [compressor, reverb];

  return instrument;
}


function createSoundfontBassSampler(
  soundfontInstrument: string,
  filterFrequency: number,
  release: number
) {
  const filter = new Tone.Filter({
    frequency: filterFrequency,
    type: "lowpass",
    rolloff: -24,
  }).toDestination();

  const compressor = new Tone.Compressor({
    threshold: -22,
    ratio: 4,
    attack: 0.015,
    release: 0.22,
  }).connect(filter);

  const sampler = new Tone.Sampler({
    urls: {
      E1: "E1.mp3",
      A1: "A1.mp3",
      D2: "D2.mp3",
      G2: "G2.mp3",
      C3: "C3.mp3",
      F3: "F3.mp3",
      A3: "A3.mp3",
    },
    release,
    baseUrl: `https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/${soundfontInstrument}-mp3/`,
  }).connect(compressor);

  const instrument = sampler as PlaybackInstrument;
  instrument._extraDisposables = [compressor, filter];

  return instrument;
}

function getDrumSampleDuration(midi: number, noteDuration: number) {
  if (midi === 42 || midi === 44) return Math.min(Math.max(noteDuration, 0.06), 0.12);
  if (midi === 46) return Math.min(Math.max(noteDuration, 0.35), 0.75);
  if ([49, 51, 52, 53, 55, 57, 59].includes(midi)) {
    return Math.min(Math.max(noteDuration, 0.9), 1.6);
  }
  if ([41, 43, 45, 47, 48, 50].includes(midi)) {
    return Math.min(Math.max(noteDuration, 0.22), 0.5);
  }
  if (midi === 35 || midi === 36) return Math.min(Math.max(noteDuration, 0.18), 0.42);
  if (midi === 37 || midi === 39 || midi === 54 || midi === 56) {
    return Math.min(Math.max(noteDuration, 0.12), 0.35);
  }

  return Math.min(Math.max(noteDuration, 0.16), 0.42);
}

function createDrumKitInstrument(): PlaybackInstrument {
  const reverb = new Tone.Reverb({
    decay: 0.75,
    wet: 0.06,
  }).toDestination();

  const compressor = new Tone.Compressor({
    threshold: -14,
    ratio: 5,
    attack: 0.004,
    release: 0.12,
  }).connect(reverb);

  const sampler = new Tone.Sampler({
    urls: Object.fromEntries(
      Object.entries(DRUM_SAMPLE_FILE_BY_MIDI).map(([midi, fileName]) => [
        Tone.Frequency(Number(midi), "midi").toNote(),
        `${DRUM_SAMPLE_BASE_URL}${fileName}`,
      ])
    ),
    attack: 0,
    release: 0.04,
  }).connect(compressor);

  return {
    triggerAttackRelease(noteName, duration, time, velocity = 0.8) {
      const midi = Math.round(Tone.Frequency(noteName).toMidi());
      const vel = Math.min(Math.max(velocity, 0.08), 0.98);
      const when = time ?? Tone.now();
      const sampledMidi = DRUM_SAMPLE_FILE_BY_MIDI[midi] ? midi : 38;
      const sampledNoteName = Tone.Frequency(sampledMidi, "midi").toNote();

      sampler.triggerAttackRelease(
        sampledNoteName,
        getDrumSampleDuration(sampledMidi, duration),
        when,
        vel
      );
    },
    dispose() {
      sampler.dispose();
      compressor.dispose();
      reverb.dispose();
    },
  };
}


function createInstrument(type: InstrumentType) {
  if (type === "grand_piano") return createGrandPianoSampler();

  if (type === "drum_kit") return createDrumKitInstrument();

  if (type === "double_bass") {
    return createSoundfontBassSampler(
      "acoustic_bass",
      950,
      0.75
    );
  }

  if (type === "electric_bass") {
    return createSoundfontBassSampler(
      "electric_bass_pick",
      1800,
      0.38
    );
  }

  return createGrandPianoSampler();
}

function ModeIcon({ mode }: { mode: ModeType }) {
  if (mode === "piano") {
    return (
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <rect x="7" y="8" width="34" height="32" rx="4" />
        <path d="M14 8v23m7-23v23m7-23v23m7-23v23M7 31h34" />
      </svg>
    );
  }

  if (mode === "bass") {
    return (
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <path d="M30 7 18 38m8-30-5 32M16 15h16M13 24h18M10 33h20" />
        <circle cx="31" cy="10" r="4" />
      </svg>
    );
  }

  if (mode === "drums") {
    return (
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <circle cx="17" cy="27" r="8" />
        <circle cx="31" cy="27" r="8" />
        <path d="M12 14h24M16 11l4 8m12-8-4 8M11 37h26" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path d="M10 14h28v20H10zM16 14V9m16 5V9M16 34v5m16-5v5" />
      <path d="M16 20h16M16 26h10" />
    </svg>
  );
}

function MusicUploadIcon() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <path d="M20 47V14l28-6v32" />
      <path d="M20 23l28-6" />
      <circle cx="16" cy="48" r="7" />
      <circle cx="44" cy="41" r="7" />
      <path d="M29 33h6m-3-3v6" />
    </svg>
  );
}

function getFileNameFromUrl(url: string) {
  const lastSegment = url.split("/").filter(Boolean).at(-1);
  return lastSegment ? decodeURIComponent(lastSegment) : "stem.wav";
}

function buildWaveformPeaks(audioBuffer: AudioBuffer, peakCount = STEM_WAVEFORM_PEAK_COUNT) {
  const channelCount = audioBuffer.numberOfChannels;
  const actualPeakCount = Math.max(1, Math.min(peakCount, audioBuffer.length));
  const samplesPerPeak = Math.max(1, Math.ceil(audioBuffer.length / actualPeakCount));
  const peaks = Array.from({ length: actualPeakCount }, (_, peakIndex) => {
    const start = peakIndex * samplesPerPeak;
    const end = Math.min(audioBuffer.length, start + samplesPerPeak);
    let min = 0;
    let max = 0;

    for (let channel = 0; channel < channelCount; channel += 1) {
      const data = audioBuffer.getChannelData(channel);
      for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
        const value = data[sampleIndex] ?? 0;
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }

    return { min, max };
  });
  const strongestPeak = Math.max(
    0.001,
    ...peaks.flatMap((peak) => [Math.abs(peak.min), Math.abs(peak.max)])
  );

  return peaks.map((peak) => ({
    min: Math.max(-1, Math.min(0, peak.min / strongestPeak)),
    max: Math.min(1, Math.max(0, peak.max / strongestPeak)),
  }));
}

function estimateTempoFromAudioBuffers(audioBuffers: AudioBuffer[]) {
  const referenceBuffer =
    audioBuffers.find((buffer) => buffer.duration > 8) ?? audioBuffers[0];
  if (!referenceBuffer) return null;

  const sampleRate = referenceBuffer.sampleRate;
  const channelData = referenceBuffer.getChannelData(0);
  const frameSize = 1024;
  const hopSeconds = frameSize / sampleRate;
  const frameCount = Math.floor(channelData.length / frameSize);
  if (frameCount < 64) return null;

  const energy = Array.from({ length: frameCount }, (_, frameIndex) => {
    const start = frameIndex * frameSize;
    const end = Math.min(channelData.length, start + frameSize);
    let sum = 0;

    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      const value = channelData[sampleIndex] ?? 0;
      sum += value * value;
    }

    return Math.sqrt(sum / Math.max(1, end - start));
  });

  const envelope = energy.map((value, index) =>
    Math.max(0, value - (energy[index - 1] ?? 0))
  );
  const minBpm = 70;
  const maxBpm = 180;
  let bestBpm = 120;
  let bestScore = 0;

  for (let bpm = minBpm; bpm <= maxBpm; bpm += 1) {
    const lag = Math.round((60 / bpm) / hopSeconds);
    if (lag <= 0 || lag >= envelope.length) continue;

    let score = 0;
    for (let index = lag; index < envelope.length; index += 1) {
      score += envelope[index] * envelope[index - lag];
    }

    if (score > bestScore) {
      bestScore = score;
      bestBpm = bpm;
    }
  }

  return bestScore > 0 ? bestBpm : null;
}

function buildStemBarMarkers(duration: number, tempoBpmValue: number | null) {
  if (!duration || !Number.isFinite(duration) || duration <= 0) return [];

  const secondsPerStemBar = tempoBpmValue ? (60 / tempoBpmValue) * 4 : duration / 4;
  const safeSecondsPerBar = Math.max(0.5, secondsPerStemBar);
  const markerCount = Math.min(256, Math.ceil(duration / safeSecondsPerBar) + 1);

  return Array.from({ length: markerCount }, (_, index) => {
    const time = index * safeSecondsPerBar;
    return {
      bar: index + 1,
      left: Math.min(100, (time / duration) * 100),
      time,
    };
  }).filter((marker) => marker.time <= duration);
}

function createFallbackWaveform(peakCount = STEM_WAVEFORM_PEAK_COUNT) {
  return Array.from({ length: peakCount }, () => ({ min: -0.04, max: 0.04 }));
}

const UI_COPY = {
  en: {
    studio: "Studio",
    brandCategory: "MIDI PERFORMANCE STUDIO",
    heroTitle: "Shape every note.",
    heroCopy: "Turn audio and MIDI into an expressive, performance-ready arrangement.",
    chooseWorkspace: "Choose a workspace to begin.",
    individualMidi: "Individual track MIDI conversion",
    individualMidiCopy: "Upload an isolated instrument track and edit the generated MIDI.",
    tagline: "Edit with intent. Export with confidence.",
    switchMode: "Switch mode",
    startSession: "Start a session",
    startSessionCopy: "Begin from a recording or open an existing MIDI performance.",
    createAudio: "Create from audio",
    createAudioCopy: "Convert a recording into an editable MIDI arrangement.",
    openMidi: "Open MIDI",
    openMidiCopy: "Bring in a MIDI file and refine every performance detail.",
    uploadAudio: "Upload source audio",
    uploadMusic: "Upload music file",
    uploadMusicCopy: "Upload one song and receive four playable stems.",
    uploadingStemAudio: "Uploading music file...",
    preparingStemSession: "Preparing your stem session",
    separatingStems: "Splitting your song into tracks. You can keep this window open.",
    stemsReady: "4 stems are ready.",
    timeRemaining: "Time remaining",
    finishingStems: "Finishing up",
    stemConnectionError: "Connection was interrupted. Keep the backend running and try again.",
    stemFailed: "We could not finish this separation. Please try another audio file or check the backend.",
    openStemMixer: "Opening stem mixer...",
    stemMixer: "Stem mixer",
    stemMixerCopy: "Your separated tracks are loaded into a compact DAW-style mixer.",
    analyzingTempo: "Analyzing tempo...",
    convertStemToMidi: "Convert to MIDI",
    convertingStemToMidi: "Creating MIDI from this track...",
    stemMidiReady: "MIDI is ready.",
    openMidiEditor: "Open MIDI editor",
    playAllStems: "Play all",
    pauseStems: "Pause",
    stopStems: "Stop",
    mute: "Mute",
    solo: "Solo",
    imported: "Imported",
    backToHome: "Back to start",
    stemDownload: "Download",
    stemPlayer: "Stem playback",
    stemFile: "Source",
    otherMode: "Other",
    openMidiFile: "Open MIDI file",
    processing: "Processing",
    processed: "processed",
    loadingGeneratedMidi: "Loading generated MIDI...",
    complete: "Complete.",
    uploadingAudio: "Uploading audio...",
    queued: "Queued.",
    backendError: "Error. Check backend terminal.",
    drumUploadStatus: "Creating drum MIDI from the uploaded drum track...",
    basicPitchMissing: "The backend cannot find Basic Pitch. This is a backend environment issue.",
    wrongDrumPipeline: "This drum upload reached the Basic Pitch pipeline. Hard-refresh the page and try Drums again.",
    exported: "Edited MIDI exported.",
    events: "Events",
    duration: "Duration",
    tempo: "Tempo",
    preview: "Preview & transport",
    previewCopy: "Listen, locate, and prepare your performance before editing.",
    resume: "Resume",
    play: "Play",
    pause: "Pause",
    stop: "Stop",
    sound: "Sound",
    grandPiano: "Grand Piano",
    doubleBass: "Double Bass",
    electricBass: "Electric Bass",
    drumKit: "Sampled Drum Kit",
    loadingSound: "Loading sound",
    ready: "Ready",
    readyOnPlay: "Ready on play",
    editor: "MIDI editor",
    editorCopy: "Shape timing, pitch, length, and dynamics directly in the piano roll.",
    selected: "selected",
    noSelection: "No selection",
    meter: "Meter",
    timeZoom: "Time zoom",
    waveformZoom: "Waveform zoom",
    pitchZoom: "Pitch zoom",
    snapQuantize: "Snap / Quantize",
    velocity: "Velocity",
    quantize: "Quantize selected",
    nudgeBack: "Nudge ←",
    nudgeForward: "Nudge →",
    duplicate: "Duplicate",
    copy: "Copy",
    paste: "Paste",
    selectAll: "Select all",
    clear: "Clear",
    octaveDown: "−1 octave",
    octaveUp: "+1 octave",
    delete: "Delete",
    undo: "Undo",
    redo: "Redo",
    exportMidi: "Export MIDI",
    pianoRoll: "Piano roll editor",
    help: "Drag empty piano-roll space to marquee-select notes · Click empty space to play from that time · Drag a note to move it · Drag its right edge to resize · Snap / Quantize sets edit resolution · Delete removes selected notes · Cmd/Ctrl+A selects all · Cmd/Ctrl+Z undoes · Arrow keys nudge or transpose · Cmd/Ctrl+C copies · Cmd/Ctrl+V pastes at the playhead",
  },
  ko: {
    studio: "스튜디오",
    brandCategory: "MIDI 퍼포먼스 스튜디오",
    heroTitle: "모든 노트를 완성하세요.",
    heroCopy: "오디오와 MIDI를 표현력 있는 완성형 퍼포먼스로 다듬어 보세요.",
    chooseWorkspace: "워크스페이스를 선택하여 시작하세요.",
    individualMidi: "개별 트랙 MIDI 변환",
    individualMidiCopy: "악기별로 분리된 오디오를 업로드하고 생성된 MIDI를 편집하세요.",
    tagline: "의도대로 편집하고, 자신 있게 내보내세요.",
    switchMode: "모드 변경",
    startSession: "세션 시작",
    startSessionCopy: "녹음 파일에서 시작하거나 기존 MIDI 퍼포먼스를 불러오세요.",
    createAudio: "오디오에서 만들기",
    createAudioCopy: "녹음 파일을 편집 가능한 MIDI 편곡으로 변환합니다.",
    openMidi: "MIDI 열기",
    openMidiCopy: "MIDI 파일을 불러와 퍼포먼스의 모든 디테일을 다듬으세요.",
    uploadAudio: "소스 오디오 업로드",
    uploadMusic: "음악 파일 업로드",
    uploadMusicCopy: "음악 파일 하나를 업로드하면 4개 stem을 재생하고 다운로드할 수 있습니다.",
    uploadingStemAudio: "음악 파일 업로드 중...",
    preparingStemSession: "Stem 세션을 준비하고 있습니다",
    separatingStems: "음악을 트랙별로 나누는 중입니다. 이 창을 열어 두세요.",
    stemsReady: "4개 stem 준비가 완료되었습니다.",
    timeRemaining: "남은 시간",
    finishingStems: "마무리 중",
    stemConnectionError: "백엔드 연결이 끊겼습니다. 백엔드를 실행한 상태에서 다시 시도하세요.",
    stemFailed: "분리 작업을 완료하지 못했습니다. 다른 오디오 파일로 다시 시도하거나 백엔드를 확인하세요.",
    openStemMixer: "Stem 믹서를 여는 중...",
    stemMixer: "Stem 믹서",
    stemMixerCopy: "분리된 트랙을 DAW 스타일 믹서로 불러왔습니다.",
    analyzingTempo: "템포 분석 중...",
    convertStemToMidi: "MIDI로 변환",
    convertingStemToMidi: "이 트랙에서 MIDI를 생성하는 중...",
    stemMidiReady: "MIDI 준비가 완료되었습니다.",
    openMidiEditor: "MIDI 편집기 열기",
    playAllStems: "전체 재생",
    pauseStems: "일시정지",
    stopStems: "정지",
    mute: "뮤트",
    solo: "솔로",
    imported: "임포트됨",
    backToHome: "처음으로",
    stemDownload: "다운로드",
    stemPlayer: "Stem 재생",
    stemFile: "원본 파일",
    otherMode: "Other",
    openMidiFile: "MIDI 파일 열기",
    processing: "처리 중",
    processed: "처리됨",
    loadingGeneratedMidi: "생성된 MIDI를 불러오는 중...",
    complete: "완료되었습니다.",
    uploadingAudio: "오디오 업로드 중...",
    queued: "대기열에 등록되었습니다.",
    backendError: "오류가 발생했습니다. 백엔드 터미널을 확인하세요.",
    drumUploadStatus: "업로드한 드럼 트랙에서 MIDI를 생성하는 중입니다...",
    basicPitchMissing: "백엔드에서 Basic Pitch 실행 파일을 찾지 못했습니다. 백엔드 환경 문제입니다.",
    wrongDrumPipeline: "드럼 업로드가 Basic Pitch 파이프라인으로 들어갔습니다. 페이지를 강력 새로고침한 뒤 Drums로 다시 시도하세요.",
    exported: "편집한 MIDI를 내보냈습니다.",
    events: "노트",
    duration: "길이",
    tempo: "템포",
    preview: "미리듣기 및 재생",
    previewCopy: "편집하기 전에 퍼포먼스를 듣고 재생 위치를 정하세요.",
    resume: "계속 재생",
    play: "재생",
    pause: "일시정지",
    stop: "정지",
    sound: "사운드",
    grandPiano: "그랜드 피아노",
    doubleBass: "더블 베이스",
    electricBass: "일렉트릭 베이스",
    drumKit: "샘플 드럼 키트",
    loadingSound: "사운드 불러오는 중",
    ready: "준비됨",
    readyOnPlay: "재생 시 준비",
    editor: "MIDI 편집기",
    editorCopy: "피아노롤에서 타이밍, 음정, 길이, 다이내믹을 직접 조정하세요.",
    selected: "개 선택됨",
    noSelection: "선택 없음",
    meter: "박자표",
    timeZoom: "시간 확대/축소",
    waveformZoom: "파형 확대/축소",
    pitchZoom: "음정 확대/축소",
    snapQuantize: "스냅 / 퀀타이즈",
    velocity: "벨로시티",
    quantize: "선택 노트 퀀타이즈",
    nudgeBack: "앞으로 이동 ←",
    nudgeForward: "뒤로 이동 →",
    duplicate: "복제",
    copy: "복사",
    paste: "붙여넣기",
    selectAll: "모두 선택",
    clear: "선택 해제",
    octaveDown: "−1 옥타브",
    octaveUp: "+1 옥타브",
    delete: "삭제",
    undo: "실행 취소",
    redo: "다시 실행",
    exportMidi: "MIDI 내보내기",
    pianoRoll: "피아노롤 편집기",
    help: "빈 피아노롤 영역을 드래그해 여러 노트를 선택하세요 · 빈 영역을 클릭하면 해당 위치부터 재생합니다 · 노트를 드래그해 이동하고 오른쪽 끝을 드래그해 길이를 조절하세요 · 스냅 / 퀀타이즈로 편집 단위를 설정합니다 · Delete: 삭제 · Cmd/Ctrl+A: 모두 선택 · Cmd/Ctrl+Z: 실행 취소 · 화살표 키: 이동 또는 음정 변경 · Cmd/Ctrl+C: 복사 · Cmd/Ctrl+V: 재생 헤드 위치에 붙여넣기",
  },
} as const;

export default function Home() {
  const [mode, setMode] = useState<ModeType>("piano");
  const [language, setLanguage] = useState<Language>("en");
  const [hasSelectedMode, setHasSelectedMode] = useState(false);
  const [audioFileName, setAudioFileName] = useState("");
  const [midiFileName, setMidiFileName] = useState("");
  const [notes, setNotes] = useState<NoteEvent[]>([]);
  const [pitchRange, setPitchRange] = useState(() =>
    calculateEditorPitchRange([])
  );
  const [tempoBpm, setTempoBpm] = useState(120);
  const [timeSignature, setTimeSignature] = useState<TimeSignature>({
    numerator: 4,
    denominator: 4,
  });
  const [horizontalZoom, setHorizontalZoom] = useState(1);
  const [verticalZoom, setVerticalZoom] = useState(1);
  const [stemHorizontalZoom, setStemHorizontalZoom] = useState(1);
  const [quantizeDivision, setQuantizeDivision] = useState(0.25);
  const [exportStatus, setExportStatus] = useState("");
  const [marqueeBox, setMarqueeBox] = useState<MarqueeBox | null>(null);
  const [clipboardNoteCount, setClipboardNoteCount] = useState(0);
  const [selectedNoteIds, setSelectedNoteIds] = useState<Set<string>>(
    () => new Set()
  );
  const [selectionAnchorId, setSelectionAnchorId] = useState<string | null>(null);
  const [pastNotes, setPastNotes] = useState<NoteEvent[][]>([]);
  const [futureNotes, setFutureNotes] = useState<NoteEvent[][]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("");
  const [processingProgress, setProcessingProgress] = useState(0);
  const [stemFileName, setStemFileName] = useState("");
  const [stemResult, setStemResult] = useState<StemJobResult | null>(null);
  const [isSeparatingStems, setIsSeparatingStems] = useState(false);
  const [stemStatus, setStemStatus] = useState("");
  const [stemProgress, setStemProgress] = useState(0);
  const [stemEstimateSeconds, setStemEstimateSeconds] = useState(
    DEFAULT_STEM_ESTIMATE_SECONDS
  );
  const [stemElapsedSeconds, setStemElapsedSeconds] = useState(0);
  const [isStemMixerOpen, setIsStemMixerOpen] = useState(false);
  const [isStemMixPlaying, setIsStemMixPlaying] = useState(false);
  const [stemMixTime, setStemMixTime] = useState(0);
  const [mutedStems, setMutedStems] = useState<StemTrackState>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });
  const [soloedStems, setSoloedStems] = useState<StemTrackState>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });
  const [stemTrackDurations, setStemTrackDurations] = useState<
    Partial<Record<StemName, number>>
  >({});
  const [stemWaveforms, setStemWaveforms] = useState<StemWaveformPeaks>({});
  const [stemTempoBpm, setStemTempoBpm] = useState<number | null>(null);
  const [stemMidiConversions, setStemMidiConversions] = useState<
    Record<StemName, StemMidiConversionState>
  >(() => createInitialStemMidiConversions());

  const [instrumentType, setInstrumentType] =
    useState<InstrumentType>("grand_piano");
  const [isInstrumentLoading, setIsInstrumentLoading] = useState(false);
  const [isInstrumentReady, setIsInstrumentReady] = useState(false);
  const copy = UI_COPY[language];

  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const instrumentRef = useRef<PlaybackInstrument | null>(null);
  const progressTimerRef = useRef<number | null>(null);
  const noteDragRef = useRef<NoteDragState | null>(null);
  const marqueeSelectionRef = useRef<MarqueeSelectionState | null>(null);
  const noteClipboardRef = useRef<NoteEvent[]>([]);
  const suppressNoteClickRef = useRef(false);
  const suppressPianoRollClickRef = useRef(false);
  const stemAudioContextRef = useRef<AudioContext | null>(null);
  const stemAudioBuffersRef = useRef<Partial<Record<StemName, AudioBuffer>>>({});
  const stemAudioSourcesRef = useRef<Partial<Record<StemName, AudioBufferSourceNode>>>({});
  const stemGainNodesRef = useRef<Partial<Record<StemName, GainNode>>>({});
  const stemMixTimerRef = useRef<number | null>(null);
  const stemMixStartedAtRef = useRef(0);
  const stemMixStartOffsetRef = useRef(0);
  const stemJobStartedAtRef = useRef<number | null>(null);
  const stemAudioDurationRef = useRef<number | null>(null);
  const stemArrangeScrollContainersRef = useRef<Set<HTMLElement>>(new Set());
  const isSyncingStemArrangeScrollRef = useRef(false);

  const totalDuration = useMemo(() => {
    if (notes.length === 0) return 0;

    return Math.max(...notes.map((note) => note.time + note.duration));
  }, [notes]);

  const displayedStemProgress = useMemo(() => {
    if (!isSeparatingStems) return stemProgress;

    const timedProgress = Math.min(
      92,
      Math.round((stemElapsedSeconds / Math.max(stemEstimateSeconds, 1)) * 92)
    );

    return Math.max(stemProgress, timedProgress);
  }, [isSeparatingStems, stemElapsedSeconds, stemEstimateSeconds, stemProgress]);

  const stemRemainingSeconds = Math.max(
    0,
    stemEstimateSeconds - stemElapsedSeconds
  );
  const isPastStemEstimate =
    isSeparatingStems && stemElapsedSeconds >= stemEstimateSeconds;
  const stemCountdownLabel = isPastStemEstimate
    ? formatCountdown(stemElapsedSeconds - stemEstimateSeconds)
    : formatCountdown(stemRemainingSeconds);
  const hasSoloedStems = STEM_ORDER.some((stemName) => soloedStems[stemName]);
  const importedStemNames = useMemo(
    () => STEM_ORDER.filter((stemName) => Boolean(stemResult?.stems[stemName])),
    [stemResult]
  );
  const stemMixerDuration = Math.max(
    0,
    ...Object.values(stemTrackDurations).filter(
      (duration) => Number.isFinite(duration) && duration > 0
    )
  );
  const stemMixerProgress = stemMixerDuration
    ? Math.min(100, Math.max(0, (stemMixTime / stemMixerDuration) * 100))
    : 0;
  const stemArrangeWidth = `${Math.round(stemHorizontalZoom * 100)}%`;
  const stemBarMarkers = useMemo(
    () => buildStemBarMarkers(stemMixerDuration, stemTempoBpm),
    [stemMixerDuration, stemTempoBpm]
  );

  const selectedNotes = useMemo(
    () => notes.filter((note) => selectedNoteIds.has(note.id)),
    [notes, selectedNoteIds]
  );
  const selectedVelocity = useMemo(() => {
    if (selectedNotes.length === 0) return 96;

    return Math.round(
      (selectedNotes.reduce((sum, note) => sum + note.velocity, 0) /
        selectedNotes.length) *
        127
    );
  }, [selectedNotes]);

  const safeTempoBpm = Math.min(300, Math.max(20, tempoBpm || 120));
  const secondsPerQuarterBeat = 60 / safeTempoBpm;
  const secondsPerGridBeat =
    secondsPerQuarterBeat * (4 / timeSignature.denominator);
  const secondsPerBar = secondsPerGridBeat * timeSignature.numerator;
  const pixelsPerBeat = BASE_PIXELS_PER_BEAT * horizontalZoom;
  const noteRowHeight = BASE_NOTE_ROW_HEIGHT * verticalZoom;
  const noteBarHeight = Math.max(
    5,
    Math.min(Math.max(4, noteRowHeight - 4), noteRowHeight * 0.58)
  );
  const noteVerticalInset = Math.max(2, (noteRowHeight - noteBarHeight) / 2);
  const snapSeconds = secondsPerQuarterBeat * quantizeDivision;
  const pianoRollWidth = Math.max(
    900,
    ((totalDuration + secondsPerBar) / secondsPerQuarterBeat) * pixelsPerBeat
  );
  const pianoRollHeight = Math.max(
    360,
    (pitchRange.max - pitchRange.min + 1) * noteRowHeight
  );
  const visiblePitches = Array.from(
    { length: pitchRange.max - pitchRange.min + 1 },
    (_, index) => pitchRange.max - index
  );
  const barMarkers = Array.from(
    {
      length: Math.max(
        1,
        Math.ceil(Math.max(totalDuration, secondsPerBar) / secondsPerBar)
      ),
    },
    (_, index) => ({
      bar: index + 1,
      startSeconds: index * secondsPerBar,
    })
  );
  const secondsToX = (seconds: number) =>
    (seconds / secondsPerQuarterBeat) * pixelsPerBeat;
  const pitchToY = (pitch: number) => (pitchRange.max - pitch) * noteRowHeight;

  function disposeInstrument() {
    if (instrumentRef.current) {
      const extras = instrumentRef.current._extraDisposables;
      extras?.forEach((node) => node.dispose());
      instrumentRef.current.dispose();
      instrumentRef.current = null;
    }
  }

  async function preloadInstrument(type: InstrumentType) {
    setIsInstrumentLoading(true);
    setIsInstrumentReady(false);

    disposeInstrument();

    const instrument = createInstrument(type);
    instrumentRef.current = instrument;

    await Tone.loaded();

    setIsInstrumentLoading(false);
    setIsInstrumentReady(true);
  }

  function clearProgressTimer() {
    if (progressTimerRef.current !== null) {
      window.clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  }

  function resetMidiWorkspaceSession() {
    stopMidi();
    setAudioFileName("");
    setMidiFileName("");
    setNotes([]);
    setPitchRange(calculateEditorPitchRange([]));
    setTempoBpm(120);
    setTimeSignature({ numerator: 4, denominator: 4 });
    setCurrentTime(0);
    setSelectedNoteIds(new Set());
    setSelectionAnchorId(null);
    setPastNotes([]);
    setFutureNotes([]);
    setMarqueeBox(null);
    noteClipboardRef.current = [];
    setClipboardNoteCount(0);
    setExportStatus("");
    setIsProcessing(false);
    setProcessingStatus("");
    setProcessingProgress(0);
  }

  function startProgressTimer() {
    clearProgressTimer();

    progressTimerRef.current = window.setInterval(() => {
      setCurrentTime(Tone.Transport.seconds);
    }, 100);
  }

  async function loadMidiArrayBuffer(arrayBuffer: ArrayBuffer, fileName: string) {
    const midi = new Midi(arrayBuffer);

    const detectedTempo = midi.header.tempos[0]?.bpm;
    const detectedTimeSignature = midi.header.timeSignatures[0]?.timeSignature;

    setTempoBpm(
      Number.isFinite(detectedTempo) && detectedTempo && detectedTempo > 0
        ? Math.round(detectedTempo * 10) / 10
        : 120
    );
    setTimeSignature({
      numerator:
        detectedTimeSignature && detectedTimeSignature[0] > 0
          ? detectedTimeSignature[0]
          : 4,
      denominator:
        detectedTimeSignature && detectedTimeSignature[1] > 0
          ? detectedTimeSignature[1]
          : 4,
    });

    const extractedNotes: NoteEvent[] = [];

    midi.tracks.forEach((track, trackIndex) => {
      track.notes.forEach((note, noteIndex) => {
        extractedNotes.push({
          id: `track-${trackIndex}-note-${noteIndex}`,
          name: note.name,
          midi: note.midi,
          time: note.time,
          duration: note.duration,
          velocity: note.velocity,
        });
      });
    });

    extractedNotes.sort((a, b) => a.time - b.time);

    setMidiFileName(fileName);
    setNotes(extractedNotes);
    setPitchRange(calculateEditorPitchRange(extractedNotes));
    setSelectedNoteIds(new Set());
    setSelectionAnchorId(null);
    setPastNotes([]);
    setFutureNotes([]);
    setCurrentTime(0);
  }

  function commitNotes(nextNotes: NoteEvent[]) {
    if (nextNotes === notes) return;

    stopMidi();
    setPastNotes((history) => [...history.slice(-49), notes]);
    setFutureNotes([]);
    setNotes(nextNotes);
  }

  function clearSelection() {
    setSelectedNoteIds(new Set());
    setSelectionAnchorId(null);
  }

  function selectAllNotes() {
    setSelectedNoteIds(new Set(notes.map((note) => note.id)));
    setSelectionAnchorId(notes[0]?.id ?? null);
  }

  function handleNoteSelect(
    event: React.MouseEvent<HTMLElement>,
    noteId: string,
    noteIndex: number
  ) {
    event.stopPropagation();

    if (suppressNoteClickRef.current) {
      suppressNoteClickRef.current = false;
      return;
    }

    if (event.shiftKey && selectionAnchorId) {
      const anchorIndex = notes.findIndex((note) => note.id === selectionAnchorId);

      if (anchorIndex >= 0) {
        const start = Math.min(anchorIndex, noteIndex);
        const end = Math.max(anchorIndex, noteIndex);
        const rangeIds = notes.slice(start, end + 1).map((note) => note.id);

        setSelectedNoteIds((current) => {
          const next = new Set(event.metaKey || event.ctrlKey ? current : []);
          rangeIds.forEach((id) => next.add(id));
          return next;
        });
        return;
      }
    }

    if (event.metaKey || event.ctrlKey) {
      setSelectedNoteIds((current) => {
        const next = new Set(current);

        if (next.has(noteId)) {
          next.delete(noteId);
        } else {
          next.add(noteId);
        }

        return next;
      });
      setSelectionAnchorId(noteId);
      return;
    }

    setSelectedNoteIds(new Set([noteId]));
    setSelectionAnchorId(noteId);
  }

  function getEditorContentPoint(
    event: React.PointerEvent<HTMLElement>,
    editor: HTMLElement
  ) {
    const rect = editor.getBoundingClientRect();

    return {
      x: event.clientX - rect.left + editor.scrollLeft,
      y: event.clientY - rect.top + editor.scrollTop,
    };
  }

  function toMarqueeBox(
    startX: number,
    startY: number,
    currentX: number,
    currentY: number
  ): MarqueeBox {
    return {
      left: Math.min(startX, currentX),
      top: Math.min(startY, currentY),
      width: Math.abs(currentX - startX),
      height: Math.abs(currentY - startY),
    };
  }

  function beginMarqueeSelection(event: React.PointerEvent<HTMLElement>) {
    if (event.button !== 0) return;

    const target = event.target as HTMLElement;
    if (target.closest("[data-piano-note]")) return;

    const editor = event.currentTarget;
    const point = getEditorContentPoint(event, editor);

    if (point.x < PIANO_KEY_WIDTH || point.y < RULER_HEIGHT) return;

    event.preventDefault();
    editor.setPointerCapture(event.pointerId);

    marqueeSelectionRef.current = {
      pointerId: event.pointerId,
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
      initialSelectedNoteIds: new Set(selectedNoteIds),
      appendToSelection: event.metaKey || event.ctrlKey,
      didDrag: false,
    };
    setMarqueeBox(null);
  }

  function handleMarqueeSelectionMove(event: React.PointerEvent<HTMLElement>) {
    const marquee = marqueeSelectionRef.current;
    if (!marquee || marquee.pointerId !== event.pointerId) return;

    event.preventDefault();

    const point = getEditorContentPoint(event, event.currentTarget);
    marquee.currentX = point.x;
    marquee.currentY = point.y;

    const box = toMarqueeBox(
      marquee.startX,
      marquee.startY,
      marquee.currentX,
      marquee.currentY
    );

    if (!marquee.didDrag && box.width < 4 && box.height < 4) return;

    marquee.didDrag = true;
    setMarqueeBox(box);

    const intersectedNoteIds = notes
      .filter((note) => {
        const noteLeft = PIANO_KEY_WIDTH + secondsToX(note.time);
        const noteTop =
          RULER_HEIGHT +
          pitchToY(note.midi) +
          noteVerticalInset;
        const noteWidth = Math.max(
          (note.duration / secondsPerQuarterBeat) * pixelsPerBeat,
          8
        );

        return (
          noteLeft < box.left + box.width &&
          noteLeft + noteWidth > box.left &&
          noteTop < box.top + box.height &&
          noteTop + noteBarHeight > box.top
        );
      })
      .map((note) => note.id);

    const nextSelectedNoteIds = marquee.appendToSelection
      ? new Set([...marquee.initialSelectedNoteIds, ...intersectedNoteIds])
      : new Set(intersectedNoteIds);

    setSelectedNoteIds(nextSelectedNoteIds);
    setSelectionAnchorId(intersectedNoteIds[0] ?? null);
  }

  function finishMarqueeSelection(event: React.PointerEvent<HTMLElement>) {
    const marquee = marqueeSelectionRef.current;
    if (!marquee || marquee.pointerId !== event.pointerId) return;

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (marquee.didDrag) {
      suppressPianoRollClickRef.current = true;
    }

    marqueeSelectionRef.current = null;
    setMarqueeBox(null);
  }

  function cancelMarqueeSelection(event: React.PointerEvent<HTMLElement>) {
    const marquee = marqueeSelectionRef.current;
    if (!marquee || marquee.pointerId !== event.pointerId) return;

    setSelectedNoteIds(marquee.initialSelectedNoteIds);
    setSelectionAnchorId(null);
    marqueeSelectionRef.current = null;
    setMarqueeBox(null);
  }

  function beginNotePointerEdit(
    event: React.PointerEvent<HTMLElement>,
    noteId: string,
    mode: NoteDragMode
  ) {
    if (event.button !== 0) return;

    event.preventDefault();
    event.stopPropagation();

    const noteIsSelected = selectedNoteIds.has(noteId);
    const hasSelectionModifier = event.metaKey || event.ctrlKey || event.shiftKey;
    const targetNoteIds = noteIsSelected
      ? new Set(selectedNoteIds)
      : new Set([noteId]);

    if (!noteIsSelected && !hasSelectionModifier) {
      setSelectedNoteIds(new Set([noteId]));
      setSelectionAnchorId(noteId);
    }

    stopMidi();
    event.currentTarget.setPointerCapture(event.pointerId);

    noteDragRef.current = {
      mode,
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      targetNoteIds,
      originalNotes: notes,
      latestNotes: notes,
      didChange: false,
    };
  }

  function handleNotePointerMove(event: React.PointerEvent<HTMLElement>) {
    const drag = noteDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;

    event.preventDefault();

    const deltaX = event.clientX - drag.startClientX;
    const deltaY = event.clientY - drag.startClientY;
    const movedFarEnough =
      drag.mode === "move"
        ? Math.abs(deltaX) >= 2 || Math.abs(deltaY) >= 2
        : Math.abs(deltaX) >= 2;

    if (!movedFarEnough) return;

    let nextNotes: NoteEvent[];

    if (drag.mode === "move") {
      const selectedOriginalNotes = drag.originalNotes.filter((note) =>
        drag.targetNoteIds.has(note.id)
      );
      const minimumTime = Math.min(
        ...selectedOriginalNotes.map((note) => note.time)
      );
      const minimumPitch = Math.min(
        ...selectedOriginalNotes.map((note) => note.midi)
      );
      const maximumPitch = Math.max(
        ...selectedOriginalNotes.map((note) => note.midi)
      );
      const rawTimeDelta = (deltaX / pixelsPerBeat) * secondsPerQuarterBeat;
      const snappedTimeDelta =
        Math.round(rawTimeDelta / snapSeconds) * snapSeconds;
      const timeDelta = Math.max(-minimumTime, snappedTimeDelta);
      const rawPitchDelta = -Math.round(deltaY / noteRowHeight);
      const pitchDelta = Math.min(
        pitchRange.max - maximumPitch,
        Math.max(pitchRange.min - minimumPitch, rawPitchDelta)
      );

      nextNotes = drag.originalNotes.map((note) => {
        if (!drag.targetNoteIds.has(note.id)) return note;

        const midi = note.midi + pitchDelta;

        return {
          ...note,
          time: Math.max(0, note.time + timeDelta),
          midi,
          name: Tone.Frequency(midi, "midi").toNote(),
        };
      });
    } else {
      const durationDelta = (deltaX / pixelsPerBeat) * secondsPerQuarterBeat;

      nextNotes = drag.originalNotes.map((note) => {
        if (!drag.targetNoteIds.has(note.id)) return note;

        const unsnappedDuration = Math.max(
          MIN_NOTE_DURATION,
          note.duration + durationDelta
        );
        const duration = Math.max(
          MIN_NOTE_DURATION,
          Math.round(unsnappedDuration / snapSeconds) * snapSeconds
        );

        return { ...note, duration };
      });
    }

    drag.didChange = true;
    drag.latestNotes = nextNotes;
    setNotes(nextNotes);
  }

  function finishNotePointerEdit(event: React.PointerEvent<HTMLElement>) {
    const drag = noteDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (drag.didChange) {
      setNotes(drag.latestNotes);
      setPastNotes((history) => [
        ...history.slice(-49),
        drag.originalNotes,
      ]);
      setFutureNotes([]);
      suppressNoteClickRef.current = true;
    }

    noteDragRef.current = null;
  }

  function cancelNotePointerEdit(event: React.PointerEvent<HTMLElement>) {
    const drag = noteDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;

    setNotes(drag.originalNotes);
    noteDragRef.current = null;
    suppressNoteClickRef.current = false;
  }

  function transposeSelectedNotes(semitones: number) {
    if (selectedNoteIds.size === 0) return;

    const nextNotes = notes.map((note) => {
      if (!selectedNoteIds.has(note.id)) return note;

      const midi = Math.min(
        pitchRange.max,
        Math.max(pitchRange.min, note.midi + semitones)
      );

      return {
        ...note,
        midi,
        name: Tone.Frequency(midi, "midi").toNote(),
      };
    });

    commitNotes(nextNotes);
  }

  function setSelectedVelocity(rawValue: number) {
    if (selectedNoteIds.size === 0 || !Number.isFinite(rawValue)) return;

    const velocity = Math.min(127, Math.max(1, Math.round(rawValue))) / 127;
    const nextNotes = notes.map((note) =>
      selectedNoteIds.has(note.id) ? { ...note, velocity } : note
    );

    if (
      nextNotes.some((note, index) => note.velocity !== notes[index].velocity)
    ) {
      commitNotes(nextNotes);
    }
  }

  function quantizeNotes() {
    if (notes.length === 0) return;

    const targetNoteIds =
      selectedNoteIds.size > 0
        ? selectedNoteIds
        : new Set(notes.map((note) => note.id));
    const nextNotes = notes.map((note) => {
      if (!targetNoteIds.has(note.id)) return note;

      return {
        ...note,
        time: Math.max(0, Math.round(note.time / snapSeconds) * snapSeconds),
      };
    });

    if (nextNotes.some((note, index) => note.time !== notes[index].time)) {
      commitNotes(nextNotes);
    }
  }

  function nudgeSelectedNotes(direction: -1 | 1) {
    if (selectedNoteIds.size === 0) return;

    const selectedNotes = notes.filter((note) => selectedNoteIds.has(note.id));
    const minimumTime = Math.min(...selectedNotes.map((note) => note.time));
    const offset = direction === -1 ? Math.max(-minimumTime, -snapSeconds) : snapSeconds;

    commitNotes(
      notes.map((note) =>
        selectedNoteIds.has(note.id)
          ? { ...note, time: Math.max(0, note.time + offset) }
          : note
      )
    );
  }

  function duplicateSelectedNotes() {
    if (selectedNoteIds.size === 0) return;

    const copies = notes
      .filter((note) => selectedNoteIds.has(note.id))
      .map((note) => ({
        ...note,
        id: makeNoteId(),
        time: note.time + snapSeconds,
      }));

    commitNotes([...notes, ...copies].sort((a, b) => a.time - b.time));
    setSelectedNoteIds(new Set(copies.map((note) => note.id)));
    setSelectionAnchorId(copies[0]?.id ?? null);
  }

  function copySelectedNotes() {
    if (selectedNoteIds.size === 0) return;

    noteClipboardRef.current = notes
      .filter((note) => selectedNoteIds.has(note.id))
      .map((note) => ({ ...note }));
    setClipboardNoteCount(noteClipboardRef.current.length);
  }

  function pasteCopiedNotes() {
    if (noteClipboardRef.current.length === 0) return;

    const earliestTime = Math.min(
      ...noteClipboardRef.current.map((note) => note.time)
    );
    const pasteTime = Math.max(
      0,
      Math.round(currentTime / snapSeconds) * snapSeconds
    );
    const copies = noteClipboardRef.current.map((note) => ({
      ...note,
      id: makeNoteId(),
      time: pasteTime + (note.time - earliestTime),
    }));

    commitNotes([...notes, ...copies].sort((a, b) => a.time - b.time));
    setSelectedNoteIds(new Set(copies.map((note) => note.id)));
    setSelectionAnchorId(copies[0]?.id ?? null);
  }

  function deleteSelectedNotes() {
    if (selectedNoteIds.size === 0) return;

    commitNotes(notes.filter((note) => !selectedNoteIds.has(note.id)));
    clearSelection();
  }

  function undoEdit() {
    const previous = pastNotes.at(-1);
    if (!previous) return;

    stopMidi();
    setPastNotes((history) => history.slice(0, -1));
    setFutureNotes((history) => [notes, ...history].slice(0, 50));
    setNotes(previous);
    clearSelection();
  }

  function redoEdit() {
    const next = futureNotes[0];
    if (!next) return;

    stopMidi();
    setPastNotes((history) => [...history.slice(-49), notes]);
    setFutureNotes((history) => history.slice(1));
    setNotes(next);
    clearSelection();
  }

  function handlePianoRollKeyDown(event: React.KeyboardEvent<HTMLElement>) {
    const modifier = event.metaKey || event.ctrlKey;

    if (modifier && event.key.toLowerCase() === "a") {
      event.preventDefault();
      selectAllNotes();
      return;
    }

    if (modifier && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) redoEdit();
      else undoEdit();
      return;
    }

    if (modifier && event.key.toLowerCase() === "y") {
      event.preventDefault();
      redoEdit();
      return;
    }

    if (modifier && event.key.toLowerCase() === "d") {
      event.preventDefault();
      duplicateSelectedNotes();
      return;
    }

    if (modifier && event.key.toLowerCase() === "c") {
      event.preventDefault();
      copySelectedNotes();
      return;
    }

    if (modifier && event.key.toLowerCase() === "v") {
      event.preventDefault();
      pasteCopiedNotes();
      return;
    }

    if (event.key === "Escape") {
      clearSelection();
      return;
    }

    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deleteSelectedNotes();
      return;
    }

    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      const direction = event.key === "ArrowUp" ? 1 : -1;
      transposeSelectedNotes(direction * (event.shiftKey ? 12 : 1));
      return;
    }

    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      nudgeSelectedNotes(event.key === "ArrowLeft" ? -1 : 1);
    }
  }

  async function loadMidiFromUrl(url: string, fileName: string) {
    const response = await fetch(url);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to download MIDI: ${response.status} ${errorText}`);
    }

    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const errorText = await response.text();
      throw new Error(`Expected MIDI but got JSON: ${errorText}`);
    }

    const arrayBuffer = await response.arrayBuffer();

    const header = new TextDecoder().decode(arrayBuffer.slice(0, 4));

    if (header !== "MThd") {
      const preview = new TextDecoder().decode(arrayBuffer.slice(0, 80));
      throw new Error(`Bad MIDI response. Expected MThd, got: ${preview}`);
    }

    await loadMidiArrayBuffer(arrayBuffer, fileName);
  }

  function updateStemMidiConversion(
    stemName: StemName,
    patch: Partial<StemMidiConversionState>
  ) {
    setStemMidiConversions((current) => ({
      ...current,
      [stemName]: {
        ...current[stemName],
        ...patch,
      },
    }));
  }

  function getMidiConversionStatus(stemName: StemName, rawStatus: string) {
    if (rawStatus.toLowerCase().includes("done")) return copy.stemMidiReady;
    if (stemName === "drums") return copy.convertingStemToMidi;
    if (stemName === "bass") return copy.convertingStemToMidi;

    return rawStatus || copy.convertingStemToMidi;
  }

  function getTrackProcessingStatus(uploadMode: ModeType, rawStatus: string) {
    const loweredStatus = rawStatus.toLowerCase();

    if (loweredStatus.includes("done")) return copy.complete;
    if (uploadMode === "drums") return copy.drumUploadStatus;

    return rawStatus || copy.processing;
  }

  function getTrackProcessingError(uploadMode: ModeType, rawError: string) {
    if (uploadMode === "drums" && rawError.includes("basic-pitch")) {
      return copy.wrongDrumPipeline;
    }

    if (rawError.includes("basic-pitch")) {
      return copy.basicPitchMissing;
    }

    return copy.backendError;
  }

  async function createAudioFileFromUrl(url: string, fileName: string) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error("Failed to load separated audio track.");
    }

    const blob = await response.blob();
    return new File([blob], fileName, {
      type: blob.type || "audio/wav",
    });
  }

  async function pollStemMidiJob(stemName: StemName, jobId: string) {
    const pollInterval = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`);
        if (!response.ok) {
          throw new Error("Stem MIDI job polling failed.");
        }

        const job = await response.json();

        updateStemMidiConversion(stemName, {
          progress: job.progress ?? 0,
          status: getMidiConversionStatus(stemName, job.status ?? ""),
        });

        if (job.error) {
          window.clearInterval(pollInterval);
          updateStemMidiConversion(stemName, {
            isProcessing: false,
            status: getTrackProcessingError(
              stemName === "drums" ? "drums" : "bass",
              String(job.error)
            ),
          });
          return;
        }

        if (job.result) {
          window.clearInterval(pollInterval);
          updateStemMidiConversion(stemName, {
            isProcessing: false,
            progress: 100,
            status: copy.stemMidiReady,
            result: job.result as JobResult,
          });
        }
      } catch {
        window.clearInterval(pollInterval);
        updateStemMidiConversion(stemName, {
          isProcessing: false,
          status: copy.stemConnectionError,
        });
      }
    }, 1000);
  }

  async function convertStemTrackToMidi(stemName: StemName) {
    const stemUrl = stemResult?.stems[stemName];
    if (!stemUrl || !MIDI_CONVERTIBLE_STEMS.includes(stemName)) return;

    updateStemMidiConversion(stemName, {
      isProcessing: true,
      progress: 0,
      status: copy.convertingStemToMidi,
      result: null,
    });

    try {
      const audioFile = await createAudioFileFromUrl(
        stemUrl,
        getFileNameFromUrl(stemUrl)
      );

      if (isDesktopRuntime()) {
        const mode = stemName === "drums" ? "drums" : "bass";
        const result = await runDesktopTranscription(audioFile, mode, (progress, status) => {
          updateStemMidiConversion(stemName, {
            progress,
            status: getMidiConversionStatus(stemName, status),
          });
        });

        updateStemMidiConversion(stemName, {
          isProcessing: false,
          progress: 100,
          status: copy.stemMidiReady,
          result: {
            midiUrl: desktopAssetUrl(result.midiPath),
            midiFileName: result.midiFileName,
            mode,
          },
        });
        return;
      }

      const formData = new FormData();
      formData.append("file", audioFile);

      const endpoint =
        stemName === "drums"
          ? `${API_BASE_URL}/api/drums`
          : `${API_BASE_URL}/api/transcribe`;

      if (stemName === "drums") {
        formData.append("separate", "false");
      } else {
        formData.append("mode", "bass");
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Stem MIDI conversion upload failed.");
      }

      const result = await response.json();

      updateStemMidiConversion(stemName, {
        status: copy.queued,
      });
      pollStemMidiJob(stemName, result.jobId);
    } catch {
      updateStemMidiConversion(stemName, {
        isProcessing: false,
        status: copy.stemConnectionError,
      });
    }
  }

  async function openStemMidiEditor(stemName: StemName) {
    const result = stemMidiConversions[stemName]?.result;
    if (!result) return;

    stopStemMix();
    handleModeChange(result.mode);
    setIsStemMixerOpen(false);
    setHasSelectedMode(true);
    await loadMidiFromUrl(result.midiUrl, result.midiFileName);
  }

  async function pollJob(jobId: string, uploadMode: ModeType) {
    const pollInterval = window.setInterval(async () => {
      const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`);
      const job = await response.json();

      setProcessingProgress(job.progress ?? 0);
      setProcessingStatus(getTrackProcessingStatus(uploadMode, job.status ?? ""));

      if (job.error) {
        window.clearInterval(pollInterval);
        setIsProcessing(false);
        setProcessingStatus(getTrackProcessingError(uploadMode, String(job.error)));
        return;
      }

      if (job.result) {
        window.clearInterval(pollInterval);
        const result = job.result as JobResult;

        try {
          setProcessingStatus(copy.loadingGeneratedMidi);
          await loadMidiFromUrl(result.midiUrl, result.midiFileName);

          setProcessingProgress(100);
          setProcessingStatus(copy.complete);
        } catch (error) {
          console.error(error);
          setProcessingStatus(
            error instanceof Error ? error.message : "Failed to load generated MIDI."
          );
        } finally {
          setIsProcessing(false);
        }
      }
    }, 1000);
  }

  async function pollStemJob(jobId: string) {
    const pollInterval = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`);
        if (!response.ok) {
          throw new Error("Stem job polling failed.");
        }

        const job = await response.json();

        setStemProgress(job.progress ?? 0);
        setStemStatus(copy.separatingStems);

        if (job.error) {
          window.clearInterval(pollInterval);
          setIsSeparatingStems(false);
          setStemStatus(copy.stemFailed);
          return;
        }

        if (job.result) {
          window.clearInterval(pollInterval);
          const result = job.result as StemJobResult;

          setStemResult(result);
          setStemProgress(100);
          setStemStatus(copy.openStemMixer);
          setIsSeparatingStems(false);
          setIsStemMixerOpen(true);

          const startedAt = stemJobStartedAtRef.current;
          if (startedAt) {
            saveStemEstimateCalibration(
              stemAudioDurationRef.current,
              Math.max(1, Math.round((getCurrentTimeMs() - startedAt) / 1000))
            );
          }
        }
      } catch {
        window.clearInterval(pollInterval);
        setIsSeparatingStems(false);
        setStemStatus(copy.stemConnectionError);
      }
    }, 1000);
  }

  async function handleStemUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;

    setStemFileName(file.name);
    setStemResult(null);
    setIsStemMixerOpen(false);
    setIsSeparatingStems(true);
    setStemProgress(0);
    setStemElapsedSeconds(0);
    setStemEstimateSeconds(DEFAULT_STEM_ESTIMATE_SECONDS);
    setStemTrackDurations({});
    setStemWaveforms({});
    setStemTempoBpm(null);
    stemAudioBuffersRef.current = {};
    stopStemAudioSources();
    setStemMidiConversions(createInitialStemMidiConversions());
    setMutedStems({ vocals: false, drums: false, bass: false, other: false });
    setSoloedStems({ vocals: false, drums: false, bass: false, other: false });
    setStemMixTime(0);
    setIsStemMixPlaying(false);
    setStemStatus(copy.uploadingStemAudio);
    stemJobStartedAtRef.current = getCurrentTimeMs();
    stemAudioDurationRef.current = null;

    if (isDesktopRuntime()) {
      try {
        const result = await runDesktopSeparation(file, (progress, status) => {
          setStemProgress(progress);
          setStemStatus(status || copy.separatingStems);
        });

        setStemResult({
          mode: "separate",
          model: result.model,
          device: result.device,
          stems: Object.fromEntries(
            Object.entries(result.stems).map(([stemName, stemPath]) => [
              stemName,
              desktopAssetUrl(stemPath),
            ])
          ) as Partial<Record<StemName, string>>,
        });
        setStemProgress(100);
        setStemStatus(copy.openStemMixer);
        setIsSeparatingStems(false);
        setIsStemMixerOpen(true);
      } catch (error) {
        console.error(error);
        setIsSeparatingStems(false);
        setStemStatus(
          error instanceof Error ? error.message : copy.stemConnectionError
        );
      }
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", "htdemucs");

    try {
      const audioDurationSeconds = await readAudioMetadataDuration(file);
      stemAudioDurationRef.current = audioDurationSeconds;
      setStemEstimateSeconds(estimateStemSeparationSeconds(audioDurationSeconds));

      const response = await fetch(`${API_BASE_URL}/api/separate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend stem separation upload failed.");
      }

      const result = await response.json();

      setStemStatus(copy.queued);
      pollStemJob(result.jobId);
    } catch {
      setIsSeparatingStems(false);
      setStemStatus(copy.stemConnectionError);
    }
  }

  async function handleAudioUpload(
    event: React.ChangeEvent<HTMLInputElement>,
    uploadMode: ModeType = mode
  ) {
    const file = event.target.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;

    stopMidi();

    setAudioFileName(file.name);
    setMidiFileName("");
    setNotes([]);
    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingStatus(copy.uploadingAudio);

    if (isDesktopRuntime()) {
      try {
        const result = await runDesktopTranscription(file, uploadMode, (progress, status) => {
          setProcessingProgress(progress);
          setProcessingStatus(status || copy.processing);
        });

        setProcessingStatus(copy.loadingGeneratedMidi);
        await loadMidiFromUrl(
          desktopAssetUrl(result.midiPath),
          result.midiFileName
        );
        setProcessingProgress(100);
        setProcessingStatus(copy.complete);
      } catch (error) {
        console.error(error);
        setProcessingStatus(
          error instanceof Error ? error.message : copy.backendError
        );
      } finally {
        setIsProcessing(false);
      }
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoint =
        uploadMode === "drums"
          ? `${API_BASE_URL}/api/drums`
          : `${API_BASE_URL}/api/transcribe`;

      if (uploadMode === "drums") {
        formData.append("separate", "false");
      } else {
        formData.append("mode", uploadMode);
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend upload failed.");
      }

      const result = await response.json();

      setProcessingStatus(copy.queued);
      pollJob(result.jobId, uploadMode);
    } catch {
      setIsProcessing(false);
      setProcessingStatus(copy.backendError);
    }
  }

  async function handleMidiUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    stopMidi();

    const arrayBuffer = await file.arrayBuffer();
    await loadMidiArrayBuffer(arrayBuffer, file.name);
  }

  function scheduleNotes() {
    Tone.Transport.cancel();

    notes.forEach((note) => {
      Tone.Transport.schedule((time) => {
        const instrument = instrumentRef.current;
        if (!instrument) return;

        const noteName = Tone.Frequency(note.midi, "midi").toNote();

        instrument.triggerAttackRelease(
          noteName,
          Math.max(note.duration, 0.05),
          time,
          Math.min(Math.max(note.velocity * 0.95, 0.08), 0.98)
        );
      }, note.time);
    });

    Tone.Transport.scheduleOnce(() => {
      stopMidi();
    }, totalDuration + 0.5);
  }

  async function playMidi(startAt?: number) {
    if (notes.length === 0) return;

    await Tone.start();

    if (!instrumentRef.current || !isInstrumentReady) {
      await preloadInstrument(instrumentType);
    }

    if (isPaused && startAt === undefined) {
      Tone.Transport.start();
      setIsPlaying(true);
      setIsPaused(false);
      startProgressTimer();
      return;
    }

    const playbackStartTime = Math.min(
      totalDuration,
      Math.max(0, startAt ?? 0)
    );

    Tone.Transport.stop();
    Tone.getTransport().seconds = playbackStartTime;
    setCurrentTime(playbackStartTime);

    scheduleNotes();

    Tone.Transport.start();

    setIsPlaying(true);
    setIsPaused(false);
    startProgressTimer();
  }

  function pauseMidi() {
    Tone.Transport.pause();

    setIsPlaying(false);
    setIsPaused(true);
    clearProgressTimer();
    setCurrentTime(Tone.Transport.seconds);
  }

  function stopMidi() {
    Tone.Transport.stop();
    Tone.Transport.cancel();

    setIsPlaying(false);
    setIsPaused(false);
    setCurrentTime(0);
    clearProgressTimer();
  }

  function seekTo(value: number) {
    const nextTime = Math.min(totalDuration, Math.max(0, value));
    Tone.getTransport().seconds = nextTime;
    setCurrentTime(nextTime);
  }

  function isStemAudible(stemName: StemName) {
    return hasSoloedStems ? soloedStems[stemName] : !mutedStems[stemName];
  }

  function getStemAudioContext() {
    if (!stemAudioContextRef.current) {
      stemAudioContextRef.current = new AudioContext();
    }

    return stemAudioContextRef.current;
  }

  function clearStemMixTimer() {
    if (stemMixTimerRef.current !== null) {
      window.clearInterval(stemMixTimerRef.current);
      stemMixTimerRef.current = null;
    }
  }

  function stopStemAudioSources() {
    Object.values(stemAudioSourcesRef.current).forEach((source) => {
      try {
        source?.stop();
      } catch {
        // Source may already be stopped.
      }
    });

    stemAudioSourcesRef.current = {};
    stemGainNodesRef.current = {};
    clearStemMixTimer();
  }

  function pauseStemMix() {
    const audioContext = stemAudioContextRef.current;

    if (audioContext && isStemMixPlaying) {
      const elapsed = Math.max(0, audioContext.currentTime - stemMixStartedAtRef.current);
      setStemMixTime(Math.min(stemMixerDuration, stemMixStartOffsetRef.current + elapsed));
    }

    stopStemAudioSources();
    setIsStemMixPlaying(false);
  }

  function stopStemMix() {
    stopStemAudioSources();
    setStemMixTime(0);
    setIsStemMixPlaying(false);
  }

  function startStemMixTimer(audioContext: AudioContext) {
    clearStemMixTimer();

    stemMixTimerRef.current = window.setInterval(() => {
      const elapsed = Math.max(0, audioContext.currentTime - stemMixStartedAtRef.current);
      const nextTime = Math.min(stemMixerDuration, stemMixStartOffsetRef.current + elapsed);

      setStemMixTime(nextTime);

      if (stemMixerDuration > 0 && nextTime >= stemMixerDuration - 0.02) {
        stopStemMix();
      }
    }, 33);
  }

  async function playStemMix(startAt = stemMixTime) {
    const audioContext = getStemAudioContext();
    const loadedStemNames = importedStemNames.filter(
      (stemName) => Boolean(stemAudioBuffersRef.current[stemName])
    );
    if (loadedStemNames.length === 0) return;

    stopStemAudioSources();
    await audioContext.resume();

    const offset = Math.max(0, Math.min(startAt, stemMixerDuration || startAt));
    const startTime = audioContext.currentTime + 0.06;

    try {
      loadedStemNames.forEach((stemName) => {
        const buffer = stemAudioBuffersRef.current[stemName];
        if (!buffer || offset >= buffer.duration) return;

        const source = audioContext.createBufferSource();
        const gainNode = audioContext.createGain();
        source.buffer = buffer;
        gainNode.gain.value = isStemAudible(stemName) ? 1 : 0;
        source.connect(gainNode);
        gainNode.connect(audioContext.destination);
        source.start(startTime, offset);
        stemAudioSourcesRef.current[stemName] = source;
        stemGainNodesRef.current[stemName] = gainNode;
      });

      stemMixStartedAtRef.current = startTime;
      stemMixStartOffsetRef.current = offset;
      setStemMixTime(offset);
      setIsStemMixPlaying(true);
      startStemMixTimer(audioContext);
    } catch {
      stopStemAudioSources();
      setIsStemMixPlaying(false);
    }
  }

  function seekStemMix(value: number) {
    const nextTime = Math.max(0, Math.min(value, stemMixerDuration || value));
    const wasPlaying = isStemMixPlaying;

    stopStemAudioSources();
    setStemMixTime(nextTime);

    if (wasPlaying) {
      void playStemMix(nextTime);
    }
  }

  function seekStemMixFromPointer(event: React.PointerEvent<HTMLElement>) {
    if (stemMixerDuration <= 0) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const content = event.currentTarget.querySelector<HTMLElement>(
      "[data-stem-arrange-content]"
    );
    const contentWidth = content?.offsetWidth ?? rect.width;
    const pointerX = event.clientX - rect.left + event.currentTarget.scrollLeft;
    const ratio = Math.min(1, Math.max(0, pointerX / Math.max(1, contentWidth)));
    seekStemMix(stemMixerDuration * ratio);
  }

  function registerStemArrangeScrollContainer(element: HTMLElement | null) {
    if (!element) return;

    stemArrangeScrollContainersRef.current.add(element);
  }

  function syncStemArrangeScroll(event: React.UIEvent<HTMLElement>) {
    if (isSyncingStemArrangeScrollRef.current) return;

    isSyncingStemArrangeScrollRef.current = true;
    stemArrangeScrollContainersRef.current.forEach((element) => {
      if (element !== event.currentTarget) {
        element.scrollLeft = event.currentTarget.scrollLeft;
      }
    });
    window.requestAnimationFrame(() => {
      isSyncingStemArrangeScrollRef.current = false;
    });
  }

  function toggleStemMute(stemName: StemName) {
    setMutedStems((current) => ({
      ...current,
      [stemName]: !current[stemName],
    }));
  }

  function toggleStemSolo(stemName: StemName) {
    setSoloedStems((current) => ({
      ...current,
      [stemName]: !current[stemName],
    }));
  }

  function returnFromStemMixer() {
    stopStemMix();
    setIsStemMixerOpen(false);
    setStemResult(null);
    setStemStatus("");
    setStemFileName("");
  }

  useEffect(() => {
    const audioContext = stemAudioContextRef.current;

    STEM_ORDER.forEach((stemName) => {
      const gainNode = stemGainNodesRef.current[stemName];
      if (!gainNode) return;

      const hasSolo = STEM_ORDER.some((name) => soloedStems[name]);
      const value = hasSolo ? (soloedStems[stemName] ? 1 : 0) : mutedStems[stemName] ? 0 : 1;
      gainNode.gain.setTargetAtTime(value, audioContext?.currentTime ?? 0, 0.008);
    });
  }, [mutedStems, soloedStems, stemResult]);

  useEffect(() => {
    if (!stemResult) return;

    let isCancelled = false;
    const audioContext = getStemAudioContext();

    stemAudioBuffersRef.current = {};
    const loadStemAudio = async () => {
      const loadedStems = await Promise.all(
        importedStemNames.map(async (stemName) => {
          const stemUrl = stemResult.stems[stemName];
          if (!stemUrl) return null;

          const response = await fetch(stemUrl);
          if (!response.ok) {
            throw new Error("Failed to load stem audio.");
          }

          const arrayBuffer = await response.arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          return {
            audioBuffer,
            stemName,
          };
        })
      );

      if (isCancelled) return;

      const nextBuffers: Partial<Record<StemName, AudioBuffer>> = {};
      const nextDurations: Partial<Record<StemName, number>> = {};
      const nextWaveforms: StemWaveformPeaks = {};
      const audioBuffers: AudioBuffer[] = [];

      loadedStems.forEach((loadedStem) => {
        if (!loadedStem) return;

        nextBuffers[loadedStem.stemName] = loadedStem.audioBuffer;
        nextDurations[loadedStem.stemName] = loadedStem.audioBuffer.duration;
        nextWaveforms[loadedStem.stemName] = buildWaveformPeaks(loadedStem.audioBuffer);
        audioBuffers.push(loadedStem.audioBuffer);
      });

      stemAudioBuffersRef.current = nextBuffers;
      setStemTrackDurations(nextDurations);
      setStemWaveforms(nextWaveforms);
      setStemTempoBpm(estimateTempoFromAudioBuffers(audioBuffers));
    };

    void loadStemAudio().catch(() => {
      if (isCancelled) return;

      setStemWaveforms(
        Object.fromEntries(
          importedStemNames.map((stemName) => [stemName, createFallbackWaveform()])
        ) as StemWaveformPeaks
      );
    });

    return () => {
      isCancelled = true;
    };
  }, [stemResult, importedStemNames]);

  useEffect(() => {
    if (!isSeparatingStems) return;

    const timerId = window.setInterval(() => {
      setStemElapsedSeconds((seconds) => seconds + 1);
    }, 1000);

    return () => window.clearInterval(timerId);
  }, [isSeparatingStems]);

  useEffect(() => {
    function handleGlobalEditorShortcut(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const isTextOrFormControl = target?.closest(
        'input, select, textarea, [contenteditable="true"]'
      );

      if (isTextOrFormControl) return;

      if (event.code === "Space" && !event.repeat) {
        if (notes.length === 0 || isProcessing || isInstrumentLoading) return;

        event.preventDefault();

        if (isPlaying) {
          pauseMidi();
        } else {
          void playMidi();
        }
        return;
      }

      if (target?.closest('[role="region"]')) return;

      const modifier = event.metaKey || event.ctrlKey;
      const key = event.key.toLowerCase();

      if (modifier && key === "a") {
        event.preventDefault();
        selectAllNotes();
        return;
      }

      if (modifier && key === "z") {
        event.preventDefault();
        if (event.shiftKey) redoEdit();
        else undoEdit();
        return;
      }

      if (modifier && key === "y") {
        event.preventDefault();
        redoEdit();
        return;
      }

      if (modifier && key === "d") {
        event.preventDefault();
        duplicateSelectedNotes();
        return;
      }

      if (modifier && key === "c") {
        event.preventDefault();
        copySelectedNotes();
        return;
      }

      if (modifier && key === "v") {
        event.preventDefault();
        pasteCopiedNotes();
        return;
      }

      if (event.key === "Escape") {
        clearSelection();
        return;
      }

      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        deleteSelectedNotes();
        return;
      }

      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        nudgeSelectedNotes(event.key === "ArrowLeft" ? -1 : 1);
        return;
      }

      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        event.preventDefault();
        const direction = event.key === "ArrowUp" ? 1 : -1;
        transposeSelectedNotes(direction * (event.shiftKey ? 12 : 1));
      }
    }

    window.addEventListener("keydown", handleGlobalEditorShortcut);
    return () => {
      window.removeEventListener("keydown", handleGlobalEditorShortcut);
    };
  });

  function handleModeChange(newMode: ModeType) {
    setMode(newMode);

    const nextInstrumentType =
      newMode === "drums"
        ? "drum_kit"
        : newMode === "bass"
          ? "electric_bass"
          : "grand_piano";

    if (instrumentType !== nextInstrumentType) {
      stopMidi();
      disposeInstrument();
      setIsInstrumentReady(false);
      setInstrumentType(nextInstrumentType);
    }
  }

  function enterWorkspace(newMode: ModeType) {
    resetMidiWorkspaceSession();
    handleModeChange(newMode);
    setHasSelectedMode(true);
  }

  function returnToModeSelection() {
    resetMidiWorkspaceSession();
    stopStemMix();
    setIsStemMixerOpen(false);
    setHasSelectedMode(false);
  }

  function selectMeter(value: string) {
    const meter = MUSICAL_METER_PRESETS.find(
      (preset) => `${preset.numerator}/${preset.denominator}` === value
    );
    if (!meter) return;

    setTimeSignature({
      numerator: meter.numerator,
      denominator: meter.denominator,
    });
    setExportStatus("");
  }

  function exportEditedMidi() {
    const numerator = Math.round(timeSignature.numerator);
    const denominator = Math.round(timeSignature.denominator);

    if (!isPowerOfTwo(denominator)) {
      setExportStatus(
        "Standard MIDI only supports power-of-two meter denominators (for example 2, 4, 8, or 16)."
      );
      return;
    }

    const midi = new Midi();
    midi.header.setTempo(safeTempoBpm);
    midi.header.timeSignatures = [
      { ticks: 0, timeSignature: [numerator, denominator] },
    ];
    midi.header.update();

    const track = midi.addTrack();
    track.name = "Edited Piano Roll";
    notes
      .slice()
      .sort((a, b) => a.time - b.time)
      .forEach((note) => {
        track.addNote({
          midi: note.midi,
          time: note.time,
          duration: Math.max(MIN_NOTE_DURATION, note.duration),
          velocity: Math.min(Math.max(note.velocity, 0), 1),
        });
      });

    const bytes = midi.toArray();
    const exportBuffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(exportBuffer).set(bytes);
    const url = URL.createObjectURL(
      new Blob([exportBuffer], { type: "audio/midi" })
    );
    const link = document.createElement("a");
    const sourceName = midiFileName.replace(/\.(mid|midi)$/i, "") || "edited-midi";

    link.href = url;
    link.download = `${sourceName}-edited.mid`;
    link.click();
    URL.revokeObjectURL(url);
    setExportStatus(copy.exported);
  }

  function handlePianoRollClick(event: React.MouseEvent<HTMLElement>) {
    if (suppressPianoRollClickRef.current) {
      suppressPianoRollClickRef.current = false;
      return;
    }

    const target = event.target as HTMLElement;
    if (target.closest("[data-piano-note]")) return;

    const editor = event.currentTarget;
    const rect = editor.getBoundingClientRect();
    const relativeX =
      event.clientX - rect.left + editor.scrollLeft - PIANO_KEY_WIDTH;
    const clickedTime = Math.min(
      totalDuration,
      Math.max(0, (relativeX / pixelsPerBeat) * secondsPerQuarterBeat)
    );

    clearSelection();
    void playMidi(clickedTime);
  }

  const glassPanelStyle = {
    backdropFilter: "blur(26px) saturate(180%)",
    WebkitBackdropFilter: "blur(26px) saturate(180%)",
  };
  const glassCardStyle = {
    backdropFilter: "blur(18px) saturate(160%)",
    WebkitBackdropFilter: "blur(18px) saturate(160%)",
  };

  if (isStemMixerOpen && stemResult) {
    return (
      <main
        className="logic-daw stem-mixer-page"
        style={{ padding: "32px clamp(18px, 4vw, 56px)", fontFamily: "sans-serif" }}
      >
        <header className="logic-daw-header">
          <div className="workspace-brand">
            <span className="trackform-mark" aria-hidden="true"><i /><i /><i /></span>
            <div><p>TRACKFORM</p><h1>{copy.stemMixer}</h1></div>
          </div>
          <span className="workspace-tagline">{copy.stemMixerCopy}</span>
          <div className="workspace-header-actions">
            <div className="language-switch compact" role="group" aria-label="Language">
              <button aria-pressed={language === "en"} onClick={() => setLanguage("en")}>EN</button>
              <button aria-pressed={language === "ko"} onClick={() => setLanguage("ko")}>한국어</button>
            </div>
            <button onClick={returnFromStemMixer}>{copy.backToHome}</button>
          </div>
        </header>

        <section className="stem-mixer-shell" style={glassPanelStyle}>
          <div className="stem-transport">
            <div>
              <span className="stem-session-label">{copy.imported}</span>
              <strong>{stemFileName || "Trackform stems"}</strong>
              <small>{stemTempoBpm ? `${stemTempoBpm} BPM · 4/4` : copy.analyzingTempo}</small>
            </div>
            <div className="stem-transport-buttons">
              <button onClick={() => { void (isStemMixPlaying ? pauseStemMix() : playStemMix()); }}>
                {isStemMixPlaying ? copy.pauseStems : copy.playAllStems}
              </button>
              <button onClick={stopStemMix}>{copy.stopStems}</button>
            </div>
            <div className="stem-time-readout">
              <span>{formatTime(stemMixTime)}</span>
              <input
                type="range"
                min="0"
                max={Math.max(stemMixerDuration, 1)}
                step="0.01"
                value={Math.min(stemMixTime, Math.max(stemMixerDuration, 1))}
                onChange={(event) => seekStemMix(Number(event.target.value))}
                aria-label={copy.stemPlayer}
              />
              <span>{formatTime(stemMixerDuration)}</span>
            </div>
            <div className="stem-waveform-zoom">
              <span>{copy.waveformZoom}</span>
              <button
                type="button"
                onClick={() => setStemHorizontalZoom((zoom) => Math.max(1, Number((zoom - 0.25).toFixed(2))))}
                aria-label={`${copy.waveformZoom} down`}
              >
                −
              </button>
              <input
                type="range"
                min="1"
                max="10"
                step="0.25"
                value={stemHorizontalZoom}
                onChange={(event) => setStemHorizontalZoom(Number(event.target.value))}
                aria-label={copy.waveformZoom}
              />
              <button
                type="button"
                onClick={() => setStemHorizontalZoom((zoom) => Math.min(10, Number((zoom + 0.25).toFixed(2))))}
                aria-label={`${copy.waveformZoom} up`}
              >
                +
              </button>
              <strong>{Math.round(stemHorizontalZoom * 100)}%</strong>
            </div>
          </div>

          <div
            className="stem-arrange-timeline"
            ref={registerStemArrangeScrollContainer}
            onPointerDown={seekStemMixFromPointer}
            onScroll={syncStemArrangeScroll}
            role="slider"
            aria-label={copy.stemPlayer}
            aria-valuemin={0}
            aria-valuemax={Math.max(stemMixerDuration, 1)}
            aria-valuenow={stemMixTime}
          >
            <div
              className="stem-arrange-content"
              data-stem-arrange-content
              style={{ width: stemArrangeWidth }}
            >
              {stemBarMarkers.map((marker) => (
                <span
                  className={marker.bar % 4 === 1 ? "is-major" : ""}
                  key={`stem-marker-${marker.bar}`}
                  style={{ left: `${marker.left}%` }}
                >
                  {marker.bar}
                </span>
              ))}
              <i style={{ left: `${stemMixerProgress}%` }} />
            </div>
          </div>

          <div className="stem-track-list">
            {importedStemNames.map((stemName, index) => {
              const stemUrl = stemResult.stems[stemName];
              if (!stemUrl) return null;

              const isMuted = mutedStems[stemName];
              const isSoloed = soloedStems[stemName];
              const isAudible = isStemAudible(stemName);
              const midiConversion = stemMidiConversions[stemName];
              const canConvertToMidi = MIDI_CONVERTIBLE_STEMS.includes(stemName);

              return (
                <article className={`stem-track${isAudible ? "" : " is-silent"}`} key={stemName}>
                  <div className="stem-track-strip">
                    <span className="stem-track-number">{String(index + 1).padStart(2, "0")}</span>
                    <div>
                      <strong>{STEM_LABELS[stemName]}</strong>
                      <span>{getFileNameFromUrl(stemUrl)}</span>
                    </div>
                    <div className="stem-track-controls">
                      <button
                        className={isMuted ? "is-active" : ""}
                        aria-pressed={isMuted}
                        onClick={() => toggleStemMute(stemName)}
                      >
                        {copy.mute}
                      </button>
                      <button
                        className={isSoloed ? "is-active" : ""}
                        aria-pressed={isSoloed}
                        onClick={() => toggleStemSolo(stemName)}
                      >
                        {copy.solo}
                      </button>
                    </div>
                    {canConvertToMidi && (
                      <div className="stem-midi-actions">
                        <button
                          onClick={() => { void convertStemTrackToMidi(stemName); }}
                          disabled={midiConversion.isProcessing}
                        >
                          {copy.convertStemToMidi}
                        </button>
                        {midiConversion.status && (
                          <span>{midiConversion.status}</span>
                        )}
                        {(midiConversion.isProcessing || midiConversion.progress > 0) && (
                          <div className="processing-track" aria-label={`${midiConversion.progress}% ${copy.processed}`}>
                            <div style={{ width: `${midiConversion.progress}%` }} />
                          </div>
                        )}
                        {midiConversion.result && (
                          <div className="stem-midi-result-actions">
                            <button onClick={() => { void openStemMidiEditor(stemName); }}>
                              {copy.openMidiEditor}
                            </button>
                            <a href={midiConversion.result.midiUrl} download={midiConversion.result.midiFileName}>
                              {copy.stemDownload}
                            </a>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  <div
                    className="stem-arrange-lane"
                    ref={registerStemArrangeScrollContainer}
                    onPointerDown={seekStemMixFromPointer}
                    onScroll={syncStemArrangeScroll}
                  >
                    <div
                      className="stem-arrange-content stem-lane-content"
                      data-stem-arrange-content
                      style={{ width: stemArrangeWidth }}
                    >
                      {stemBarMarkers.map((marker) => (
                        <span
                          className={`stem-bar-line${marker.bar % 4 === 1 ? " is-major" : ""}`}
                          key={`${stemName}-bar-${marker.bar}`}
                          style={{ left: `${marker.left}%` }}
                        />
                      ))}
                      <span
                        className="stem-lane-playhead"
                        style={{ left: `${stemMixerProgress}%` }}
                      />
                      <div className={`stem-region stem-region-${stemName}`}>
                        <span className="stem-waveform-center" aria-hidden="true" />
                        {(stemWaveforms[stemName]?.length ? stemWaveforms[stemName] : createFallbackWaveform()).map((peak, peakIndex) => {
                          const top = `${Math.max(0, Math.min(100, (1 - peak.max) * 50))}%`;
                          const height = `${Math.max(1, Math.min(100, (peak.max - peak.min) * 50))}%`;

                          return (
                            <i
                              key={`${stemName}-wave-${peakIndex}`}
                              style={{ top, height }}
                            />
                          );
                        })}
                      </div>
                    </div>
                  </div>
                  <a className="stem-track-download" href={stemUrl} download={getFileNameFromUrl(stemUrl)}>
                    {copy.stemDownload}
                  </a>
                </article>
              );
            })}
          </div>
        </section>
      </main>
    );
  }

  if (!hasSelectedMode) {
    return (
      <main
        className="logic-daw trackform-onboarding"
        style={{ padding: "32px clamp(18px, 4vw, 56px)", fontFamily: "sans-serif" }}
      >
        <section className="mode-launcher" style={glassPanelStyle}>
          <div className="language-switch" role="group" aria-label="Language">
            <button aria-pressed={language === "en"} onClick={() => setLanguage("en")}>EN</button>
            <button aria-pressed={language === "ko"} onClick={() => setLanguage("ko")}>한국어</button>
          </div>
          <div className="trackform-hero-brand" aria-label="Trackform">
            <span className="trackform-mark" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <p className="brand-kicker">TRACKFORM</p>
          </div>
          <p className="brand-category">{copy.brandCategory}</p>
          <h1>{copy.heroTitle}</h1>
          <p className="mode-launcher-copy">
            {copy.heroCopy}
          </p>
          <div className="stem-upload-panel">
            <input
              id="trackform-stem-upload"
              className="stem-upload-input"
              type="file"
              accept="audio/*"
              onChange={handleStemUpload}
              disabled={isSeparatingStems}
              aria-label={copy.uploadMusic}
            />
            <label
              className={`stem-upload-trigger${isSeparatingStems ? " is-disabled" : ""}`}
              htmlFor="trackform-stem-upload"
            >
              <span className="stem-upload-icon"><MusicUploadIcon /></span>
              <span className="stem-upload-text">
                <strong>{copy.uploadMusic}</strong>
                <small>{copy.uploadMusicCopy}</small>
              </span>
            </label>
            {stemFileName && (
              <p className="stem-source-file">
                <span>{copy.stemFile}</span>
                <strong>{stemFileName}</strong>
              </p>
            )}
            {stemStatus && !isSeparatingStems && !stemResult && (
              <p className="stem-inline-message" aria-live="polite">{stemStatus}</p>
            )}
          </div>
          <div className="individual-convert-panel">
            <div className="individual-convert-heading">
              <span>02</span>
              <div>
                <h2>{copy.individualMidi}</h2>
                <p>{copy.individualMidiCopy}</p>
              </div>
            </div>
            <div className="mode-option-grid individual-mode-grid">
              <button onClick={() => enterWorkspace("piano")}>
                <span className="mode-icon"><ModeIcon mode="piano" /></span>
                <span>Piano</span>
              </button>
              <button onClick={() => enterWorkspace("bass")}>
                <span className="mode-icon"><ModeIcon mode="bass" /></span>
                <span>Bass</span>
              </button>
              <button onClick={() => enterWorkspace("drums")}>
                <span className="mode-icon"><ModeIcon mode="drums" /></span>
                <span>Drums</span>
              </button>
              <button onClick={() => enterWorkspace("general")}>
                <span className="mode-icon"><ModeIcon mode="general" /></span>
                <span>{copy.otherMode}</span>
              </button>
            </div>
          </div>
          <p className="mode-launcher-footnote">{copy.chooseWorkspace}</p>
        </section>
        {isSeparatingStems && (
          <div className="stem-modal-backdrop" role="dialog" aria-modal="true" aria-label={copy.preparingStemSession}>
            <div className="stem-progress-modal" style={glassPanelStyle}>
              <span className="stem-modal-icon" aria-hidden="true"><MusicUploadIcon /></span>
              <p className="stem-modal-kicker">TRACKFORM STEMS</p>
              <h2>{copy.preparingStemSession}</h2>
              <p>{stemStatus || copy.separatingStems}</p>
              <div className="stem-modal-countdown">
                <span>{isPastStemEstimate ? copy.finishingStems : copy.timeRemaining}</span>
                <strong>{stemCountdownLabel}</strong>
              </div>
              <div className="processing-track" aria-label={`${displayedStemProgress}% ${copy.processed}`}>
                <div style={{ width: `${displayedStemProgress}%` }} />
              </div>
            </div>
          </div>
        )}
      </main>
    );
  }

  return (
    <main
      className="logic-daw"
      style={{ padding: "32px clamp(18px, 4vw, 56px)", fontFamily: "sans-serif" }}
      >
      <header className="logic-daw-header">
        <div className="workspace-brand">
          <span className="trackform-mark" aria-hidden="true"><i /><i /><i /></span>
          <div><p>TRACKFORM</p><h1>{copy.studio}</h1></div>
        </div>
        <span className="workspace-tagline">{copy.tagline}</span>
        <div className="workspace-header-actions">
          <div className="language-switch compact" role="group" aria-label="Language">
            <button aria-pressed={language === "en"} onClick={() => setLanguage("en")}>EN</button>
            <button aria-pressed={language === "ko"} onClick={() => setLanguage("ko")}>한국어</button>
          </div>
          <span className="mode-badge">{mode} mode</span>
          <button onClick={returnToModeSelection}>{copy.switchMode}</button>
        </div>
      </header>

      <section className="session-start" style={glassPanelStyle}>
        <div className="section-heading">
          <span>01</span>
          <div><h2>{copy.startSession}</h2><p>{copy.startSessionCopy}</p></div>
        </div>
        <div className="source-grid">
          <div className="source-card" style={glassCardStyle}>
            <span className="source-card-icon" aria-hidden="true">↗</span>
            <div><h3>{copy.createAudio}</h3><p>{copy.createAudioCopy}</p></div>
            <input
              type="file"
              accept="audio/*"
              onChange={(event) => { void handleAudioUpload(event, mode); }}
              disabled={isProcessing}
              aria-label={copy.uploadAudio}
            />
            {audioFileName && <strong className="source-file">{audioFileName}</strong>}
            {(isProcessing || processingStatus) && (
              <div className="processing-status">
                <p><strong>{copy.processing}</strong> {processingStatus}</p>
                <div className="processing-track" aria-label={`${processingProgress}% ${copy.processed}`}><div style={{ width: `${processingProgress}%` }} /></div>
              </div>
            )}
          </div>
          <div className="source-card" style={glassCardStyle}>
            <span className="source-card-icon" aria-hidden="true">⌁</span>
            <div><h3>{copy.openMidi}</h3><p>{copy.openMidiCopy}</p></div>
            <input type="file" accept=".mid,.midi" onChange={handleMidiUpload} disabled={isProcessing} aria-label={copy.openMidiFile} />
          </div>
        </div>
      </section>

      {midiFileName && (
        <section className="session-summary" style={glassPanelStyle}>
          <strong className="session-file">{midiFileName}</strong>
          <div><span>{copy.events}</span><strong>{notes.length}</strong></div>
          <div><span>{copy.duration}</span><strong>{formatTime(totalDuration)}</strong></div>
          <div><span>{copy.tempo}</span><strong>{tempoBpm} BPM</strong></div>
        </section>
      )}

      <section className="playback-panel" style={glassPanelStyle}>
        <div className="section-heading">
          <span>02</span>
          <div><h2>{copy.preview}</h2><p>{copy.previewCopy}</p></div>
        </div>
        <div className="playback-controls">
          <div className="transport-buttons">
            <button className="primary-action" onClick={() => void playMidi()} disabled={notes.length === 0 || isProcessing || isInstrumentLoading}>{isPaused ? copy.resume : copy.play}</button>
            <button onClick={pauseMidi} disabled={!isPlaying}>{copy.pause}</button>
            <button onClick={stopMidi} disabled={!isPlaying && !isPaused}>{copy.stop}</button>
          </div>
          <label className="sound-picker">
            <span>{copy.sound}</span>
            <select value={instrumentType} onChange={(event) => { stopMidi(); disposeInstrument(); setIsInstrumentReady(false); setInstrumentType(event.target.value as InstrumentType); }}>
              <option value="grand_piano">{copy.grandPiano}</option>
              <option value="double_bass">{copy.doubleBass}</option>
              <option value="electric_bass">{copy.electricBass}</option>
              <option value="drum_kit">{copy.drumKit}</option>
            </select>
          </label>
          <span className="instrument-status">{isInstrumentLoading ? copy.loadingSound : isInstrumentReady ? copy.ready : copy.readyOnPlay}</span>
        </div>
        <label className="transport-timeline">
          <span>{formatTime(currentTime)} <em>/</em> {formatTime(totalDuration)}</span>
          <input type="range" min="0" max={Math.max(totalDuration, 0)} step="0.01" value={Math.min(currentTime, totalDuration)} onChange={(event) => seekTo(Number(event.target.value))} disabled={notes.length === 0} />
        </label>
      </section>

      <section className="editor-section" style={glassPanelStyle}>
        <div className="section-heading">
          <span>03</span>
          <div><h2>{copy.editor}</h2><p>{copy.editorCopy}</p></div>
        </div>

        <div
          className="piano-roll-toolbar"
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 8,
            marginBottom: 12,
            ...glassCardStyle,
          }}
        >
          <strong className="selection-status" style={{ minWidth: 110 }}>
            {selectedNoteIds.size > 0
              ? `${selectedNoteIds.size} ${copy.selected}`
              : copy.noSelection}
          </strong>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            {copy.tempo}
            <input
              type="number"
              min="20"
              max="300"
              step="0.1"
              value={tempoBpm}
              onChange={(event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value)) {
                  setTempoBpm(Math.min(300, Math.max(20, value)));
                }
              }}
              aria-label="Grid tempo in beats per minute"
              style={{ width: 66 }}
            />
            BPM
          </label>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            {copy.meter}
            <select
              value={`${timeSignature.numerator}/${timeSignature.denominator}`}
              onChange={(event) => selectMeter(event.target.value)}
              aria-label="Time signature meter"
            >
              {MUSICAL_METER_PRESETS.map((preset) => (
                <option
                  key={`${preset.numerator}/${preset.denominator}`}
                  value={`${preset.numerator}/${preset.denominator}`}
                >
                  {preset.label}
                </option>
              ))}
            </select>
          </label>

          <span aria-hidden="true" style={{ color: "#9ca3af" }}>
            │
          </span>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            {copy.timeZoom}
            <button
              onClick={() =>
                setHorizontalZoom((value) => Math.max(0.15, value - 0.1))
              }
              aria-label="Zoom out horizontally"
            >
              −
            </button>
            <input
              type="range"
              min="0.15"
              max="4"
              step="0.05"
              value={horizontalZoom}
              onChange={(event) => setHorizontalZoom(Number(event.target.value))}
              aria-label="Horizontal piano roll zoom"
            />
            <button
              onClick={() =>
                setHorizontalZoom((value) => Math.min(4, value + 0.1))
              }
              aria-label="Zoom in horizontally"
            >
              +
            </button>
          </label>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            {copy.pitchZoom}
            <button
              onClick={() =>
                setVerticalZoom((value) => Math.max(0.75, value - 0.25))
              }
              aria-label="Zoom out vertically"
            >
              −
            </button>
            <input
              type="range"
              min="0.75"
              max="2.5"
              step="0.25"
              value={verticalZoom}
              onChange={(event) => setVerticalZoom(Number(event.target.value))}
              aria-label="Vertical piano roll zoom"
            />
            <button
              onClick={() =>
                setVerticalZoom((value) => Math.min(2.5, value + 0.25))
              }
              aria-label="Zoom in vertically"
            >
              +
            </button>
          </label>

          <span aria-hidden="true" style={{ color: "#9ca3af" }}>
            │
          </span>

          <label
            title="Sets the snap resolution used when dragging notes and when quantizing selected notes."
            style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
          >
            {copy.snapQuantize}
            <select
              value={quantizeDivision}
              onChange={(event) => setQuantizeDivision(Number(event.target.value))}
              aria-label="Snap and quantize resolution"
            >
              <option value="1">1/4</option>
              <option value="0.5">1/8</option>
              <option value="0.25">1/16</option>
              <option value="0.125">1/32</option>
              <option value="0.0625">1/64</option>
            </select>
          </label>
          <label
            className="velocity-control"
            style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
          >
            {copy.velocity}
            <input
              type="range"
              min="1"
              max="127"
              step="1"
              value={selectedVelocity}
              onChange={(event) => setSelectedVelocity(Number(event.target.value))}
              aria-label="Selected notes velocity"
              disabled={selectedNoteIds.size === 0}
            />
            <input
              type="number"
              min="1"
              max="127"
              step="1"
              value={selectedVelocity}
              onChange={(event) => setSelectedVelocity(Number(event.target.value))}
              aria-label="Selected notes velocity value"
              disabled={selectedNoteIds.size === 0}
              style={{ width: 48 }}
            />
          </label>
          <button
            onClick={quantizeNotes}
            disabled={notes.length === 0}
            title="Snap the selected notes to the resolution chosen above"
          >
            {copy.quantize}
          </button>
          <button
            onClick={() => nudgeSelectedNotes(-1)}
            disabled={selectedNoteIds.size === 0}
            title="Move selected notes one grid division earlier"
          >
            {copy.nudgeBack}
          </button>
          <button
            onClick={() => nudgeSelectedNotes(1)}
            disabled={selectedNoteIds.size === 0}
            title="Move selected notes one grid division later"
          >
            {copy.nudgeForward}
          </button>
          <button
            onClick={duplicateSelectedNotes}
            disabled={selectedNoteIds.size === 0}
            title="Duplicate selected notes one grid division later"
          >
            {copy.duplicate}
          </button>
          <button
            onClick={copySelectedNotes}
            disabled={selectedNoteIds.size === 0}
            title="Copy selected notes (Cmd/Ctrl+C)"
          >
            {copy.copy}
          </button>
          <button
            onClick={pasteCopiedNotes}
            disabled={clipboardNoteCount === 0}
            title="Paste notes at the playhead (Cmd/Ctrl+V)"
          >
            {copy.paste}
          </button>

          <button onClick={selectAllNotes} disabled={notes.length === 0}>{copy.selectAll}</button>
          <button onClick={clearSelection} disabled={selectedNoteIds.size === 0}>
            {copy.clear}
          </button>
          <button
            onClick={() => transposeSelectedNotes(-12)}
            disabled={selectedNoteIds.size === 0}
            title="Move selected notes down one octave"
          >
            {copy.octaveDown}
          </button>
          <button
            onClick={() => transposeSelectedNotes(12)}
            disabled={selectedNoteIds.size === 0}
            title="Move selected notes up one octave"
          >
            {copy.octaveUp}
          </button>
          <button
            onClick={deleteSelectedNotes}
            disabled={selectedNoteIds.size === 0}
          >
            {copy.delete}
          </button>
          <button onClick={undoEdit} disabled={pastNotes.length === 0}>{copy.undo}</button>
          <button onClick={redoEdit} disabled={futureNotes.length === 0}>{copy.redo}</button>
          <button onClick={exportEditedMidi} disabled={notes.length === 0}>{copy.exportMidi}</button>
        </div>

        <p className="piano-roll-help" style={{ marginTop: 0, fontSize: 14 }}>
          {copy.help}
        </p>

        {exportStatus && (
          <p
            role="status"
            style={{ marginTop: 0, color: exportStatus.startsWith("Standard") ? "#b91c1c" : "#166534" }}
          >
            {exportStatus}
          </p>
        )}

        <div
          role="region"
          tabIndex={0}
          aria-label={copy.pianoRoll}
          onClick={handlePianoRollClick}
          onPointerDown={beginMarqueeSelection}
          onPointerMove={handleMarqueeSelectionMove}
          onPointerUp={finishMarqueeSelection}
          onPointerCancel={cancelMarqueeSelection}
          onKeyDown={handlePianoRollKeyDown}
          className="piano-roll-surface"
          style={{
            width: "100%",
            overflow: "auto",
            border: "1px solid rgba(93, 112, 136, 0.28)",
            borderRadius: 20,
            background: "rgba(255, 255, 255, 0.38)",
            height: 480,
            position: "relative",
            outlineOffset: 3,
            cursor: notes.length > 0 ? "crosshair" : "default",
            ...glassPanelStyle,
          }}
        >
          <div
            style={{
              width: PIANO_KEY_WIDTH + pianoRollWidth,
              height: RULER_HEIGHT + pianoRollHeight,
              position: "relative",
            }}
          >
            <div
              aria-hidden="true"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: RULER_HEIGHT,
                background: "rgba(255, 255, 255, 0.72)",
                borderBottom: "1px solid rgba(93, 112, 136, 0.28)",
                color: "#17202e",
                zIndex: 25,
                pointerEvents: "none",
                backdropFilter: "blur(18px) saturate(170%)",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  width: PIANO_KEY_WIDTH,
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRight: "1px solid rgba(93, 112, 136, 0.3)",
                  color: "#5f6c7d",
                  fontSize: 10,
                  fontWeight: 700,
                }}
              >
                BAR
              </div>

              {barMarkers.map(({ bar, startSeconds }) => (
                <span
                  key={`bar-label-${bar}`}
                  style={{
                    position: "absolute",
                    left: PIANO_KEY_WIDTH + secondsToX(startSeconds) + 6,
                    top: 7,
                    color: "#253246",
                    fontSize: 11,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {bar}
                </span>
              ))}
            </div>

            {visiblePitches.map((pitch) => (
              <div
                key={`row-${pitch}`}
                style={{
                  position: "absolute",
                  left: PIANO_KEY_WIDTH,
                  right: 0,
                  top: RULER_HEIGHT + pitchToY(pitch),
                  height: noteRowHeight,
                  background: isBlackKey(pitch)
                    ? "rgba(214, 223, 234, 0.5)"
                    : "rgba(247, 250, 253, 0.58)",
                  borderBottom:
                    pitch % 12 === 0
                      ? "1px solid rgba(93, 112, 136, 0.34)"
                      : "1px solid rgba(93, 112, 136, 0.16)",
                  pointerEvents: "none",
                }}
              />
            ))}

            {barMarkers.flatMap(({ bar, startSeconds }) =>
              Array.from(
                { length: timeSignature.numerator },
                (_, beat) => ({
                  bar,
                  beat,
                  seconds: startSeconds + beat * secondsPerGridBeat,
                })
              )
            ).map(({ bar, beat, seconds }) => (
              <div
                key={`grid-${bar}-${beat}`}
                style={{
                  position: "absolute",
                  left: PIANO_KEY_WIDTH + secondsToX(seconds),
                  top: RULER_HEIGHT,
                  width: 1,
                  height: pianoRollHeight,
                  background:
                    beat === 0
                      ? "rgba(0, 122, 255, 0.28)"
                      : "rgba(93, 112, 136, 0.18)",
                  pointerEvents: "none",
                  zIndex: 1,
                }}
              />
            ))}

            {marqueeBox && (
              <div
                aria-hidden="true"
                style={{
                  position: "absolute",
                  left: marqueeBox.left,
                  top: marqueeBox.top,
                  width: marqueeBox.width,
                  height: marqueeBox.height,
                  border: "1px solid rgba(0, 122, 255, 0.86)",
                  background: "rgba(0, 122, 255, 0.14)",
                  boxShadow: "0 0 0 1px rgba(255, 255, 255, 0.64)",
                  pointerEvents: "none",
                  zIndex: 22,
                }}
              />
            )}

            <div
              aria-hidden="true"
              style={{
                position: "sticky",
                left: 0,
                top: RULER_HEIGHT,
                marginTop: RULER_HEIGHT,
                width: PIANO_KEY_WIDTH,
                height: pianoRollHeight,
                background: "rgba(255, 255, 255, 0.76)",
                borderRight: "1px solid rgba(93, 112, 136, 0.34)",
                zIndex: 30,
                backdropFilter: "blur(18px) saturate(170%)",
              }}
            >
              {visiblePitches.map((pitch) => {
                const drumName = mode === "drums" ? GM_DRUM_LABELS[pitch] : undefined;
                const black = mode !== "drums" && isBlackKey(pitch);
                const noteName = Tone.Frequency(pitch, "midi").toNote();
                const keyLabel = drumName ?? (mode === "drums" ? "Perc." : noteName);

                return (
                  <div
                    key={`key-${pitch}`}
                    style={{
                      position: "absolute",
                      left: 0,
                      top: pitchToY(pitch),
                      width: black ? PIANO_KEY_WIDTH - 14 : PIANO_KEY_WIDTH,
                      height: noteRowHeight,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: mode === "drums" ? "space-between" : "flex-end",
                      gap: 5,
                      paddingRight: 7,
                      paddingLeft: mode === "drums" ? 8 : 0,
                      boxSizing: "border-box",
                      borderBottom: "1px solid rgba(93, 112, 136, 0.24)",
                      borderRight: black ? "5px solid #1f2937" : "none",
                      background: mode === "drums"
                        ? drumName
                          ? "rgba(255, 255, 255, 0.78)"
                          : "rgba(241, 244, 248, 0.58)"
                        : black
                        ? "#303846"
                        : "rgba(255, 255, 255, 0.72)",
                      color: black ? "#ffffff" : drumName ? "#17202e" : "#7a8798",
                      fontSize: mode === "drums" ? 9 : 10,
                      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <span
                      style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {keyLabel}
                    </span>
                    <strong>{pitch}</strong>
                  </div>
                );
              })}
            </div>

            <div
              style={{
                position: "absolute",
                left: PIANO_KEY_WIDTH + secondsToX(currentTime),
                top: RULER_HEIGHT,
                width: 2,
                height: pianoRollHeight,
                background: "#ff375f",
                boxShadow: "0 0 0 1px rgba(255, 255, 255, 0.72), 0 0 18px rgba(255, 55, 95, 0.32)",
                zIndex: 20,
                pointerEvents: "none",
              }}
            />

            {notes.map((note, index) => {
              const x = secondsToX(note.time);
              const y = pitchToY(note.midi);
              const width = Math.max(
                (note.duration / secondsPerQuarterBeat) * pixelsPerBeat,
                8
              );
              const isSelected = selectedNoteIds.has(note.id);

              return (
                <button
                  type="button"
                  data-piano-note
                  key={note.id}
                  aria-label={`${note.name}, ${note.time.toFixed(2)} seconds, ${note.duration.toFixed(2)} seconds long`}
                  aria-pressed={isSelected}
                  title={`${note.name} · ${note.time.toFixed(2)}s`}
                  onClick={(event) => handleNoteSelect(event, note.id, index)}
                  onPointerDown={(event) =>
                    beginNotePointerEdit(event, note.id, "move")
                  }
                  onPointerMove={handleNotePointerMove}
                  onPointerUp={finishNotePointerEdit}
                  onPointerCancel={cancelNotePointerEdit}
                  style={{
                    position: "absolute",
                    left: PIANO_KEY_WIDTH + x,
                    top:
                      RULER_HEIGHT +
                      y +
                      noteVerticalInset,
                    width,
                    height: noteBarHeight,
                    borderRadius: 3,
                    border: isSelected
                      ? "1px solid rgba(255, 255, 255, 0.86)"
                      : "1px solid rgba(0, 95, 203, 0.5)",
                    padding: 0,
                    overflow: "hidden",
                    background: isSelected
                      ? "#ff9f0a"
                      : "#34c759",
                    boxShadow: isSelected
                      ? "inset 0 1px 0 rgba(255, 255, 255, 0.46)"
                      : "inset 0 1px 0 rgba(255, 255, 255, 0.38)",
                    color: "#052e16",
                    opacity: 1,
                    cursor: "grab",
                    touchAction: "none",
                    zIndex: isSelected ? 12 : 10,
                  }}
                >
                  {width >= 36 && (
                    <span
                      aria-hidden="true"
                      style={{
                        position: "absolute",
                        left: 4,
                        top: 0,
                        maxWidth: "calc(100% - 11px)",
                        overflow: "hidden",
                        color: "#052e16",
                        fontSize: 9,
                        fontWeight: 700,
                        lineHeight: `${Math.max(8, noteBarHeight - 2)}px`,
                        pointerEvents: "none",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {note.name}
                    </span>
                  )}

                  <span
                    data-resize-handle
                    aria-hidden="true"
                    onClick={(event) => {
                      event.stopPropagation();
                      suppressNoteClickRef.current = false;
                    }}
                    onPointerDown={(event) =>
                      beginNotePointerEdit(event, note.id, "resize-end")
                    }
                    style={{
                      position: "absolute",
                      top: 0,
                      right: 0,
                      width: 7,
                      height: "100%",
                      borderLeft: isSelected
                        ? "1px solid rgba(120, 53, 15, 0.45)"
                        : "1px solid rgba(5, 46, 22, 0.34)",
                      borderRadius: "0 3px 3px 0",
                      background: isSelected
                        ? "rgba(255, 255, 255, 0.38)"
                        : "rgba(255, 255, 255, 0.26)",
                      cursor: "ew-resize",
                    }}
                  />
                </button>
              );
            })}
          </div>
        </div>
      </section>

    </main>
  );
}
