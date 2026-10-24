import * as Tone from "tone";

/** A pitch-bend point relative to the note start (seconds), in cents. */
export type BendPoint = { time: number; cents: number };

type ManifestLayer = { maxVel: number; notes: Record<string, string> };
type ManifestEntry = {
  dir: string;
  layers: ManifestLayer[];
  release?: number;
  attack?: number;
  drum?: boolean;
  filterHz?: number;
};

export type SampleInstrumentKey =
  | "piano"
  | "electric_piano"
  | "electric_bass"
  | "upright_bass"
  | "drums"
  | "strings"
  | "pad";

const SAMPLE_ROOT = "/samples/";
let manifestPromise: Promise<Record<string, ManifestEntry>> | null = null;

function loadManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(`${SAMPLE_ROOT}manifest.json`).then((response) => {
      if (!response.ok) throw new Error("Sample manifest is missing.");
      return response.json();
    });
  }
  return manifestPromise;
}

type Layer = { maxVel: number; keys: number[]; buffers: Map<number, Tone.ToneAudioBuffer> };

// Open hi-hat is choked by closed or pedal hi-hat, as on a real kit.
const HIHAT_CHOKERS = new Set([42, 44]);
const OPEN_HIHAT = 46;
const DRUM_FALLBACK = 38;

/**
 * Offline multi-sample instrument (samples live in public/samples, see LICENSES.md there).
 * Picks the velocity layer, plays the nearest sampled pitch at the needed rate and applies
 * pitch bends by automating the voice's playback rate.
 */
export class MultiSampleInstrument {
  private layers: Layer[] = [];
  private entry: ManifestEntry | null = null;
  private output: Tone.Gain;
  private chain: Tone.ToneAudioNode[] = [];
  private openHihat: { source: Tone.ToneBufferSource; env: Tone.Gain } | null = null;

  constructor(private key: SampleInstrumentKey) {
    this.output = new Tone.Gain(0.9);
  }

  async load() {
    const manifest = await loadManifest();
    const entry = manifest[this.key];
    if (!entry) throw new Error(`Unknown instrument: ${this.key}`);
    this.entry = entry;
    this.buildChain(entry);
    this.layers = await Promise.all(
      entry.layers.map(async (layer) => {
        const buffers = new Map<number, Tone.ToneAudioBuffer>();
        await Promise.all(
          Object.entries(layer.notes).map(async ([midi, file]) => {
            const buffer = await new Tone.ToneAudioBuffer().load(`${SAMPLE_ROOT}${entry.dir.replace(/^samples\//, "")}${file}`);
            buffers.set(Number(midi), buffer);
          })
        );
        return { maxVel: layer.maxVel, keys: [...buffers.keys()].sort((a, b) => a - b), buffers };
      })
    );
  }

  private buildChain(entry: ManifestEntry) {
    const nodes: Tone.ToneAudioNode[] = [];
    if (entry.filterHz) nodes.push(new Tone.Filter({ frequency: entry.filterHz, type: "lowpass", rolloff: -24 }));
    if (this.key === "piano" || this.key === "electric_piano" || this.key === "strings" || this.key === "pad") {
      nodes.push(new Tone.Reverb({ decay: this.key === "pad" ? 4 : 2.2, wet: this.key === "pad" ? 0.35 : 0.16 }));
    } else if (this.key === "drums") {
      nodes.push(new Tone.Reverb({ decay: 0.9, wet: 0.08 }));
    }
    this.chain = nodes;
    this.output.chain(...nodes, Tone.getDestination());
  }

  private pick(midi: number, velocity: number) {
    const vel127 = Math.round(Math.min(Math.max(velocity, 0), 1) * 127);
    const layer = this.layers.find((item) => vel127 <= item.maxVel) ?? this.layers[this.layers.length - 1];
    if (!layer || layer.keys.length === 0) return null;
    if (this.entry?.drum) {
      const key = layer.buffers.has(midi) ? midi : DRUM_FALLBACK;
      const buffer = layer.buffers.get(key);
      return buffer ? { buffer, sourceMidi: midi } : null;
    }
    let nearest = layer.keys[0];
    for (const key of layer.keys) {
      if (Math.abs(key - midi) < Math.abs(nearest - midi)) nearest = key;
    }
    return { buffer: layer.buffers.get(nearest)!, sourceMidi: nearest };
  }

  triggerAttackRelease(
    noteName: string,
    duration: number,
    time?: Tone.Unit.Time,
    velocity = 0.8,
    bend?: BendPoint[]
  ) {
    const entry = this.entry;
    if (!entry) return;
    const midi = Math.round(Tone.Frequency(noteName).toMidi());
    const choice = this.pick(midi, velocity);
    if (!choice) return;
    const when = Tone.Time(time ?? Tone.now()).toSeconds();
    const release = entry.release ?? 0.3;
    const attack = entry.attack ?? 0.003;
    const level = 0.3 + 0.7 * Math.min(Math.max(velocity, 0), 1);

    const env = new Tone.Gain(0).connect(this.output);
    const source = new Tone.ToneBufferSource(choice.buffer).connect(env);
    const baseSemitones = entry.drum ? 0 : midi - choice.sourceMidi;
    const rateFor = (cents: number) => Math.pow(2, (baseSemitones + cents / 100) / 12);
    source.playbackRate.value = rateFor(0);
    if (bend && !entry.drum) {
      for (const point of bend) {
        if (point.time >= 0 && point.time <= duration) {
          source.playbackRate.setValueAtTime(rateFor(point.cents), when + point.time);
        }
      }
    }

    if (entry.drum && HIHAT_CHOKERS.has(midi) && this.openHihat) {
      this.openHihat.env.gain.setTargetAtTime(0, when, 0.02);
    }

    env.gain.setValueAtTime(0, when);
    env.gain.linearRampToValueAtTime(level, when + attack);
    const releaseAt = when + Math.max(duration, attack);
    env.gain.setValueAtTime(level, releaseAt);
    env.gain.setTargetAtTime(0, releaseAt, release / 4);
    source.start(when);
    source.stop(releaseAt + release * 1.5);
    source.onended = () => {
      source.dispose();
      env.dispose();
      if (this.openHihat?.source === source) this.openHihat = null;
    };
    if (entry.drum && midi === OPEN_HIHAT) this.openHihat = { source, env };
  }

  dispose() {
    this.output.dispose();
    this.chain.forEach((node) => node.dispose());
    this.layers = [];
  }
}

/** Collect pitch bends from a @tonejs/midi track onto the notes they sound under (±2 semitone range). */
export function bendsForNotes(
  notes: { time: number; duration: number }[],
  pitchBends: { time: number; value: number }[],
  rangeSemitones = 2
): (BendPoint[] | undefined)[] {
  const result: (BendPoint[] | undefined)[] = notes.map(() => undefined);
  if (pitchBends.length === 0) return result;
  const order = notes.map((_, index) => index).sort((a, b) => notes[a].time - notes[b].time);
  for (const bendEvent of pitchBends) {
    let owner = -1;
    for (const index of order) {
      const note = notes[index];
      if (note.time > bendEvent.time + 1e-4) break;
      if (bendEvent.time < note.time + note.duration) owner = index;
    }
    if (owner < 0) continue;
    const points = result[owner] ?? [];
    points.push({ time: Math.max(0, bendEvent.time - notes[owner].time), cents: bendEvent.value * rangeSemitones * 100 });
    result[owner] = points;
  }
  return result.map((points) =>
    points && points.some((point) => Math.abs(point.cents) >= 1) ? points.sort((a, b) => a.time - b.time) : undefined
  );
}
