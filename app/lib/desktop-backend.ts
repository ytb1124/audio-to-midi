import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type DesktopMode = "piano" | "bass" | "drums" | "general";

type BackendEvent = {
  jobId: string;
  event: {
    type?: "progress" | "completed" | "error";
    progress?: number;
    status?: string;
    mode?: string;
    midiPath?: string;
    midiFileName?: string;
    stems?: Record<string, string>;
    model?: string;
    device?: string;
    error?: string;
  };
};

export type DesktopTranscriptionResult = {
  mode: DesktopMode;
  midiPath: string;
  midiFileName: string;
  drumsStemPath?: string;
  counts?: Record<string, number>;
};

export type DesktopSeparationResult = {
  stems: Record<string, string>;
  model?: string;
  device?: string;
  mode: "separate";
};

export type DesktopProgressHandler = (progress: number, status: string) => void;

/** True only inside a Tauri WebView. Browser development remains HTTP-based. */
export function isDesktopRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function desktopAssetUrl(path: string) {
  return convertFileSrc(path);
}

async function runPythonJob<T>(
  request: {
    kind: "transcribe" | "separate";
    mode?: DesktopMode;
    file: File;
  },
  onProgress?: DesktopProgressHandler
): Promise<T> {
  const eventName = "trackform://backend-event";
  let unlisten: UnlistenFn | undefined;
  let settled = false;

  const result = new Promise<T>(async (resolve, reject) => {
    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      unlisten?.();
      callback();
    };

    try {
      unlisten = await listen<BackendEvent>(eventName, (event) => {
        const payload = event.payload;
        const message = payload.event;
        if (!message || message.type === undefined) return;

        if (message.type === "progress") {
          onProgress?.(message.progress ?? 0, message.status ?? "");
          return;
        }

        if (message.type === "error") {
          finish(() => reject(new Error(message.error || "Desktop backend failed.")));
          return;
        }

        if (message.type === "completed") {
          finish(() => resolve(message as T));
        }
      });

      const bytes = Array.from(new Uint8Array(await request.file.arrayBuffer()));
      await invoke("start_python_job", {
        kind: request.kind,
        mode: request.mode ?? null,
        fileName: request.file.name,
        fileBytes: bytes,
      });
    } catch (error) {
      finish(() => reject(error instanceof Error ? error : new Error(String(error))));
    }
  });

  return result;
}

export function runDesktopTranscription(
  file: File,
  mode: DesktopMode,
  onProgress?: DesktopProgressHandler
) {
  return runPythonJob<DesktopTranscriptionResult>(
    { kind: "transcribe", mode, file },
    onProgress
  );
}

export function runDesktopSeparation(file: File, onProgress?: DesktopProgressHandler) {
  return runPythonJob<DesktopSeparationResult>({ kind: "separate", file }, onProgress);
}
