use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

#[derive(Debug, Serialize, Clone)]
struct BackendEvent {
  job_id: String,
  event: Value,
}

fn emit_backend_event(app: &AppHandle, job_id: &str, event: Value) {
  let payload = BackendEvent {
    job_id: job_id.to_string(),
    event,
  };
  if let Err(error) = app.emit("trackform://backend-event", payload) {
    eprintln!("Failed to emit backend event: {error}");
  }
}

fn safe_file_name(file_name: &str) -> String {
  Path::new(file_name)
    .file_name()
    .and_then(|name| name.to_str())
    .filter(|name| !name.is_empty())
    .unwrap_or("input.audio")
    .to_string()
}

fn bundled_sidecar(app: &AppHandle, project_root: &Path) -> Option<std::path::PathBuf> {
  let candidates = [
    project_root.join("dist/trackform-backend/trackform-backend"),
    project_root.join(
      "src-tauri/binaries/trackform-backend-aarch64-apple-darwin/trackform-backend",
    ),
    app.path()
      .resource_dir()
      .ok()?
      .join("_up_/dist/trackform-backend/trackform-backend"),
  ];
  candidates.into_iter().find(|candidate| candidate.is_file())
}

#[tauri::command]
fn start_python_job(
  app: AppHandle,
  kind: String,
  mode: Option<String>,
  file_name: String,
  file_bytes: Vec<u8>,
) -> Result<String, String> {
  if kind != "transcribe" && kind != "separate" {
    return Err(format!("Unsupported backend job: {kind}"));
  }

  let mode = if kind == "transcribe" {
    let value = mode.ok_or_else(|| "A transcription mode is required.".to_string())?;
    match value.as_str() {
      "piano" | "bass" | "drums" | "general" => Some(value),
      _ => return Err(format!("Unsupported transcription mode: {value}")),
    }
  } else {
    None
  };

  let project_root = std::env::current_dir()
    .map_err(|error| format!("Could not locate the project directory: {error}"))?;
  let python = project_root.join(".venv-backend/bin/python");
  let cli = project_root.join("backend/cli.py");
  let sidecar = bundled_sidecar(&app, &project_root);
  if sidecar.is_none() && (!python.is_file() || !cli.is_file()) {
    return Err(format!(
      "Trackform sidecar not found and development Python is unavailable. Expected sidecar under {}/dist/trackform-backend.",
      project_root.display()
    ));
  }

  let nonce = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_err(|error| format!("Could not create job id: {error}"))?
    .as_nanos();
  let job_id = format!("desktop-{nonce}");
  let job_dir = std::env::temp_dir().join("trackform").join(&job_id);
  let output_dir = job_dir.join("outputs");
  fs::create_dir_all(&output_dir)
    .map_err(|error| format!("Could not create job directory: {error}"))?;
  let input_path = job_dir.join(safe_file_name(&file_name));
  fs::write(&input_path, file_bytes)
    .map_err(|error| format!("Could not write the selected audio file: {error}"))?;

  let mut command = if let Some(sidecar) = sidecar {
    let mut command = Command::new(sidecar);
    command.arg(&kind);
    command
  } else {
    let mut command = Command::new(&python);
    command.arg(&cli).arg(&kind);
    command
  };
  command
    .current_dir(&project_root)
    .env("PYTHONUNBUFFERED", "1")
    .arg("--input")
    .arg(&input_path)
    .arg("--output-dir")
    .arg(&output_dir)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

  if let Some(mode) = mode {
    command.arg("--mode").arg(mode);
  }

  let mut child = command
    .spawn()
    .map_err(|error| format!("Could not start Trackform CLI: {error}"))?;
  let stdout = child
    .stdout
    .take()
    .ok_or_else(|| "Trackform CLI stdout was not available.".to_string())?;
  let stderr = child.stderr.take();
  let app_for_job = app.clone();
  let job_id_for_job = job_id.clone();

  std::thread::spawn(move || {
    if let Some(stderr) = stderr {
      std::thread::spawn(move || {
        let mut text = String::new();
        let mut reader = BufReader::new(stderr);
        if reader.read_to_string(&mut text).is_ok() && !text.trim().is_empty() {
          eprintln!("Trackform CLI stderr:\n{text}");
        }
      });
    }

    let mut saw_terminal_event = false;
    for line in BufReader::new(stdout).lines() {
      match line {
        Ok(line) if line.trim().is_empty() => continue,
        Ok(line) => match serde_json::from_str::<Value>(&line) {
          Ok(event) => {
            if matches!(event.get("type").and_then(Value::as_str), Some("completed" | "error")) {
              saw_terminal_event = true;
            }
            emit_backend_event(&app_for_job, &job_id_for_job, event);
          }
          Err(error) => eprintln!("Ignoring non-JSON CLI stdout: {error}: {line}"),
        },
        Err(error) => eprintln!("Could not read Trackform CLI stdout: {error}"),
      }
    }

    match child.wait() {
      Ok(status) if !status.success() && !saw_terminal_event => emit_backend_event(
        &app_for_job,
        &job_id_for_job,
        json!({"type": "error", "error": format!("Trackform CLI exited with {status}")}),
      ),
      Err(error) => emit_backend_event(
        &app_for_job,
        &job_id_for_job,
        json!({"type": "error", "error": format!("Could not wait for Trackform CLI: {error}")}),
      ),
      _ => {}
    }
  });

  Ok(job_id)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![start_python_job])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
