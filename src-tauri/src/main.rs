// Hides the extra console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Manager, RunEvent, State};

struct Backend {
    child: Mutex<Option<Child>>,
    port: u16,
    token: String,
}

#[tauri::command]
fn backend_info(state: State<Backend>) -> serde_json::Value {
    serde_json::json!({ "port": state.port, "token": state.token })
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("no free local port")
        .local_addr()
        .unwrap()
        .port()
}

/// Release: bundled PyInstaller build (resources/backend/rotobrush-backend.exe).
/// Dev: `python -m rotobrush` from ../backend (override the interpreter with ROTOBRUSH_PYTHON).
fn spawn_backend(app: &tauri::AppHandle, port: u16, token: &str) -> std::io::Result<Child> {
    let bundled: Option<PathBuf> = app
        .path()
        .resource_dir()
        .ok()
        .map(|d| d.join("backend").join("rotobrush-backend.exe"))
        .filter(|p| p.exists());

    let mut cmd = match bundled {
        Some(exe) => {
            let mut c = Command::new(&exe);
            c.current_dir(exe.parent().unwrap());
            c
        }
        None => {
            let py = std::env::var("ROTOBRUSH_PYTHON").unwrap_or_else(|_| "python".into());
            let mut c = Command::new(py);
            c.args(["-m", "rotobrush"]);
            c.current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../backend"));
            c
        }
    };
    cmd.args(["--port", &port.to_string()])
        .env("ROTOBRUSH_TOKEN", token)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd.spawn()
}

fn main() {
    let port = free_port();
    let token = uuid::Uuid::new_v4().simple().to_string();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![backend_info])
        .setup(move |app| {
            let child = spawn_backend(app.handle(), port, &token)
                .map_err(|e| format!("Could not start the processing backend: {e}. Is Python installed and on PATH?"))?;
            app.manage(Backend { child: Mutex::new(Some(child)), port, token: token.clone() });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building RotoBrush")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(b) = app.try_state::<Backend>() {
                    if let Some(mut c) = b.child.lock().unwrap().take() {
                        let _ = c.kill();
                    }
                }
            }
        });
}
