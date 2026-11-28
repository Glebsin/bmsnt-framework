//! Engine log: file sink behind a mutex, initialized once by the binary's
//! main (osu-framework `Logging/` analogue).

/// log file handle behind a mutex; initialized once by the binary's main
pub static LOG_SINK: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

/// Redirect the log to `file_name` next to the exe. Errors are ignored
/// (a missing log must not kill the app).
pub fn init_log(file_name: &str) {
    let path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(file_name)));
    if let Some(path) = &path {
        if let Ok(file) = std::fs::File::create(path) {
            *LOG_SINK.lock().unwrap() = Some(file);
        }
    }
}

/// Log one line to the sink (no-op when the sink is not initialized).
#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {{
        use std::io::Write;
        if let Ok(mut f) = $crate::logging::LOG_SINK.lock() {
            if let Some(file) = f.as_mut() {
                let _ = writeln!(file, $($arg)*);
            }
        }
    }};
}
