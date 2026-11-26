use std::{env, fs, path::Path};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // compile both shaders: the text/quad pipeline and the sprite batcher
    for (src, out) in [("src/shader.wgsl", "shader.spv"), ("src/sprite.wgsl", "sprite.spv")] {
        let src_text = fs::read_to_string(src).unwrap_or_else(|_| panic!("read {src}"));
        let module = naga::front::wgsl::parse_str(&src_text).unwrap_or_else(|e| panic!("parse {src}: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        let info = validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("validate {src}: {e:?}"));
        let options = naga::back::spv::Options::default();
        let spv = naga::back::spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("spv {src}: {e:?}"));

        let mut bytes = Vec::with_capacity(spv.len() * 4);
        for w in &spv {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        fs::write(Path::new(&out_dir).join(out), &bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
        println!("cargo:rerun-if-changed={src}");
    }

    // Windows manifest for the testapp binary only: requireAdministrator so
    // NVML clock locking works on a normal double-click. Compiled with the
    // SDK resource compiler (rc.exe) and linked into testapp.exe via a
    // per-binary link arg (vk5k.exe stays without the elevation requirement).
    // Fallback: without rc.exe the build continues without the manifest.
    println!("cargo:rerun-if-changed=src/bin/testapp/testapp.manifest");
    println!("cargo:rerun-if-changed=src/bin/testapp/testapp.rc");
    // measurement builds launched from a non-elevated automation session
    // skip the manifest (env switch, not part of normal builds)
    println!("cargo:rerun-if-env-changed=TESTAPP_NO_MANIFEST");
    let rc_script = "src/bin/testapp/testapp.rc";
    if env::var("TESTAPP_NO_MANIFEST").is_ok() {
        println!("cargo:warning=TESTAPP_NO_MANIFEST set: building testapp WITHOUT the admin manifest (measurement build only)");
    } else if Path::new(rc_script).exists() {
        if let Some(rc) = find_rc() {
            let res = Path::new(&out_dir).join("testapp.res");
            let status = std::process::Command::new(&rc)
                .args(["/nologo", "/fo", res.to_str().unwrap(), rc_script])
                .status();
            match status {
                Ok(s) if s.success() => {
                    println!("cargo:rustc-link-arg-bin=testapp={}", res.to_str().unwrap());
                }
                _ => panic!(
                    "rc.exe failed to compile testapp.rc; refusing to build testapp.exe without the admin manifest"
                ),
            }
        } else {
            panic!(
                "rc.exe not found: cannot embed the requireAdministrator manifest into testapp.exe; \
                 install the Windows SDK or set the RC env variable to the full path of rc.exe"
            );
        }
    }
}

/// Locate the Windows SDK resource compiler (newest kit first).
fn find_rc() -> Option<std::path::PathBuf> {
    if let Ok(rc) = env::var("RC") {
        if Path::new(&rc).exists() {
            return Some(rc.into());
        }
    }
    let root = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut kits: Vec<_> = fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("x64\\rc.exe").exists())
        .map(|e| e.path())
        .collect();
    kits.sort();
    kits.pop().map(|p| p.join("x64\\rc.exe"))
}
