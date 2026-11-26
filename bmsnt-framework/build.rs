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

    // Windows manifest for the bmsnt_test_browser binary only: requireAdministrator so
    // NVML clock locking works on a normal double-click. Compiled with the
    // SDK resource compiler (rc.exe) and linked into bmsnt_test_browser.exe via a
    // per-binary link arg (bmsnt-framework.exe stays without the elevation requirement).
    // A missing rc.exe is a hard error: the manifest is required for normal
    // use; measurement builds may skip it with BMSNT_NO_MANIFEST=1.
    println!("cargo:rerun-if-changed=src/bin/bmsnt_test_browser/bmsnt_test_browser.manifest");
    println!("cargo:rerun-if-changed=src/bin/bmsnt_test_browser/bmsnt_test_browser.rc");
    // measurement builds launched from a non-elevated automation session
    // skip the manifest (env switch, not part of normal builds)
    println!("cargo:rerun-if-env-changed=BMSNT_NO_MANIFEST");
    let rc_script = "src/bin/bmsnt_test_browser/bmsnt_test_browser.rc";
    if env::var("BMSNT_NO_MANIFEST").as_deref() == Ok("1") {
        println!("cargo:warning=BMSNT_NO_MANIFEST=1: building bmsnt_test_browser WITHOUT the admin manifest (measurement build only)");
    } else if !Path::new(rc_script).exists() {
        panic!(
            "src/bin/bmsnt_test_browser/bmsnt_test_browser.rc is missing: cannot embed the requireAdministrator manifest; \
             restore the file or build with BMSNT_NO_MANIFEST=1"
        );
    } else if Path::new(rc_script).exists() {
        if let Some(rc) = find_rc() {
            let res = Path::new(&out_dir).join("bmsnt_test_browser.res");
            let status = std::process::Command::new(&rc)
                .args(["/nologo", "/fo", res.to_str().unwrap(), rc_script])
                .status();
            match status {
                Ok(s) if s.success() => {
                    println!("cargo:rustc-link-arg-bin=bmsnt_test_browser={}", res.to_str().unwrap());
                }
                _ => panic!(
                    "rc.exe failed to compile bmsnt_test_browser.rc; refusing to build bmsnt_test_browser.exe without the admin manifest"
                ),
            }
        } else {
            panic!(
                "rc.exe not found: cannot embed the requireAdministrator manifest into bmsnt_test_browser.exe; \
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
