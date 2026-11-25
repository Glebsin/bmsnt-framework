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
}
