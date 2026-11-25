use std::{env, fs, path::Path};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let src = fs::read_to_string("src/shader.wgsl").expect("read src/shader.wgsl");

    let module = naga::front::wgsl::parse_str(&src).expect("parse WGSL");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let info = validator.validate(&module).expect("validate WGSL");
    let options = naga::back::spv::Options::default();
    let spv = naga::back::spv::write_vec(&module, &info, &options, None).expect("write SPIR-V");

    let mut bytes = Vec::with_capacity(spv.len() * 4);
    for w in &spv {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    fs::write(Path::new(&out_dir).join("shader.spv"), &bytes).expect("write shader.spv");

    println!("cargo:rerun-if-changed=src/shader.wgsl");
}
