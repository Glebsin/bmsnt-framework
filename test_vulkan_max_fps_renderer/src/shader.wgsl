struct VSIn {
    @location(0) pos: vec2f,
    @location(1) col: vec3f,
};

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) col: vec3f,
};

@vertex
fn vs_main(in: VSIn) -> VSOut {
    var out: VSOut;
    out.pos = vec4f(in.pos, 0.0, 1.0);
    out.col = in.col;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    return vec4f(in.col, 1.0);
}
