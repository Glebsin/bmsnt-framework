// Sprite batcher shader: instanced quads sampled from a texture atlas,
// plus a special "gradient" mode with per-pixel dithering to kill banding
// (dithering hides banding on subtle gradients).
//
// Instance attributes (binding 0, instance rate, 80 bytes):
//   loc0: pos.xy, size.xy              (pixels, top-left origin)
//   loc1: uv0.xy, uv1.xy               (atlas uv corners; gradient: t0, t1)
//   loc2: col_top.rgba                 (textured mode: tint; gradient: top color)
//   loc3: col_bot.rgba                 (gradient: bottom color)
//   loc4: mode, res.x, res.y, unused   (mode 0 = textured, 1 = gradient)

@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) col_top: vec4f,
    @location(2) col_bot: vec4f,
    @location(3) params: vec4f,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) pos_size: vec4f,
    @location(1) uv01: vec4f,
    @location(2) col_top: vec4f,
    @location(3) col_bot: vec4f,
    @location(4) params: vec4f,
) -> VsOut {
    // two triangles from vertex index only: no vertex buffer needed
    let corners = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
        vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0),
    );
    let c = corners[vi];
    let p = pos_size.xy + c * pos_size.zw;
    let res = vec2f(params.y, params.z);
    let ndc = vec2f(p.x / res.x * 2.0 - 1.0, 1.0 - p.y / res.y * 2.0);

    var out: VsOut;
    out.pos = vec4f(ndc, 0.0, 1.0);
    out.uv = mix(uv01.xy, uv01.zw, c);
    out.col_top = col_top;
    out.col_bot = col_bot;
    out.params = params;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    if in.params.x < 0.5 {
        // textured mode: atlas sample tinted by col_top
        let tex = textureSample(atlas, atlas_sampler, in.uv);
        return tex * in.col_top;
    }
    // gradient mode: vertical mix between col_top and col_bot
    var col = mix(in.col_top.rgb, in.col_bot.rgb, in.uv.y);
    // per-pixel dither noise of ~1/255 amplitude
    let n = fract(sin(dot(in.pos.xy, vec2f(12.9898, 78.233))) * 43758.5453);
    col = col + (n - 0.5) / 255.0;
    return vec4f(col, 1.0);
}
