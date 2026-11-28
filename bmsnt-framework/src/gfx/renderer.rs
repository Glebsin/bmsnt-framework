//! Renderer: dynamic rendering (Vulkan 1.4 core), per-frame ring vertex
//! buffer, pre-recorded command buffers (one per swapchain image).
//!
//! Ring buffer: a single allocation cut into FRAMES_IN_FLIGHT slots; slot s
//! is read only by the CB of the frame submitted on slot s, and a slot is
//! reused only after ensure_idle — so wait_all before vertex writes is not
//! needed.

use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::ptr;

use ash::vk;

use super::device::GfxContext;
use super::sprite::SpriteBatcher;
use super::swapchain::Swapchain;
use crate::ui::text::Vertex;

/// per-slot vertex limit — shares the source of truth with ui::text
pub const MAX_VERTS: usize = crate::ui::text::MAX_VERTS;

pub struct Renderer {
    cmd_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    vert_mod: vk::ShaderModule,
    frag_mod: vk::ShaderModule,
    vbuf: vk::Buffer,
    vmem: vk::DeviceMemory,
    vmap: *mut Vertex,
    /// number of per-image regions the vertex allocation covers
    vert_regions: usize,
    /// allocation size in bytes (bounds-check base for region ranges)
    vbuf_bytes: u64,
    /// latest UI text vertex content (CPU snapshot; image regions are
    /// written per-image after their fence signals)
    verts: Vec<Vertex>,
    // swapchain-dependent
    views: Vec<vk::ImageView>,
    cmds: Vec<vk::CommandBuffer>,
    /// one fence per image: signals when that image's previous submit has
    /// fully completed — a CB must not be reset/re-recorded while still in
    /// flight (IMMEDIATE present does not throttle acquire)
    img_fences: Vec<vk::Fence>,
    cb_dirty: Vec<bool>,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    // in-app screenshots: host-visible buffer for the last presented image
    images: Vec<vk::Image>,
    shot_buf: vk::Buffer,
    shot_mem: vk::DeviceMemory,
    shot_map: *mut c_void,
    shot_extent: vk::Extent2D,
    shot_pending: bool,
    shot_ready: bool,
}

/// Bytes of one per-image vertex region.
const VERTEX_REGION_BYTES: u64 = (MAX_VERTS * std::mem::size_of::<Vertex>()) as u64;

/// Validate that region `img_idx` of `regions` regions of `region_bytes`
/// each lies fully inside a `buf_bytes` allocation. Called before ANY
/// pointer arithmetic on the mapped base.
pub(crate) fn check_region_index(
    img_idx: usize,
    regions: usize,
    region_bytes: u64,
    buf_bytes: u64,
) -> Result<()> {
    if img_idx >= regions {
        return Err(anyhow!(
            "region index {img_idx} out of range ({regions} regions)"
        ));
    }
    let end = (img_idx as u64 + 1) * region_bytes;
    if end > buf_bytes {
        return Err(anyhow!(
            "region {img_idx} end {end} exceeds buffer size {buf_bytes}"
        ));
    }
    Ok(())
}

/// Allocate the host-visible per-image vertex buffer: `regions` regions of
/// MAX_VERTS vertices each, mapped for CPU writes.
///
/// # Safety
///
/// `device` must be valid; the returned buffer/memory must be unmapped,
/// destroyed and freed (in that order) before the device is destroyed.
unsafe fn alloc_vertex_regions(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    regions: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut Vertex, u64)> {
    let buf_size = VERTEX_REGION_BYTES * regions as u64;
    let vb_info = vk::BufferCreateInfo {
        size: buf_size,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let vbuf = device
        .create_buffer(&vb_info, None)
        .map_err(|e| anyhow!("vbuf: {e:?}"))?;
    let fail_buf = |e: anyhow::Error| {
        unsafe { device.destroy_buffer(vbuf, None) };
        e
    };
    let mem_req = device.get_buffer_memory_requirements(vbuf);
    let mem_type = (0..mem_props.memory_type_count)
        .find(|i| {
            let t = mem_props.memory_types[*i as usize];
            mem_req.memory_type_bits & (1 << i) != 0
                && t.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .ok_or_else(|| fail_buf(anyhow!("host visible memory type")))?;
    let vmem = device
        .allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: mem_req.size,
                memory_type_index: mem_type,
                ..Default::default()
            },
            None,
        )
        .map_err(|e| fail_buf(anyhow!("vmem: {e:?}")))?;
    let fail_all = |e: anyhow::Error| {
        // buffer must be destroyed before its bound memory is freed
        unsafe { device.destroy_buffer(vbuf, None) };
        unsafe { device.free_memory(vmem, None) };
        e
    };
    device
        .bind_buffer_memory(vbuf, vmem, 0)
        .map_err(|e| fail_all(anyhow!("bind vmem: {e:?}")))?;
    let vmap = device
        .map_memory(vmem, 0, buf_size, vk::MemoryMapFlags::empty())
        .map_err(|e| fail_all(anyhow!("map vmem: {e:?}")))? as *mut Vertex;
    Ok((vbuf, vmem, vmap, buf_size))
}

impl Renderer {
    /// Everything that does not depend on the swapchain. `region_count` must
    /// cover the ACTUAL swapchain image count (see Swapchain::images).
    ///
    /// # Safety
    ///
    /// Creates Vulkan buffers/memory and maps them; `ctx` must be a valid
    /// initialized GfxContext.
    pub unsafe fn new(ctx: &GfxContext, region_count: usize) -> Result<Self> {
        let device = ctx.device();
        let queue_family = ctx.queue_family;

        // --- pipeline (dynamic rendering: no render pass) ---
        let spv: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shader.spv"));
        let code = u32_slice(spv);
        let sm_info = vk::ShaderModuleCreateInfo {
            code_size: (code.len() * 4) as usize,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        let vert_mod = device
            .create_shader_module(&sm_info, None)
            .map_err(|e| anyhow!("vert shader: {e:?}"))?;
        let frag_mod = device
            .create_shader_module(&sm_info, None)
            .map_err(|e| anyhow!("frag shader: {e:?}"))?;

        let vert_name = c"vs_main".as_ptr();
        let frag_name = c"fs_main".as_ptr();
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_mod,
                p_name: vert_name,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_mod,
                p_name: frag_name,
                ..Default::default()
            },
        ];

        let vert_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attrs = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 8,
            },
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &vert_binding,
            vertex_attribute_description_count: 2,
            p_vertex_attribute_descriptions: attrs.as_ptr(),
            ..Default::default()
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };
        let raster = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        };
        let multisample = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let blend_att = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };
        let blend_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &blend_att,
            ..Default::default()
        };
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic.len() as u32,
            p_dynamic_states: dynamic.as_ptr(),
            ..Default::default()
        };
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .map_err(|e| anyhow!("pipeline layout: {e:?}"))?;

        // dynamic rendering: color attachment formats via p_next
        let mut rendering_create = vk::PipelineRenderingCreateInfo {
            color_attachment_count: 1,
            p_color_attachment_formats: &vk::Format::B8G8R8A8_UNORM,
            ..Default::default()
        };
        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &raster,
            p_multisample_state: &multisample,
            p_color_blend_state: &blend_state,
            p_dynamic_state: &dynamic_state,
            layout,
            p_next: &mut rendering_create as *mut _ as *const c_void,
            ..Default::default()
        };
        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| anyhow!("create_graphics_pipelines: {e:?}"))?;
        let pipeline = pipelines[0];

        // --- command pool ---
        let cp_info = vk::CommandPoolCreateInfo {
            queue_family_index: queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let cmd_pool = device
            .create_command_pool(&cp_info, None)
            .map_err(|e| anyhow!("cmd pool: {e:?}"))?;

        // --- host-visible vertex buffer: one region per swapchain image ---
        let mem_props = ctx
            .instance()
            .get_physical_device_memory_properties(ctx.pdevice);
        let (vbuf, vmem, vmap, vbuf_bytes) =
            unsafe { alloc_vertex_regions(device, &mem_props, region_count)? };

        Ok(Self {
            cmd_pool,
            pipeline,
            layout,
            vert_mod,
            frag_mod,
            vbuf,
            vmem,
            vmap,
            vert_regions: region_count,
            vbuf_bytes,
            verts: Vec::new(),
            views: Vec::new(),
            cmds: Vec::new(),
            img_fences: Vec::new(),
            cb_dirty: Vec::new(),
            viewport: vk::Viewport::default(),
            scissor: vk::Rect2D::default(),
            images: Vec::new(),
            shot_buf: vk::Buffer::null(),
            shot_mem: vk::DeviceMemory::null(),
            shot_map: ptr::null_mut(),
            shot_extent: vk::Extent2D::default(),
            shot_pending: false,
            shot_ready: false,
        })
    }

    /// Swapchain-dependent objects: views, command buffers, viewport/scissor.
    /// Called on creation and on every recreation.
    pub unsafe fn create_swapchain_objects(
        &mut self,
        ctx: &GfxContext,
        sc: &Swapchain,
    ) -> Result<()> {
        let device = ctx.device();
        // reset-able command buffers: one per image
        if !self.cmds.is_empty() {
            device.free_command_buffers(self.cmd_pool, &self.cmds);
            self.cmds.clear();
        }
        let cb_alloc = vk::CommandBufferAllocateInfo {
            command_pool: self.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: sc.image_count() as u32,
            ..Default::default()
        };
        self.cmds = device
            .allocate_command_buffers(&cb_alloc)
            .map_err(|e| anyhow!("allocate cmds: {e:?}"))?;
        self.cb_dirty = vec![true; sc.image_count()];
        // per-image fences, created SIGNALED (an image's first record has no
        // previous submit to wait for)
        for old in self.img_fences.drain(..) {
            device.destroy_fence(old, None);
        }
        for _ in 0..sc.image_count() {
            let f = device
                .create_fence(
                    &vk::FenceCreateInfo {
                        flags: vk::FenceCreateFlags::SIGNALED,
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|e| anyhow!("img fence: {e:?}"))?;
            self.img_fences.push(f);
        }

        for old in self.views.drain(..) {
            device.destroy_image_view(old, None);
        }

        for img in &sc.images {
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let iv_info = vk::ImageViewCreateInfo {
                image: *img,
                view_type: vk::ImageViewType::TYPE_2D,
                format: sc.format,
                subresource_range: range,
                ..Default::default()
            };
            let view = device
                .create_image_view(&iv_info, None)
                .map_err(|e| anyhow!("image view: {e:?}"))?;
            self.views.push(view);
        }

        self.viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: sc.extent.width as f32,
            height: sc.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: sc.extent,
        };

        // --- screenshot buffer (host visible, BGRA, w*h*4) ---
        // non-fatal: a transient allocation failure (e.g. after device-lost
        // churn) only disables screenshots, the renderer keeps running
        self.images = sc.images.clone();
        match self.setup_shot(ctx, sc.extent) {
            Ok(()) => {}
            Err(e) => {
                crate::elog!("screenshot buffer unavailable: {e:#}; screenshots disabled");
                self.shot_pending = false;
                self.shot_ready = false;
            }
        }
        Ok(())
    }

    /// Allocate the host-visible screenshot buffer. On error the partial
    /// resources are cleaned up.
    unsafe fn setup_shot(&mut self, ctx: &GfxContext, extent: vk::Extent2D) -> Result<()> {
        let device = ctx.device();
        let byte_size = (extent.width as u64) * (extent.height as u64) * 4;
        let shot_buf = device
            .create_buffer(
                &vk::BufferCreateInfo {
                    size: byte_size,
                    usage: vk::BufferUsageFlags::TRANSFER_DST,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("shot buf: {e:?}"))?;
        let req = device.get_buffer_memory_requirements(shot_buf);
        let mem_props = ctx
            .instance()
            .get_physical_device_memory_properties(ctx.pdevice);
        let host_idx = (0..mem_props.memory_type_count)
            .find(|i| {
                let t = mem_props.memory_types[*i as usize];
                req.memory_type_bits & (1 << i) != 0
                    && t.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .ok_or_else(|| anyhow!("host visible memory type"))? as u32;
        let shot_mem = match device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: req.size,
                memory_type_index: host_idx,
                ..Default::default()
            },
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                // partial: destroy the buffer, keep the renderer alive
                device.destroy_buffer(shot_buf, None);
                return Err(anyhow!("shot mem: {e:?}"));
            }
        };
        if let Err(e) = device.bind_buffer_memory(shot_buf, shot_mem, 0) {
            device.destroy_buffer(shot_buf, None);
            device.free_memory(shot_mem, None);
            return Err(anyhow!("bind shot: {e:?}"));
        }
        let shot_map = match device.map_memory(shot_mem, 0, byte_size, vk::MemoryMapFlags::empty())
        {
            Ok(m) => m,
            Err(e) => {
                device.destroy_buffer(shot_buf, None);
                device.free_memory(shot_mem, None);
                return Err(anyhow!("map shot: {e:?}"));
            }
        };
        self.shot_buf = shot_buf;
        self.shot_mem = shot_mem;
        self.shot_map = shot_map;
        self.shot_extent = extent;
        Ok(())
    }

    /// Request a screenshot of the next recorded frame.
    pub fn request_screenshot(&mut self) {
        if !self.shot_pending {
            self.shot_pending = true;
        }
    }

    /// True after the pending screenshot has been copied out.
    pub fn shot_ready(&self) -> bool {
        self.shot_ready
    }

    /// True while a screenshot copy is still pending (not yet recorded).
    /// The caller must force CB re-record + upload this frame.
    pub fn shot_pending(&self) -> bool {
        self.shot_pending
    }

    /// Write the captured frame as a PNG next to the exe.
    /// Clears the ready flag so one request saves one file.
    ///
    /// Minimal PNG encoder: RGBA8, filter 0 per row, zlib "stored" deflate
    /// blocks (no compression) — no external crates, ~8 MB per 1080p shot.
    pub unsafe fn write_png(&mut self, path: &str) -> Result<()> {
        use std::io::Write;
        let w = self.shot_extent.width;
        let h = self.shot_extent.height;
        let data = std::slice::from_raw_parts(self.shot_map as *const u8, (w * h * 4) as usize);

        // BGRA (swapchain) -> RGBA, prefixed with a 0 filter byte per row
        let stride = (w * 4) as usize;
        let mut raw = Vec::with_capacity((stride + 1) * h as usize);
        for y in 0..h as usize {
            raw.push(0u8); // filter: none
            let row = &data[y * stride..(y + 1) * stride];
            for px in row.chunks_exact(4) {
                raw.push(px[2]); // R
                raw.push(px[1]); // G
                raw.push(px[0]); // B
                raw.push(px[3]); // A
            }
        }

        // zlib wrapper + stored deflate blocks
        let mut zlib = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);
        zlib.push(0x78);
        zlib.push(0x01);
        let mut off = 0usize;
        while off < raw.len() {
            let n = (raw.len() - off).min(65535);
            let last = off + n >= raw.len();
            zlib.push(if last { 1 } else { 0 });
            zlib.extend_from_slice(&(n as u16).to_le_bytes());
            zlib.extend_from_slice(&(!(n as u16)).to_le_bytes());
            zlib.extend_from_slice(&raw[off..off + n]);
            off += n;
        }
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut f = std::fs::File::create(path)?;
        f.write_all(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])?;
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // depth 8, RGBA, deflate, filter, interlace
        write_chunk(&mut f, b"IHDR", &ihdr)?;
        write_chunk(&mut f, b"IDAT", &zlib)?;
        write_chunk(&mut f, b"IEND", &[])?;
        self.shot_ready = false;
        Ok(())
    }

    /// Destroy swapchain-dependent objects (before recreation/exit).
    pub unsafe fn destroy_swapchain_objects(&mut self, device: &ash::Device) {
        if !self.cmds.is_empty() {
            device.free_command_buffers(self.cmd_pool, &self.cmds);
            self.cmds.clear();
        }
        self.cb_dirty.clear();
        for f in self.img_fences.drain(..) {
            device.destroy_fence(f, None);
        }
        for v in self.views.drain(..) {
            device.destroy_image_view(v, None);
        }
        self.images.clear();
        if self.shot_buf != vk::Buffer::null() {
            device.unmap_memory(self.shot_mem);
            device.destroy_buffer(self.shot_buf, None);
            device.free_memory(self.shot_mem, None);
            self.shot_buf = vk::Buffer::null();
            self.shot_pending = false;
            self.shot_ready = false;
        }
    }

    /// Retain the new UI vertex content as the latest CPU snapshot. Mapped
    /// image regions are NOT written here; each image's region is written in
    /// write_image_verts() after that image's fence has signaled. The caller
    /// must mark every swapchain image dirty — a cached CB bakes the old
    /// vertex count.
    pub unsafe fn upload_verts(&mut self, verts: &[Vertex]) {
        assert!(verts.len() <= MAX_VERTS);
        self.verts.clear();
        self.verts.extend_from_slice(verts);
    }

    /// Write the latest UI vertices into `img_idx`'s mapped region.
    ///
    /// # Safety
    ///
    /// The image fence for `img_idx` must have been waited on first
    /// (ensure_image_idle): the image owns its command buffer, so the fence
    /// covers every submission that can read this region.
    pub unsafe fn write_image_verts(&mut self, img_idx: usize) -> Result<()> {
        check_region_index(
            img_idx,
            self.vert_regions,
            VERTEX_REGION_BYTES,
            self.vbuf_bytes,
        )
        .map_err(|e| anyhow!("write_image_verts(img {img_idx}): {e:#}"))?;
        let base = self.vmap.add(img_idx * MAX_VERTS);
        for (n, v) in self.verts.iter().enumerate() {
            ptr::copy_nonoverlapping(v, base.add(n), 1);
        }
        Ok(())
    }

    pub fn mark_dirty(&mut self, img_idx: usize) {
        if img_idx < self.cb_dirty.len() {
            self.cb_dirty[img_idx] = true;
        }
    }

    pub fn is_dirty(&self, img_idx: usize) -> bool {
        self.cb_dirty[img_idx]
    }

    /// Enqueue an empty submit signaling the image's fence: the fence fires
    /// after ALL earlier queue work (this frame's real submit included) has
    /// completed, so the next reset/record of this image's CB is safe.
    pub unsafe fn signal_image_fence(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        img_idx: usize,
    ) {
        let f = self.img_fences[img_idx];
        device.reset_fences(&[f]).expect("reset img fence");
        let empty = vk::SubmitInfo::default();
        device
            .queue_submit(queue, &[empty], f)
            .expect("submit img fence");
    }

    pub fn command_buffer(&self, img_idx: usize) -> &vk::CommandBuffer {
        &self.cmds[img_idx]
    }

    /// Wait until this image's previous CB (and its fence submit) has fully
    /// completed. MUST be called every frame before the record-or-reuse path:
    /// skipped-record frames still resubmit the cached CB, and IMMEDIATE
    /// present does not throttle acquire, so the GPU may lag several frames.
    pub unsafe fn ensure_image_idle(&self, device: &ash::Device, img_idx: usize) -> Result<()> {
        device
            .wait_for_fences(&[self.img_fences[img_idx]], true, 2_000_000_000)
            .map_err(|e| anyhow!("img cb fence wait: {e:?}"))
    }

    /// Number of per-image vertex regions the mapped allocation covers.
    pub fn vertex_region_capacity(&self) -> usize {
        self.vert_regions
    }

    /// Grow the vertex region allocation to cover `image_count` regions.
    /// The CPU snapshot (`verts`) is preserved; regions are rewritten
    /// per-image after their fences signal.
    ///
    /// # Safety
    ///
    /// No GPU work may reference the old buffer: call only with the device
    /// idle (swapchain recreation path) or on a freshly rebuilt device.
    pub unsafe fn ensure_region_capacity(
        &mut self,
        ctx: &GfxContext,
        image_count: usize,
    ) -> Result<()> {
        if image_count <= self.vert_regions {
            return Ok(());
        }
        crate::elog!(
            "vertex regions: {} -> {} (actual swapchain images)",
            self.vert_regions,
            image_count
        );
        let device = ctx.device();
        let mem_props = ctx
            .instance()
            .get_physical_device_memory_properties(ctx.pdevice);
        // allocate the replacement FIRST: on any failure the old allocation
        // stays intact and usable (transactional growth)
        let (vbuf, vmem, vmap, vbuf_bytes) =
            unsafe { alloc_vertex_regions(device, &mem_props, image_count)? };
        // replacement fully initialized — only now retire the old allocation
        // (unmap, then destroy the buffer to release the memory binding, then free)
        unsafe { device.unmap_memory(self.vmem) };
        unsafe { device.destroy_buffer(self.vbuf, None) };
        unsafe { device.free_memory(self.vmem, None) };
        self.vbuf = vbuf;
        self.vmem = vmem;
        self.vmap = vmap;
        self.vert_regions = image_count;
        self.vbuf_bytes = vbuf_bytes;
        Ok(())
    }

    /// Record the image's command buffer (if dirty). `slot` — this frame's
    /// ring slot. Without sprites (text-only frames).
    pub unsafe fn record(&mut self, device: &ash::Device, img_idx: usize) -> Result<()> {
        self.record_frame(device, img_idx, None)
    }

    /// Same, with an optional sprite batch drawn first (under the text).
    pub unsafe fn record_frame(
        &mut self,
        device: &ash::Device,
        img_idx: usize,
        sprites: Option<&SpriteBatcher>,
    ) -> Result<()> {
        if img_idx >= self.cmds.len() {
            return Err(anyhow!("record: img_idx out of range"));
        }
        check_region_index(
            img_idx,
            self.vert_regions,
            VERTEX_REGION_BYTES,
            self.vbuf_bytes,
        )?;
        self.cb_dirty[img_idx] = false;
        // the image's fence was already waited on by ensure_image_idle this
        // frame (including skipped-record frames) — reset/record is safe here
        let cmd = self.cmds[img_idx];
        device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .expect("reset cmd");
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .expect("begin cmd");

        // image layout transitions: the swapchain image arrives in an
        // arbitrary (usually UNDEFINED) layout — transition to
        // COLOR_ATTACHMENT_OPTIMAL before rendering and to PRESENT_SRC_KHR
        // after rendering (VK_LAYER_KHRONOS_validation requirement)
        let img = self.images[img_idx];
        let subres = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_color = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: img,
            subresource_range: subres,
            ..Default::default()
        };
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_color],
        );

        // dynamic rendering: begin/end rendering instead of a render pass
        let clear_vals = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let color_att = vk::RenderingAttachmentInfo {
            image_view: self.views[img_idx],
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: clear_vals[0],
            ..Default::default()
        };
        let rendering = vk::RenderingInfo {
            render_area: self.scissor,
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_att,
            ..Default::default()
        };
        device.cmd_begin_rendering(cmd, &rendering);
        // dynamic state must be set before ANY draw (sprite batch included)
        device.cmd_set_viewport(cmd, 0, &[self.viewport]);
        device.cmd_set_scissor(cmd, 0, &[self.scissor]);
        // sprites first (under the text)
        if let Some(sb) = sprites {
            sb.draw(device, cmd, img_idx)?;
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_set_viewport(cmd, 0, &[self.viewport]);
        device.cmd_set_scissor(cmd, 0, &[self.scissor]);
        // the image owns its vertex region: the bind offset selects it
        let vert_off = (img_idx * MAX_VERTS * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
        device.cmd_bind_vertex_buffers(cmd, 0, &[self.vbuf], &[vert_off]);
        if !self.verts.is_empty() {
            device.cmd_draw(cmd, self.verts.len() as u32, 1, 0, 0);
        }
        device.cmd_end_rendering(cmd);
        // transition to the present layout (images must be in
        // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR at vkQueuePresentKHR) — skipped
        // when the screenshot path below does its own final transition
        let shot_now = self.shot_pending
            && self.shot_buf != vk::Buffer::null()
            && (img_idx as u32) < self.images.len() as u32;
        if !shot_now {
            let to_present = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::empty(),
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: img,
                subresource_range: subres,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        }
        // screenshot: copy the swapchain image out before present
        if self.shot_pending
            && self.shot_buf != vk::Buffer::null()
            && (img_idx as u32) < self.images.len() as u32
        {
            self.shot_pending = false;
            self.shot_ready = true;
            let img = self.images[img_idx];
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_src = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: img,
                subresource_range: range,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src],
            );
            let copy = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D {
                    width: self.shot_extent.width,
                    height: self.shot_extent.height,
                    depth: 1,
                },
                ..Default::default()
            };
            device.cmd_copy_image_to_buffer(
                cmd,
                img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.shot_buf,
                &[copy],
            );
            let to_present = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::empty(),
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: img,
                subresource_range: range,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        }
        device.end_command_buffer(cmd).expect("end cmd");
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        self.destroy_swapchain_objects(device);
        // unmap, then destroy the buffer (releases the memory binding), then free
        device.unmap_memory(self.vmem);
        device.destroy_buffer(self.vbuf, None);
        device.free_memory(self.vmem, None);
        device.destroy_command_pool(self.cmd_pool, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_shader_module(self.vert_mod, None);
        device.destroy_shader_module(self.frag_mod, None);
    }
}

fn u32_slice(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// PNG chunk: length + type + data + CRC32.
fn write_chunk(f: &mut std::fs::File, ty: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(&(data.len() as u32).to_be_bytes())?;
    f.write_all(ty)?;
    f.write_all(data)?;
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(ty);
    crc_input.extend_from_slice(data);
    f.write_all(&crc32(0, &crc_input).to_be_bytes())?;
    Ok(())
}

/// CRC-32 (IEEE, bit-by-bit; prev is the previous crc32() output or 0).
fn crc32(prev: u32, data: &[u8]) -> u32 {
    let mut c = prev ^ 0xFFFFFFFF;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    c ^ 0xFFFFFFFF
}

/// Adler-32 (zlib trailer).
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::check_region_index;

    const RB: u64 = 100;

    #[test]
    fn region_index_bounds() {
        let buf = 8 * RB;
        assert!(check_region_index(0, 8, RB, buf).is_ok());
        assert!(check_region_index(7, 8, RB, buf).is_ok());
        assert!(check_region_index(8, 8, RB, buf).is_err());
        assert!(check_region_index(usize::MAX, 8, RB, buf).is_err());
        // allocation smaller than the region count: the last region extends
        // past the buffer and must be rejected
        assert!(check_region_index(7, 8, RB, 7 * RB).is_err());
        assert!(check_region_index(6, 8, RB, 7 * RB).is_ok());
    }

    #[test]
    fn capacity_follows_actual_image_count() {
        // a driver may return MORE images than the requested minImageCount:
        // the region capacity must be derived from the actual count, so an
        // actual count above FIF is covered (pure capacity-rule test, not a
        // Vulkan driver simulation)
        let actual_images = 12usize;
        let fif = 8usize;
        assert!(actual_images > fif);
        let regions = region_capacity_for(actual_images);
        assert!(regions >= actual_images);
        assert!(check_region_index(actual_images - 1, regions, RB, regions as u64 * RB).is_ok());
        assert!(check_region_index(actual_images, regions, RB, regions as u64 * RB).is_err());
    }

    /// Region capacity for a swapchain: exactly the actual image count.
    fn region_capacity_for(actual_image_count: usize) -> usize {
        actual_image_count
    }
}
