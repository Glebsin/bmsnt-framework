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
use super::swapchain::Swapchain;
use crate::ui::text::Vertex;

/// per-slot vertex limit — shares the source of truth with ui::text
pub const MAX_VERTS: usize = crate::ui::text::MAX_VERTS;
/// ring slots = frames in flight
const RING_SLOTS: usize = super::frames::FRAMES_IN_FLIGHT;

pub struct Renderer {
    cmd_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    vert_mod: vk::ShaderModule,
    frag_mod: vk::ShaderModule,
    vbuf: vk::Buffer,
    vmem: vk::DeviceMemory,
    vmap: *mut Vertex,
    /// current text vertices (to re-upload ring slots when dirty)
    verts: Vec<Vertex>,
    // swapchain-dependent
    views: Vec<vk::ImageView>,
    cmds: Vec<vk::CommandBuffer>,
    cb_dirty: Vec<bool>,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
}

impl Renderer {
    /// Everything that does not depend on the swapchain.
    pub unsafe fn new(ctx: &GfxContext) -> Result<Self> {
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

        // --- ring host-visible vertex buffer: one slot per frame in flight ---
        let slice = (MAX_VERTS * std::mem::size_of::<Vertex>()) as u64;
        let buf_size = slice * RING_SLOTS as u64;
        let vb_info = vk::BufferCreateInfo {
            size: buf_size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let vbuf = device.create_buffer(&vb_info, None).expect("vbuf");
        let mem_req = device.get_buffer_memory_requirements(vbuf);
        let mem_props = ctx.instance().get_physical_device_memory_properties(ctx.pdevice);
        let mem_type = (0..mem_props.memory_type_count)
            .find(|i| {
                let t = mem_props.memory_types[*i as usize];
                mem_req.memory_type_bits & (1 << i) != 0
                    && t
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .ok_or_else(|| anyhow!("host visible memory type"))? as u32;
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: mem_type,
            ..Default::default()
        };
        let vmem = device.allocate_memory(&alloc_info, None).expect("vmem");
        device.bind_buffer_memory(vbuf, vmem, 0).expect("bind vmem");
        let vmap = device
            .map_memory(vmem, 0, buf_size, vk::MemoryMapFlags::empty())
            .expect("map vmem") as *mut Vertex;

        Ok(Self {
            cmd_pool,
            pipeline,
            layout,
            vert_mod,
            frag_mod,
            vbuf,
            vmem,
            vmap,
            verts: Vec::new(),
            views: Vec::new(),
            cmds: Vec::new(),
            cb_dirty: Vec::new(),
            viewport: vk::Viewport::default(),
            scissor: vk::Rect2D::default(),
        })
    }

    /// Swapchain-dependent objects: views, command buffers, viewport/scissor.
    /// Called on creation and on every recreation.
    pub unsafe fn create_swapchain_objects(&mut self, device: &ash::Device, sc: &Swapchain) -> Result<()> {
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
        Ok(())
    }

    /// Destroy swapchain-dependent objects (before recreation/exit).
    pub unsafe fn destroy_swapchain_objects(&mut self, device: &ash::Device) {
        if !self.cmds.is_empty() {
            device.free_command_buffers(self.cmd_pool, &self.cmds);
            self.cmds.clear();
        }
        self.cb_dirty.clear();
        for v in self.views.drain(..) {
            device.destroy_image_view(v, None);
        }
    }

    /// Copy vertices into EVERY ring slot. An image's CB is recorded at its
    /// acquire and binds the current frame's slot — so every slot must hold
    /// the current vertices (8 copies of ~5 KB is nothing).
    pub unsafe fn upload_verts(&mut self, verts: &[Vertex]) {
        assert!(verts.len() <= MAX_VERTS);
        for slot in 0..RING_SLOTS {
            let base = self.vmap.add(slot * MAX_VERTS);
            for (n, v) in verts.iter().enumerate() {
                ptr::copy_nonoverlapping(v, base.add(n), 1);
            }
        }
        self.verts.clear();
        self.verts.extend_from_slice(verts);
    }

    pub fn mark_dirty(&mut self, img_idx: usize) {
        if img_idx < self.cb_dirty.len() {
            self.cb_dirty[img_idx] = true;
        }
    }

    pub fn is_dirty(&self, img_idx: usize) -> bool {
        self.cb_dirty[img_idx]
    }

    pub fn command_buffer(&self, img_idx: usize) -> &vk::CommandBuffer {
        &self.cmds[img_idx]
    }

    /// Record the image's command buffer (if dirty). `slot` — this frame's
    /// ring slot.
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        img_idx: usize,
        slot: usize,
    ) -> Result<()> {
        if img_idx >= self.cmds.len() {
            return Err(anyhow!("record: img_idx out of range"));
        }
        self.cb_dirty[img_idx] = false;
        let cmd = self.cmds[img_idx];
        device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .expect("reset cmd");
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .expect("begin cmd");

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
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_set_viewport(cmd, 0, &[self.viewport]);
        device.cmd_set_scissor(cmd, 0, &[self.scissor]);
        // ring slot is selected by the bind offset
        let slot_off = (slot * MAX_VERTS * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
        device.cmd_bind_vertex_buffers(cmd, 0, &[self.vbuf], &[slot_off]);
        if !self.verts.is_empty() {
            device.cmd_draw(cmd, self.verts.len() as u32, 1, 0, 0);
        }
        device.cmd_end_rendering(cmd);
        device.end_command_buffer(cmd).expect("end cmd");
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        self.destroy_swapchain_objects(device);
        device.unmap_memory(self.vmem);
        device.free_memory(self.vmem, None);
        device.destroy_buffer(self.vbuf, None);
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
