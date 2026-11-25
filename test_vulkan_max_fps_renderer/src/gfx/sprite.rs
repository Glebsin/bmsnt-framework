//! Sprite batcher: one instanced draw for the whole UI.
//!
//! - Instanced quads (no vertex buffer: corners from vertex_index) sampled
//!   from a small baked atlas (white patch + anti-aliased rounded corners).
//! - "Gradient" mode computes a vertical color mix per pixel with dithering
//!   in the fragment shader (no CPU gradient strips).
//! - Instance data lives in a host-visible ring buffer: one slot per frame
//!   in flight, same scheme as the text vertex ring in renderer.rs.

use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::ptr;

use ash::vk;

use super::device::GfxContext;
use super::frames::FRAMES_IN_FLIGHT;

/// per-frame instance limit
pub const MAX_INSTANCES: usize = 4096;
/// 20 f32 per instance (5 vec4 attributes)
const INSTANCE_F32: usize = 20;
const INSTANCE_BYTES: usize = INSTANCE_F32 * 4;
const ATLAS_SIZE: u32 = 512;

/// atlas rects of pre-baked rounded-corner sprites: (x0, y0, x1, y1, radius);
/// boxes are 64x64, stretchable (corners distort on big aspect deltas — fine
/// for now, 9-slice comes with the widget stage)
const ROUNDED_SPRITES: [(f32, f32, f32, f32, u32); 3] = [
    (16.0, 0.0, 80.0, 64.0, 6),
    (96.0, 0.0, 160.0, 64.0, 12),
    (176.0, 0.0, 240.0, 64.0, 24),
];
/// white patch uv: inset by half a texel so LINEAR filtering never mixes
/// with the transparent border of the atlas
const WHITE_UV: [f32; 4] = [
    0.5 / ATLAS_SIZE as f32,
    0.5 / ATLAS_SIZE as f32,
    3.5 / ATLAS_SIZE as f32,
    3.5 / ATLAS_SIZE as f32,
];

/// one contiguous run of instances sharing a scissor rect (framebuffer px);
/// sections let the test browser clip the "game window" separately from UI
struct Section {
    first_instance: u32,
    count: u32,
    sc: [f32; 4], // x, y, w, h in framebuffer pixels
}

/// CPU-side instance list; uploaded per ring slot.
pub struct SpriteBatcher {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    desc_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    atlas_img: vk::Image,
    atlas_mem: vk::DeviceMemory,
    atlas_view: vk::ImageView,
    sampler: vk::Sampler,
    inst_buf: vk::Buffer,
    inst_mem: vk::DeviceMemory,
    inst_map: *mut u8,
    instances: Vec<f32>,
    /// current scissor for pushed instances
    cur_section: Option<[f32; 4]>,
    sections: Vec<Section>,
}

impl SpriteBatcher {
    pub unsafe fn new(ctx: &GfxContext) -> Result<Self> {
        let device = ctx.device();

        // --- bake the atlas (CPU, in memory; no image crate needed) ---
        let pixels = bake_atlas();

        // --- atlas image (device local) ---
        let img_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        let atlas_img = device.create_image(&img_info, None).expect("atlas image");
        let img_req = device.get_image_memory_requirements(atlas_img);
        let mem_props = ctx.instance().get_physical_device_memory_properties(ctx.pdevice);
        let dev_local = (0..mem_props.memory_type_count)
            .find(|i| {
                let t = mem_props.memory_types[*i as usize];
                img_req.memory_type_bits & (1 << i) != 0
                    && t.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| anyhow!("device local memory type"))? as u32;
        let atlas_mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: img_req.size,
                    memory_type_index: dev_local,
                    ..Default::default()
                },
                None,
            )
            .expect("atlas mem");
        device.bind_image_memory(atlas_img, atlas_mem, 0).expect("bind atlas");

        // --- staging buffer + upload with a one-time command buffer ---
        let byte_size = pixels.len() as u64;
        let (staging, staging_mem) = {
            let buf_info = vk::BufferCreateInfo {
                size: byte_size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let buf = device.create_buffer(&buf_info, None).expect("staging");
            let req = device.get_buffer_memory_requirements(buf);
            let host_idx = (0..mem_props.memory_type_count)
                .find(|i| {
                    let t = mem_props.memory_types[*i as usize];
                    req.memory_type_bits & (1 << i) != 0
                        && t
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
                })
                .ok_or_else(|| anyhow!("host visible memory type"))? as u32;
            let mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: req.size,
                        memory_type_index: host_idx,
                        ..Default::default()
                    },
                    None,
                )
                .expect("staging mem");
            device.bind_buffer_memory(buf, mem, 0).expect("bind staging");
            let map = device
                .map_memory(mem, 0, byte_size, vk::MemoryMapFlags::empty())
                .expect("map staging");
            ptr::copy_nonoverlapping(pixels.as_ptr(), map as *mut u8, pixels.len());
            (buf, mem)
        };

        // one-time command buffer: layout transitions + copy
        let cp_info = vk::CommandPoolCreateInfo {
            queue_family_index: ctx.queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let tmp_pool = device
            .create_command_pool(&cp_info, None)
            .map_err(|e| anyhow!("tmp cmd pool: {e:?}"))?;
        let cb = device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: tmp_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })
            .map_err(|e| anyhow!("tmp cmd: {e:?}"))?[0];
        device
            .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
            .expect("begin tmp cmd");

        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let pre = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: atlas_img,
            subresource_range: range,
            ..Default::default()
        };
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[pre],
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth: 1,
            },
            ..Default::default()
        };
        device.cmd_copy_buffer_to_image(
            cb,
            staging,
            atlas_img,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        let post = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: atlas_img,
            subresource_range: range,
            ..Default::default()
        };
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[post],
        );
        device.end_command_buffer(cb).expect("end tmp cmd");
        let upload_submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cb,
            ..Default::default()
        };
        device
            .queue_submit(ctx.queue, &[upload_submit], vk::Fence::null())
            .expect("submit upload");
        device.device_wait_idle().expect("wait upload");

        device.unmap_memory(staging_mem);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
        device.destroy_command_pool(tmp_pool, None);

        // --- image view + sampler + descriptor ---
        let atlas_view = device
            .create_image_view(
                &vk::ImageViewCreateInfo {
                    image: atlas_img,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::R8G8B8A8_UNORM,
                    subresource_range: range,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("atlas view: {e:?}"))?;
        let sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::LINEAR,
                    min_filter: vk::Filter::LINEAR,
                    mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    min_lod: 0.0,
                    max_lod: 0.0,
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("sampler: {e:?}"))?;
        // --- descriptor layout, pool, set ---
        let desc_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: 1,
                    p_bindings: [vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    }]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("desc layout: {e:?}"))?;
        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 1,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                        }
                    ]
                    .as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("desc pool: {e:?}"))?;
        let desc_set = device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: desc_pool,
                descriptor_set_count: 1,
                p_set_layouts: [desc_layout].as_ptr(),
                ..Default::default()
            })
            .map_err(|e| anyhow!("desc set: {e:?}"))?[0];
        let img_infos = [vk::DescriptorImageInfo {
            sampler,
            image_view: atlas_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: img_infos.as_ptr(),
                ..Default::default()
            }],
            &[],
        );

        // --- pipeline layout ---
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: [desc_layout].as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("sprite layout: {e:?}"))?;

        // --- pipeline (same dynamic-rendering setup as the text pipeline) ---
        let spv: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite.spv"));
        let code = u32_slice(spv);
        let sm_info = vk::ShaderModuleCreateInfo {
            code_size: (code.len() * 4) as usize,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        let sm = device
            .create_shader_module(&sm_info, None)
            .map_err(|e| anyhow!("sprite shader: {e:?}"))?;
        let entry = c"vs_main".as_ptr();
        let stages = [vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: sm,
            p_name: entry,
            ..Default::default()
        }];
        let frag_entry = c"fs_main".as_ptr();
        let stages = [
            stages[0],
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: sm,
                p_name: frag_entry,
                ..Default::default()
            },
        ];

        // instances only: corners come from vertex_index in the shader
        let inst_binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: INSTANCE_BYTES as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        };
        let attrs = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 16,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 48,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 64,
            },
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &inst_binding,
            vertex_attribute_description_count: attrs.len() as u32,
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
        // straight-alpha blending for the AA edges of rounded sprites
        let blend_att = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
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
        let mut rendering_create = vk::PipelineRenderingCreateInfo {
            color_attachment_count: 1,
            p_color_attachment_formats: &vk::Format::B8G8R8A8_UNORM,
            ..Default::default()
        };
        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
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
            .map_err(|e| anyhow!("sprite pipeline: {e:?}"))?;
        let pipeline = pipelines[0];
        device.destroy_shader_module(sm, None);

        // --- instance ring buffer (host visible, one slot per frame) ---
        let buf_size = (MAX_INSTANCES * INSTANCE_BYTES * FRAMES_IN_FLIGHT) as u64;
        let inst_buf = device
            .create_buffer(
                &vk::BufferCreateInfo {
                    size: buf_size,
                    usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )
            .expect("inst buf");
        let req = device.get_buffer_memory_requirements(inst_buf);
        let host_idx = (0..mem_props.memory_type_count)
            .find(|i| {
                let t = mem_props.memory_types[*i as usize];
                req.memory_type_bits & (1 << i) != 0
                    && t
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .ok_or_else(|| anyhow!("host visible memory type"))? as u32;
        let inst_mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: req.size,
                    memory_type_index: host_idx,
                    ..Default::default()
                },
                None,
            )
            .expect("inst mem");
        device.bind_buffer_memory(inst_buf, inst_mem, 0).expect("bind inst");
        let inst_map = device
            .map_memory(inst_mem, 0, buf_size, vk::MemoryMapFlags::empty())
            .expect("map inst") as *mut u8;

        Ok(Self {
            pipeline,
            layout,
            desc_layout,
            desc_pool,
            desc_set,
            atlas_img,
            atlas_mem,
            atlas_view,
            sampler,
            inst_buf,
            inst_mem,
            inst_map,
            instances: Vec::new(),
            cur_section: None,
            sections: Vec::new(),
        })
    }

    // --- CPU-side batching ---

    /// Clear the instance list. Call once per frame.
    pub fn begin(&mut self) {
        self.instances.clear();
        self.sections.clear();
        self.cur_section = None;
    }

    /// Set the scissor rect (framebuffer pixels) for subsequent sprites.
    /// Pushing with a new rect starts a new section (one extra draw call).
    pub fn begin_section(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.cur_section = Some([x, y, w, h]);
    }

    fn push(&mut self, data: [f32; INSTANCE_F32]) {
        assert!(
            self.instances.len() / INSTANCE_F32 < MAX_INSTANCES,
            "sprite batch overflow ({} instances max)",
            MAX_INSTANCES
        );
        let idx = self.instances.len() / INSTANCE_F32;
        let cur = self.cur_section.unwrap_or([0.0, 0.0, f32::MAX, f32::MAX]);
        match self.sections.last_mut() {
            Some(s) if s.sc == cur => s.count += 1,
            _ => self.sections.push(Section {
                first_instance: idx as u32,
                count: 1,
                sc: cur,
            }),
        }
        self.instances.extend_from_slice(&data);
    }

    /// Textured sprite from an explicit atlas rect.
    pub fn sprite(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uv: [f32; 4],
        color: (f32, f32, f32, f32),
        res: (f32, f32),
    ) {
        self.push([
            x, y, w, h, uv[0], uv[1], uv[2], uv[3], color.0, color.1, color.2, color.3, 0.0, 0.0,
            0.0, 0.0, 0.0, res.0, res.1, 0.0,
        ]);
    }

    /// Solid tinted rectangle (white atlas patch, sharp edges).
    pub fn solid(&mut self, x: f32, y: f32, w: f32, h: f32, color: (f32, f32, f32), res: (f32, f32)) {
        self.sprite(x, y, w, h, WHITE_UV, (color.0, color.1, color.2, 1.0), res);
    }

    /// Rounded rectangle; `radius_idx` selects a pre-baked corner radius.
    pub fn rounded(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius_idx: usize,
        color: (f32, f32, f32),
        res: (f32, f32),
    ) {
        let sp = ROUNDED_SPRITES[radius_idx % ROUNDED_SPRITES.len()];
        let uv = [
            sp.0 / ATLAS_SIZE as f32,
            sp.1 / ATLAS_SIZE as f32,
            sp.2 / ATLAS_SIZE as f32,
            sp.3 / ATLAS_SIZE as f32,
        ];
        self.sprite(x, y, w, h, uv, (color.0, color.1, color.2, 1.0), res);
    }

    /// Full-height vertical gradient with shader-side dithering.
    pub fn gradient(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        top: (f32, f32, f32),
        bottom: (f32, f32, f32),
        res: (f32, f32),
    ) {
        self.push([
            x, y, w, h, 0.0, 0.0, 0.0, 1.0, top.0, top.1, top.2, 1.0, bottom.0, bottom.1,
            bottom.2, 1.0, 1.0, res.0, res.1, 0.0,
        ]);
    }

    // --- GPU side ---

    /// Copy the current instance list into the ring slot of this frame.
    pub unsafe fn upload_slot(&mut self, _device: &ash::Device, slot: usize) {
        let n = self.instances.len() / INSTANCE_F32;
        let base = self.inst_map.add(slot * MAX_INSTANCES * INSTANCE_BYTES);
        if n > 0 {
            let src = self.instances.as_ptr() as *const u8;
            let len = n * INSTANCE_BYTES;
            ptr::copy_nonoverlapping(src, base, len);
        }
    }

    /// Bind the sprite pipeline and draw all sections of this frame's slot,
    /// setting each section's scissor. Must be called inside an active
    /// rendering scope (viewport must already be set).
    pub unsafe fn draw(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.layout,
            0,
            &[self.desc_set],
            &[],
        );
        for s in &self.sections {
            let off = vk::Offset2D {
                x: s.sc[0].max(0.0) as i32,
                y: s.sc[1].max(0.0) as i32,
            };
            let ext = vk::Extent2D {
                width: (s.sc[2].max(0.0).min(8192.0)) as u32,
                height: (s.sc[3].max(0.0).min(8192.0)) as u32,
            };
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: off, extent: ext }]);
            // address the section through the bind offset: the ring slot's
            // instances live at slot * MAX_INSTANCES within the shared buffer
            let byte_off =
                ((slot * MAX_INSTANCES + s.first_instance as usize) * INSTANCE_BYTES) as vk::DeviceSize;
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.inst_buf], &[byte_off]);
            device.cmd_draw(cmd, 6, s.count, 0, 0);
        }
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.unmap_memory(self.inst_mem);
        device.free_memory(self.inst_mem, None);
        device.destroy_buffer(self.inst_buf, None);
        device.destroy_descriptor_pool(self.desc_pool, None);
        device.destroy_descriptor_set_layout(self.desc_layout, None);
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.atlas_view, None);
        device.destroy_image(self.atlas_img, None);
        device.free_memory(self.atlas_mem, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
    }
}

/// Bake the atlas: white 4x4 patch + 64x64 rounded boxes with analytic AA.
fn bake_atlas() -> Vec<u8> {
    let mut px = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE * 4) as usize];
    let put = |px: &mut Vec<u8>, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8| {
        let i = ((y * ATLAS_SIZE + x) * 4) as usize;
        px[i] = r;
        px[i + 1] = g;
        px[i + 2] = b;
        px[i + 3] = a;
    };
    // white patch (used for tinted solids)
    for y in 0..4u32 {
        for x in 0..4u32 {
            put(&mut px, x, y, 255, 255, 255, 255);
        }
    }
    // rounded boxes: signed-distance rounded rect, alpha = coverage
    for (x0, y0, x1, y1, rad) in ROUNDED_SPRITES {
        let w = x1 - x0;
        let h = y1 - y0;
        let half = (w / 2.0 - rad as f32, h / 2.0 - rad as f32);
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let p = (
                    x as f32 - w / 2.0 + 0.5,
                    y as f32 - h / 2.0 + 0.5,
                );
                let q = (p.0.abs() - half.0, p.1.abs() - half.1);
                let d_out = q.0.max(0.0).hypot(q.1.max(0.0));
                let d_in = q.0.max(q.1).min(0.0);
                let d = d_out + d_in - rad as f32;
                let a = (0.5 - d).clamp(0.0, 1.0);
                put(
                    &mut px,
                    x0 as u32 + x,
                    y0 as u32 + y,
                    255,
                    255,
                    255,
                    (a * 255.0) as u8,
                );
            }
        }
    }
    px
}

fn u32_slice(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
