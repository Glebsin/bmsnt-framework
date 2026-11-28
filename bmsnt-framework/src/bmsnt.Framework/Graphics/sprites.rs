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

use super::vulkan::device::GfxContext;
use super::vulkan::renderer::check_region_index;

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
    /// number of per-image regions the instance allocation covers
    inst_regions: usize,
    /// allocation size in bytes (bounds-check base for region ranges)
    inst_buf_bytes: u64,
    instances: Vec<f32>,
    /// current scissor for pushed instances
    cur_section: Option<[f32; 4]>,
    sections: Vec<Section>,
}

/// Bytes of one per-image instance region.
const INSTANCE_REGION_BYTES: u64 = (MAX_INSTANCES * INSTANCE_BYTES) as u64;

/// Allocate the host-visible per-image instance buffer: `regions` regions of
/// MAX_INSTANCES instances each, mapped for CPU writes.
///
/// # Safety
///
/// `device` must be valid; the returned buffer/memory must be unmapped,
/// destroyed and freed (in that order) before the device is destroyed.
unsafe fn alloc_instance_regions(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    regions: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)> {
    let buf_size = INSTANCE_REGION_BYTES * regions as u64;
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
        .map_err(|e| anyhow!("inst buf: {e:?}"))?;
    let fail_buf = |e: anyhow::Error| {
        unsafe { device.destroy_buffer(inst_buf, None) };
        e
    };
    let req = device.get_buffer_memory_requirements(inst_buf);
    let host_idx = (0..mem_props.memory_type_count)
        .find(|i| {
            let t = mem_props.memory_types[*i as usize];
            req.memory_type_bits & (1 << i) != 0
                && t.property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .ok_or_else(|| fail_buf(anyhow!("host visible memory type")))?;
    let inst_mem = device
        .allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: req.size,
                memory_type_index: host_idx,
                ..Default::default()
            },
            None,
        )
        .map_err(|e| fail_buf(anyhow!("inst mem: {e:?}")))?;
    let fail_all = |e: anyhow::Error| {
        // buffer must be destroyed before its bound memory is freed
        unsafe { device.destroy_buffer(inst_buf, None) };
        unsafe { device.free_memory(inst_mem, None) };
        e
    };
    device
        .bind_buffer_memory(inst_buf, inst_mem, 0)
        .map_err(|e| fail_all(anyhow!("bind inst: {e:?}")))?;
    let inst_map = device
        .map_memory(inst_mem, 0, buf_size, vk::MemoryMapFlags::empty())
        .map_err(|e| fail_all(anyhow!("map inst: {e:?}")))? as *mut u8;
    Ok((inst_buf, inst_mem, inst_map, buf_size))
}

/// Validate that appending `add_f32` floats keeps the composed instance
/// list within MAX_INSTANCES. Shared by push_cached; pure, unit-testable.
pub(crate) fn check_cached_capacity(existing_f32: usize, add_f32: usize) -> Result<()> {
    if add_f32 % INSTANCE_F32 != 0 {
        return Err(anyhow!(
            "cached instance data not a multiple of the instance stride"
        ));
    }
    let total = existing_f32
        .checked_add(add_f32)
        .ok_or_else(|| anyhow!("instance list size overflow"))?;
    if total > MAX_INSTANCES * INSTANCE_F32 {
        return Err(anyhow!(
            "sprite batch overflow: {} + {} instances exceeds MAX_INSTANCES {}",
            existing_f32 / INSTANCE_F32,
            add_f32 / INSTANCE_F32,
            MAX_INSTANCES
        ));
    }
    Ok(())
}

/// Validate one sprite section against the instance limit and the number
/// of instances actually uploaded for the image being drawn. Checked
/// arithmetic: never trusts stale or malformed section metadata.
fn validate_section(first: usize, count: usize, uploaded: usize) -> Result<()> {
    let end = first
        .checked_add(count)
        .ok_or_else(|| anyhow!("sprite section [{first}+{count}] overflows usize"))?;
    if end > MAX_INSTANCES {
        return Err(anyhow!(
            "sprite section [{first}+{count}] exceeds MAX_INSTANCES {MAX_INSTANCES}"
        ));
    }
    if end > uploaded {
        return Err(anyhow!(
            "sprite section [{first}+{count}] exceeds the {uploaded} uploaded instances"
        ));
    }
    Ok(())
}

impl SpriteBatcher {
    /// Everything that does not depend on the current scene. `region_count`
    /// must cover the ACTUAL swapchain image count; it can be grown later
    /// via ensure_region_capacity().
    ///
    /// # Safety
    ///
    /// Creates Vulkan buffers/memory and maps them; `ctx` must be a valid
    /// initialized GfxContext.
    pub unsafe fn new(ctx: &GfxContext, region_count: usize) -> Result<Self> {
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
        let mem_props = ctx
            .instance()
            .get_physical_device_memory_properties(ctx.pdevice);
        let dev_local = (0..mem_props.memory_type_count)
            .find(|i| {
                let t = mem_props.memory_types[*i as usize];
                img_req.memory_type_bits & (1 << i) != 0
                    && t.property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
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
        device
            .bind_image_memory(atlas_img, atlas_mem, 0)
            .expect("bind atlas");

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
                        && t.property_flags.contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                })
                .ok_or_else(|| anyhow!("host visible memory type"))?
                as u32;
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
            device
                .bind_buffer_memory(buf, mem, 0)
                .expect("bind staging");
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
        // ONE set with two bindings, exactly as the WGSL declares them
        // (texture @0, sampler @1 in group 0): VK_LAYER_KHRONOS_validation
        // confirms this matches the compiled SPIR-V; a COMBINED_IMAGE_SAMPLER
        // here leaves the shader's sampler unbound and every textured sprite
        // invisible.
        let b_image = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let b_sampler = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };
        let bindings = [b_image, b_sampler];
        let desc_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .map_err(|e| anyhow!("desc layout: {e:?}"))?;
        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 1,
                    pool_size_count: 2,
                    p_pool_sizes: [
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::SAMPLED_IMAGE,
                            descriptor_count: 1,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::SAMPLER,
                            descriptor_count: 1,
                        },
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
        let tex_infos = [vk::DescriptorImageInfo {
            image_view: atlas_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        }];
        let sampler_infos = [vk::DescriptorImageInfo {
            sampler,
            ..Default::default()
        }];
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet {
                    dst_set: desc_set,
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    p_image_info: tex_infos.as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: desc_set,
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    p_image_info: sampler_infos.as_ptr(),
                    ..Default::default()
                },
            ],
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

        // --- instance ring buffer (host visible, one region per image) ---
        let (inst_buf, inst_mem, inst_map, inst_buf_bytes) =
            unsafe { alloc_instance_regions(device, &mem_props, region_count)? };

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
            inst_regions: region_count,
            inst_buf_bytes,
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

    /// Snapshot of all instances pushed so far (cache a static UI layer).
    pub fn instances(&self) -> &[f32] {
        &self.instances
    }

    /// Push pre-built instances that all share one scissor rect; restores
    /// them as a single section (used to re-apply a cached UI layer without
    /// rebuilding its geometry). Errors instead of exceeding MAX_INSTANCES.
    pub fn push_cached(&mut self, data: &[f32], sc: [f32; 4]) -> Result<()> {
        check_cached_capacity(self.instances.len(), data.len())?;
        if data.is_empty() {
            return Ok(());
        }
        let first = (self.instances.len() / INSTANCE_F32) as u32;
        self.instances.extend_from_slice(data);
        self.sections.push(Section {
            first_instance: first,
            count: (data.len() / INSTANCE_F32) as u32,
            sc,
        });
        Ok(())
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
    pub fn solid(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: (f32, f32, f32),
        res: (f32, f32),
    ) {
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
            x, y, w, h, 0.0, 0.0, 0.0, 1.0, top.0, top.1, top.2, 1.0, bottom.0, bottom.1, bottom.2,
            1.0, 1.0, res.0, res.1, 0.0,
        ]);
    }

    // --- GPU side ---

    /// Write the current composed instance list into `img_idx`'s mapped
    /// region.
    ///
    /// # Safety
    ///
    /// The image fence for `img_idx` must have been waited on first
    /// (ensure_image_idle): the image owns its command buffer — the only
    /// GPU consumer of this region — and all its submissions are complete.
    pub unsafe fn upload_region(&mut self, img_idx: usize) -> Result<()> {
        if self.instances.len() > MAX_INSTANCES * INSTANCE_F32 {
            return Err(anyhow!(
                "upload_region(img {img_idx}): instance list {} exceeds MAX_INSTANCES {}",
                self.instances.len() / INSTANCE_F32,
                MAX_INSTANCES
            ));
        }
        check_region_index(
            img_idx,
            self.inst_regions,
            INSTANCE_REGION_BYTES,
            self.inst_buf_bytes,
        )
        .map_err(|e| anyhow!("upload_region(img {img_idx}): {e:#}"))?;
        let n = self.instances.len() / INSTANCE_F32;
        let base = self.inst_map.add(img_idx * MAX_INSTANCES * INSTANCE_BYTES);
        if n > 0 {
            let src = self.instances.as_ptr() as *const u8;
            let len = n * INSTANCE_BYTES;
            ptr::copy_nonoverlapping(src, base, len);
        }
        Ok(())
    }

    /// Bind the sprite pipeline and draw all sections of this image's
    /// region, setting each section's scissor. Must be called inside an
    /// active rendering scope (viewport must already be set).
    ///
    /// # Safety
    ///
    /// Active rendering scope required; the image fence for `img_idx` must
    /// have been waited on (the region is the image's exclusively).
    pub unsafe fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        img_idx: usize,
    ) -> Result<()> {
        check_region_index(
            img_idx,
            self.inst_regions,
            INSTANCE_REGION_BYTES,
            self.inst_buf_bytes,
        )?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.layout,
            0,
            &[self.desc_set],
            &[],
        );
        // the composed list was uploaded to this image's region this frame;
        // every section must lie inside the limit and that uploaded content
        let uploaded = self.instances.len() / INSTANCE_F32;
        for s in &self.sections {
            validate_section(s.first_instance as usize, s.count as usize, uploaded)?;
            let off = vk::Offset2D {
                x: s.sc[0].max(0.0) as i32,
                y: s.sc[1].max(0.0) as i32,
            };
            let ext = vk::Extent2D {
                width: (s.sc[2].max(0.0).min(8192.0)) as u32,
                height: (s.sc[3].max(0.0).min(8192.0)) as u32,
            };
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: off,
                    extent: ext,
                }],
            );
            // address the section through the bind offset: this image's
            // instances live at img_idx * MAX_INSTANCES within the shared
            // buffer; checked range math, before any bind/draw
            let byte_off =
                ((img_idx * MAX_INSTANCES + s.first_instance as usize) * INSTANCE_BYTES) as u64;
            let byte_end = byte_off + (s.count as u64) * INSTANCE_BYTES as u64;
            let region_end = (img_idx as u64 + 1) * MAX_INSTANCES as u64 * INSTANCE_BYTES as u64;
            if byte_end > region_end {
                return Err(anyhow!(
                    "sprite section [{}, {}) exceeds image {} region bounds",
                    s.first_instance,
                    s.first_instance as usize + s.count as usize,
                    img_idx
                ));
            }
            let byte_off = byte_off as vk::DeviceSize;
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.inst_buf], &[byte_off]);
            device.cmd_draw(cmd, 6, s.count, 0, 0);
        }
        Ok(())
    }

    /// Number of per-image instance regions the mapped allocation covers.
    pub fn instance_region_capacity(&self) -> usize {
        self.inst_regions
    }

    /// Grow the instance region allocation to cover `image_count` regions.
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
        if image_count <= self.inst_regions {
            return Ok(());
        }
        crate::elog!(
            "instance regions: {} -> {} (actual swapchain images)",
            self.inst_regions,
            image_count
        );
        let device = ctx.device();
        let mem_props = ctx
            .instance()
            .get_physical_device_memory_properties(ctx.pdevice);
        // allocate the replacement FIRST: on any failure the old allocation
        // stays intact and usable (transactional growth)
        let (inst_buf, inst_mem, inst_map, inst_buf_bytes) =
            unsafe { alloc_instance_regions(device, &mem_props, image_count)? };
        // replacement fully initialized — only now retire the old allocation
        // (unmap, then destroy the buffer to release the memory binding, then free)
        unsafe { device.unmap_memory(self.inst_mem) };
        unsafe { device.destroy_buffer(self.inst_buf, None) };
        unsafe { device.free_memory(self.inst_mem, None) };
        self.inst_buf = inst_buf;
        self.inst_mem = inst_mem;
        self.inst_map = inst_map;
        self.inst_regions = image_count;
        self.inst_buf_bytes = inst_buf_bytes;
        Ok(())
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        // unmap, then destroy the buffer (releases the memory binding), then free
        device.unmap_memory(self.inst_mem);
        device.destroy_buffer(self.inst_buf, None);
        device.free_memory(self.inst_mem, None);
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
                let p = (x as f32 - w / 2.0 + 0.5, y as f32 - h / 2.0 + 0.5);
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

#[cfg(test)]
mod tests {
    use super::{check_cached_capacity, validate_section, INSTANCE_F32, MAX_INSTANCES};

    #[test]
    fn cached_push_at_and_over_limit() {
        // exactly MAX_INSTANCES is fine
        let full = MAX_INSTANCES * INSTANCE_F32;
        assert!(check_cached_capacity(0, full).is_ok());
        // one more instance exceeds the limit
        assert!(check_cached_capacity(0, full + INSTANCE_F32).is_err());
        // partially filled list: remaining space is ok, +1 is not
        let half = (MAX_INSTANCES / 2) * INSTANCE_F32;
        assert!(check_cached_capacity(half, half).is_ok());
        assert!(check_cached_capacity(half, half + INSTANCE_F32).is_err());
        // stride mismatch and overflow are rejected
        assert!(check_cached_capacity(0, full + 1).is_err());
        assert!(check_cached_capacity(usize::MAX, full).is_err());
    }

    #[test]
    fn section_range_bounds() {
        let uploaded = 100usize;
        assert!(validate_section(0, 100, uploaded).is_ok());
        assert!(validate_section(50, 50, uploaded).is_ok());
        // beyond the uploaded content of this image
        assert!(validate_section(50, 51, uploaded).is_err());
        // beyond MAX_INSTANCES even if uploaded were larger
        assert!(validate_section(0, MAX_INSTANCES + 1, uploaded).is_err());
        assert!(validate_section(MAX_INSTANCES, 1, uploaded).is_err());
        // checked arithmetic
        assert!(validate_section(usize::MAX, 1, uploaded).is_err());
    }
}
