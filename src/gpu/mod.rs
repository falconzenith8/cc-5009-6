//! wgpu renderer: shadow-map pass per light → opaque pass → fullscreen
//! bounded-step volumetric pass → composite pass. Parameter edits are
//! uploaded to GPU uniform buffers every frame (`queue.write_buffer`), so
//! UI adjustments take effect immediately. Device/resource recovery
//! recreates the device (headless mode) or all GPU resources (shared mode)
//! from kernel-held parameters.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};

use crate::cpu::Frame;
use crate::kernel::integrate::{GpuParams, PreparedLight};
use crate::kernel::math::Mat4;
use crate::kernel::params::{StageVolParams, MAX_LIGHTS};
use crate::kernel::scene::Scene;

const COMMON_WGSL: &str = include_str!("../../shaders/common.wgsl");
const SCENE_WGSL: &str = include_str!("../../shaders/scene.wgsl");
const VOLUME_WGSL: &str = include_str!("../../shaders/volume.wgsl");
const COMPOSITE_WGSL: &str = include_str!("../../shaders/composite.wgsl");

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SHFACTOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub const COMPOSITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const OPAQUE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SHADOW_SIZE: u32 = 1024;

pub struct GpuRenderer {
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Present only when the renderer owns the device (headless/CLI mode).
    instance: Option<wgpu::Instance>,
    adapter: Option<wgpu::Adapter>,
    res: Resources,
    errors: Arc<Mutex<Vec<String>>>,
    pub device_label: String,
}

struct Resources {
    params_buf: wgpu::Buffer,
    scene_buf: wgpu::Buffer,
    scene_buf_capacity: usize,
    light_vp_bufs: [wgpu::Buffer; MAX_LIGHTS],
    #[allow(dead_code)] // kept alive for the shadow views
    shadow_texs: [wgpu::Texture; MAX_LIGHTS],
    shadow_views: [wgpu::TextureView; MAX_LIGHTS],
    cam_depth_view: wgpu::TextureView,
    opaque_view: wgpu::TextureView,
    volume_tex: wgpu::Texture,
    volume_view: wgpu::TextureView,
    shfactor_tex: wgpu::Texture,
    shfactor_view: wgpu::TextureView,
    composite_tex: wgpu::Texture,
    composite_view: wgpu::TextureView,
    #[allow(dead_code)] // kept alive for the bind groups
    cmp_sampler: wgpu::Sampler,
    shadow_pipeline: wgpu::RenderPipeline,
    scene_pipeline: wgpu::RenderPipeline,
    volume_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    scene_bg: wgpu::BindGroup,
    volume_bg: wgpu::BindGroup,
    composite_bg: wgpu::BindGroup,
    shadow_bgs: [wgpu::BindGroup; MAX_LIGHTS],
}

impl GpuRenderer {
    /// Headless mode: creates its own instance/adapter/device.
    pub async fn init_headless(width: u32, height: u32) -> Result<GpuRenderer> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| anyhow!("请求 GPU 适配器失败: {e}"))?;
        let (device, queue) = request_device(&adapter).await?;
        let device_label = adapter.get_info().name;
        let errors = install_error_handler(&device);
        let res = build_resources(&device, width, height)?;
        Ok(GpuRenderer {
            width,
            height,
            device,
            queue,
            instance: Some(instance),
            adapter: Some(adapter),
            res,
            errors,
            device_label,
        })
    }

    /// Shared mode: borrows an existing device/queue (egui_wgpu render
    /// state). Resource recovery rebuilds pipelines/textures on this device.
    pub fn init_shared(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        device_label: String,
    ) -> Result<GpuRenderer> {
        let errors = install_error_handler(&device);
        let res = build_resources(&device, width, height)?;
        Ok(GpuRenderer {
            width,
            height,
            device,
            queue,
            instance: None,
            adapter: None,
            res,
            errors,
            device_label,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Texture the UI displays; re-register with egui after resize/recover.
    pub fn composite_view(&self) -> &wgpu::TextureView {
        &self.res.composite_view
    }

    /// GPU/uncaptured errors collected since the last call (failure
    /// feedback channel surfaced in the UI status bar).
    pub fn take_errors(&self) -> Vec<String> {
        core::mem::take(&mut *self.errors.lock().unwrap())
    }

    pub fn backend_name(&self) -> &'static str {
        if self.instance.is_some() {
            "gpu-headless"
        } else {
            "gpu-shared"
        }
    }

    /// Recreate size-dependent textures and bind groups.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == self.width && height == self.height {
            return Ok(());
        }
        self.width = width;
        self.height = height;
        self.res = build_resources(&self.device, width, height)?;
        Ok(())
    }

    /// Device/resource recovery. Headless mode re-requests the device and
    /// rebuilds everything; shared mode rebuilds all GPU resources on the
    /// existing device. Kernel parameters live on the CPU and are re-uploaded
    /// on the next `render`, so nothing is lost.
    pub fn recover(&mut self) -> Result<()> {
        if let (Some(instance), Some(adapter)) = (&self.instance, &self.adapter) {
            let _ = instance; // instance kept alive for adapter validity
            let (device, queue) = pollster::block_on(request_device(adapter))
                .context("设备恢复：重新请求设备失败")?;
            self.device = device;
            self.queue = queue;
            self.errors = install_error_handler(&self.device);
        }
        self.res = build_resources(&self.device, self.width, self.height)
            .context("设备恢复：重建资源失败")?;
        Ok(())
    }

    /// Render the four passes into the internal textures (no readback).
    /// This is the path the UI uses every frame; parameter edits reach the
    /// GPU here via `queue.write_buffer`.
    pub fn render_to_textures(&mut self, params: &StageVolParams, time: f32) -> Result<()> {
        let (w, h) = (self.width, self.height);
        let gpu_params = GpuParams::build(params, w, h, time);
        self.queue
            .write_buffer(&self.res.params_buf, 0, bytemuck::bytes_of(&gpu_params));

        // Scene triangles (occluder may have moved) — real-time param update.
        let scene = Scene::build(params.occluder_x_at(time), params.occluder_visible, params.occluder_scale);
        let tris = scene.triangles();
        if tris.len() > self.res.scene_buf_capacity {
            self.res.scene_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene-vtx"),
                size: (tris.len() * 4).next_power_of_two() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.res.scene_buf_capacity = tris.len().max(self.res.scene_buf_capacity);
        }
        self.queue
            .write_buffer(&self.res.scene_buf, 0, bytemuck::cast_slice(&tris));
        let vtx_count = (tris.len() / 9) as u32;

        // Per-light view-projection uniforms for the shadow passes.
        for (i, l) in params.lights.iter().take(MAX_LIGHTS).enumerate() {
            let p = PreparedLight::prepare(l);
            let cols: Mat4 = p.view_proj;
            self.queue.write_buffer(
                &self.res.light_vp_bufs[i],
                0,
                bytemuck::cast_slice(&cols.to_cols_array()),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // 1. Shadow depth passes (occluders block in-volume light).
        for i in 0..MAX_LIGHTS {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.res.shadow_views[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.res.shadow_pipeline);
            pass.set_bind_group(0, &self.res.shadow_bgs[i], &[]);
            pass.set_vertex_buffer(0, self.res.scene_buf.slice(..));
            pass.draw(0..vtx_count, 0..1);
        }

        // 2. Opaque scene pass.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("opaque"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.res.opaque_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.res.cam_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.res.scene_pipeline);
            pass.set_bind_group(0, &self.res.scene_bg, &[]);
            pass.set_vertex_buffer(0, self.res.scene_buf.slice(..));
            pass.draw(0..vtx_count, 0..1);
        }

        // 3. Volumetric integration pass.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volume"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.res.volume_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.res.shfactor_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.res.volume_pipeline);
            pass.set_bind_group(0, &self.res.volume_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 4. Composite pass.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.res.composite_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.res.composite_pipeline);
            pass.set_bind_group(0, &self.res.composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Render and read back composite + volume + shadow buffers.
    pub fn render(&mut self, params: &StageVolParams, time: f32) -> Result<Frame> {
        let (w, h) = (self.width, self.height);
        self.render_to_textures(params, time)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        // Readback: composite rgba8, volume rgba16f, shadow r16f.
        let rgba_bpr = aligned_bpr(w * 4);
        let vol_bpr = aligned_bpr(w * 8);
        let sh_bpr = aligned_bpr(w * 2);
        let rgba_buf = self.readback_buffer((rgba_bpr * h) as u64, "rgba");
        let vol_buf = self.readback_buffer((vol_bpr * h) as u64, "vol");
        let sh_buf = self.readback_buffer((sh_bpr * h) as u64, "sh");
        copy_tex(&mut encoder, &self.res.composite_tex, &rgba_buf, rgba_bpr, w, h);
        copy_tex(&mut encoder, &self.res.volume_tex, &vol_buf, vol_bpr, w, h);
        copy_tex(&mut encoder, &self.res.shfactor_tex, &sh_buf, sh_bpr, w, h);
        self.queue.submit([encoder.finish()]);

        let rgba_data = map_buffer(&self.device, &rgba_buf)?;
        let vol_data = map_buffer(&self.device, &vol_buf)?;
        let sh_data = map_buffer(&self.device, &sh_buf)?;

        let npix = (w * h) as usize;
        let mut rgba = vec![0u8; npix * 4];
        let mut volume = vec![[0.0f32; 3]; npix];
        let mut transmittance = vec![1.0f32; npix];
        let mut surface_shadow = vec![1.0f32; npix];
        for y in 0..h as usize {
            let rgba_row = &rgba_data[y * rgba_bpr as usize..y * rgba_bpr as usize + w as usize * 4];
            let vol_row = &vol_data[y * vol_bpr as usize..y * vol_bpr as usize + w as usize * 8];
            let sh_row = &sh_data[y * sh_bpr as usize..y * sh_bpr as usize + w as usize * 2];
            for x in 0..w as usize {
                let i = y * w as usize + x;
                rgba[i * 4..i * 4 + 4].copy_from_slice(&rgba_row[x * 4..x * 4 + 4]);
                for c in 0..3 {
                    volume[i][c] = f16_to_f32(u16::from_le_bytes([
                        vol_row[x * 8 + c * 2],
                        vol_row[x * 8 + c * 2 + 1],
                    ]));
                }
                transmittance[i] = f16_to_f32(u16::from_le_bytes([
                    vol_row[x * 8 + 6],
                    vol_row[x * 8 + 7],
                ]));
                surface_shadow[i] =
                    f16_to_f32(u16::from_le_bytes([sh_row[x * 2], sh_row[x * 2 + 1]]));
            }
        }
        Ok(Frame { width: w, height: h, rgba, volume, transmittance, surface_shadow })
    }

    fn readback_buffer(&self, size: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }
}

async fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("stage-vol"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| anyhow!("请求 GPU 设备失败: {e}"))
}

fn install_error_handler(device: &wgpu::Device) -> Arc<Mutex<Vec<String>>> {
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        sink.lock().unwrap().push(format!("{e}"));
    }));
    errors
}

fn shader(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{body}").into()),
    })
}

fn build_resources(device: &wgpu::Device, width: u32, height: u32) -> Result<Resources> {
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: core::mem::size_of::<GpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let scene_buf_capacity = 4096usize;
    let scene_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene-vtx"),
        size: (scene_buf_capacity * 4) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let light_vp_bufs: [wgpu::Buffer; MAX_LIGHTS] = core::array::from_fn(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light-vp"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    let mk_tex = |label: &str,
                  w: u32,
                  h: u32,
                  format: wgpu::TextureFormat,
                  usage: wgpu::TextureUsages| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let shadow_texs: [wgpu::Texture; MAX_LIGHTS] = core::array::from_fn(|_| {
        mk_tex(
            "shadow-map",
            SHADOW_SIZE,
            SHADOW_SIZE,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        )
    });
    let shadow_views: [wgpu::TextureView; MAX_LIGHTS] =
        core::array::from_fn(|i| shadow_texs[i].create_view(&Default::default()));
    let cam_depth = mk_tex(
        "cam-depth",
        width,
        height,
        DEPTH_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let cam_depth_view = cam_depth.create_view(&Default::default());
    let opaque = mk_tex(
        "opaque",
        width,
        height,
        OPAQUE_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let opaque_view = opaque.create_view(&Default::default());
    let volume = mk_tex(
        "volume",
        width,
        height,
        VOLUME_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    );
    let volume_view = volume.create_view(&Default::default());
    let shfactor = mk_tex(
        "shfactor",
        width,
        height,
        SHFACTOR_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    );
    let shfactor_view = shfactor.create_view(&Default::default());
    let composite_tex = mk_tex(
        "composite",
        width,
        height,
        COMPOSITE_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    );
    let composite_view = composite_tex.create_view(&Default::default());

    let cmp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow-cmp"),
        compare: Some(wgpu::CompareFunction::LessEqual),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // --- pipelines ---
    let scene_shader = shader(device, "scene", SCENE_WGSL);
    let volume_shader = shader(device, "volume", VOLUME_WGSL);
    let composite_shader = shader(device, "composite", COMPOSITE_WGSL);

    let vtx_layout = wgpu::VertexBufferLayout {
        array_stride: 36,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
        ],
    };
    let vtx_pos_only = wgpu::VertexBufferLayout {
        array_stride: 36,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    };

    let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene-bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_depth_tex(1),
            bgl_depth_tex(2),
            bgl_cmp_sampler(3),
        ],
    });
    let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow-bgl"),
        entries: &[bgl_uniform(0)],
    });
    let volume_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("volume-bgl"),
        entries: &[
            bgl_uniform(0),
            bgl_depth_tex(1),
            bgl_depth_tex(2),
            bgl_depth_tex(3),
            bgl_cmp_sampler(4),
        ],
    });
    let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite-bgl"),
        entries: &[bgl_uniform(0), bgl_float_tex(1), bgl_float_tex(2), bgl_float_tex(3)],
    });

    let mk_layout = |bgls: &[&wgpu::BindGroupLayout]| {
        let owned: Vec<Option<&wgpu::BindGroupLayout>> = bgls.iter().map(|b| Some(*b)).collect();
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &owned,
            immediate_size: 0,
        })
    };

    let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow"),
        layout: Some(&mk_layout(&[&shadow_bgl])),
        vertex: wgpu::VertexState {
            module: &scene_shader,
            entry_point: Some("vs_shadow"),
            compilation_options: Default::default(),
            buffers: &[Some(vtx_pos_only.clone())],
        },
        fragment: None,
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let scene_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("scene"),
        layout: Some(&mk_layout(&[&scene_bgl])),
        vertex: wgpu::VertexState {
            module: &scene_shader,
            entry_point: Some("vs_scene"),
            compilation_options: Default::default(),
            buffers: &[Some(vtx_layout)],
        },
        fragment: Some(wgpu::FragmentState {
            module: &scene_shader,
            entry_point: Some("fs_scene"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: OPAQUE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("volume"),
        layout: Some(&mk_layout(&[&volume_bgl])),
        vertex: wgpu::VertexState {
            module: &volume_shader,
            entry_point: Some("vs_full"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &volume_shader,
            entry_point: Some("fs_volume"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: VOLUME_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: SHFACTOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("composite"),
        layout: Some(&mk_layout(&[&composite_bgl])),
        vertex: wgpu::VertexState {
            module: &composite_shader,
            entry_point: Some("vs_full"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_shader,
            entry_point: Some("fs_composite"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: COMPOSITE_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    // --- bind groups ---
    let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene-bg"),
        layout: &scene_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_views[0]) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&shadow_views[1]) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&cmp_sampler) },
        ],
    });
    let volume_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("volume-bg"),
        layout: &volume_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&cam_depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&shadow_views[0]) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&shadow_views[1]) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&cmp_sampler) },
        ],
    });
    let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite-bg"),
        layout: &composite_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&opaque_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&volume_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&shfactor_view) },
        ],
    });
    let shadow_bgs: [wgpu::BindGroup; MAX_LIGHTS] = core::array::from_fn(|i| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-bg"),
            layout: &shadow_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_vp_bufs[i].as_entire_binding(),
            }],
        })
    });

    Ok(Resources {
        params_buf,
        scene_buf,
        scene_buf_capacity,
        light_vp_bufs,
        shadow_texs,
        shadow_views,
        cam_depth_view,
        opaque_view,
        volume_tex: volume,
        volume_view,
        shfactor_tex: shfactor,
        shfactor_view,
        composite_tex,
        composite_view,
        cmp_sampler,
        shadow_pipeline,
        scene_pipeline,
        volume_pipeline,
        composite_pipeline,
        scene_bg,
        volume_bg,
        composite_bg,
        shadow_bgs,
    })
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_depth_tex(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_float_tex(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_cmp_sampler(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        count: None,
    }
}

fn aligned_bpr(row_bytes: u32) -> u32 {
    row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

fn copy_tex(
    encoder: &mut wgpu::CommandEncoder,
    tex: &wgpu::Texture,
    buf: &wgpu::Buffer,
    bpr: u32,
    w: u32,
    h: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

fn map_buffer(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<u8>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .context("buffer map channel closed")?
        .map_err(|e| anyhow!("buffer map 失败: {e}"))?;
    let data = slice.get_mapped_range().map_err(|e| anyhow!("读取映射缓冲失败: {e}"))?.to_vec();
    buf.unmap();
    Ok(data)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31 // ±0
        } else {
            // subnormal half: value = frac · 2^-24, normalize to f32
            let mut f = frac;
            let mut e = 113u32; // 127 - 14
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | (e << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16_decode_specials() {
        assert_eq!(super::f16_to_f32(0x0000), 0.0);
        assert_eq!(super::f16_to_f32(0x8000), -0.0);
        assert_eq!(super::f16_to_f32(0x3c00), 1.0);
        assert_eq!(super::f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(super::f16_to_f32(0x7bff), 65504.0);
        assert!(super::f16_to_f32(0x7c00).is_infinite());
    }
}
