use std::sync::Arc;

use wgpu::{
    Buffer, BufferUsages, ComputePipeline, Device, Queue,
    util::{BufferInitDescriptor, DeviceExt},
    wgt::bytemuck_wrapper,
};

pub struct WgpuBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

async fn setup() {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();

    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
    let shader = device.create_shader_module(wgpu::include_wgsl!("intro.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Introduction Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });
}

fn create_input_buffer<I: IntoIterator<Item = BufferUsages>>(
    device: &Device,
    data: Vec<f32>,
    label: &str,
    usages: I,
) -> Buffer {
    let usage = usages.into_iter().fold(BufferUsages::empty(), |a, b| a | b);
    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage: usage,
    });
    buffer
}

fn create_output_buffer<I: IntoIterator<Item = BufferUsages>>(
    device: &Device,
    input_buffer: &Buffer,
    label: &str,
    usages: I,
    mapped_at_creation: bool,
) -> Buffer {
    let usage = usages.into_iter().fold(BufferUsages::empty(), |a, b| a | b);
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: input_buffer.size(),
        usage: usage,
        mapped_at_creation,
    });

    output_buffer
}
