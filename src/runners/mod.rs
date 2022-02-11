mod offline_headless;
mod online_renderer;

pub use offline_headless::OfflineHeadless;
pub use online_renderer::OnlineRenderer;

use anyhow::Context;

async fn get_device_and_queue(
    adapter: &wgpu::Adapter,
) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool)> {
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1073741824,
                    ..wgpu::Limits::default()
                },
            },
            None,
        )
        .await
        .unwrap_or(
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits {
                            max_storage_buffer_binding_size: 1073741824,
                            ..wgpu::Limits::default()
                        },
                    },
                    None,
                )
                .await
                .context("Failed to create logical device and queue")?,
        );
    let mappable_primary_buffers = device
        .features()
        .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
    Ok((device, queue, mappable_primary_buffers))
}
