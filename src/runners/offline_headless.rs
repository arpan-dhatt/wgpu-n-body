use crate::{sims, sims::Simulator};
use anyhow::Context;

pub struct OfflineHeadless<T>
where
    T: Simulator,
{
    sim: T,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<T> OfflineHeadless<T>
where
    T: Simulator,
{
    pub async fn new(
        sim_params: sims::SimParams,
        init_fn: fn(&sims::SimParams) -> Vec<sims::Particle>,
    ) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .context("Failed to get WGPU Adapter")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .context("Failed to create logical device and queue")?;

        let sim = Simulator::new(&device, sim_params, init_fn)?;

        Ok(Self { sim, device, queue })
    }

    pub fn step(&mut self) {
        let encoder = self.sim.encode(&self.device, &self.queue);
        self.queue.submit(Some(encoder.finish()));

        self.device.poll(wgpu::Maintain::Wait);
    }
}
