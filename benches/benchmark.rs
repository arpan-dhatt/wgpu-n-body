use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use wgpu_n_body::{
    inits,
    runners::OfflineHeadless,
    sims::{NaiveSim, SimParams, TreeSim},
};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn criterion_benchmark(c: &mut Criterion) {
    static KB: usize = 8192;
    let mut naive_group = c.benchmark_group("naive");
    for size in [KB, KB * 2, KB * 4, KB * 8, KB * 16].iter() {
        naive_group.throughput(Throughput::Elements(*size as u64));
        naive_group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let sim_params = SimParams {
                particle_num: size as u32,
                ..SimParams::default()
            };
            let mut runner = pollster::block_on(OfflineHeadless::<NaiveSim>::new(
                sim_params,
                wgpu_n_body::sims::AddParams::NaiveSimParams,
                inits::uniform_init,
            ))
            .unwrap();
            b.iter(|| runner.step());
        });
    }
    naive_group.finish();

    let mut tree_group = c.benchmark_group("tree");
    for size in [KB, KB * 2, KB * 4, KB * 8, KB * 16].iter() {
        tree_group.throughput(Throughput::Elements(*size as u64));
        tree_group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let sim_params = SimParams {
                particle_num: size as u32,
                ..SimParams::default()
            };
            let mut runner = pollster::block_on(OfflineHeadless::<TreeSim>::new(
                sim_params,
                wgpu_n_body::sims::AddParams::TreeSimParams{ theta: 0.75 },
                inits::uniform_init,
            ))
            .unwrap();
            b.iter(|| runner.step());
        });
    }
    tree_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
