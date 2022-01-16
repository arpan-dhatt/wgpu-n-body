use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use wgpu_n_body::{
    inits,
    runners::OfflineHeadless,
    sims::{NaiveSim, SimParams},
};

fn criterion_benchmark(c: &mut Criterion) {
    static KB: usize = 8192;
    let mut group = c.benchmark_group("naive");
    for size in [KB, KB * 2, KB * 4, KB * 8].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let sim_params = SimParams {
                particle_num: size as u32,
                ..SimParams::default()
            };
            let mut runner = pollster::block_on(OfflineHeadless::<NaiveSim>::new(sim_params, inits::disc_init)).unwrap();
            b.iter(|| runner.step());
        });
    }
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
