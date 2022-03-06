use std::convert::TryFrom;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use tch::Tensor;
use tch_geometric::data::{CsrGraphStorage, CsrGraph, load_karate_graph};
use tch_geometric::algo::negative_sampling::{negative_sample_neighbors_homogenous};

pub fn internal_benchmark(c: &mut Criterion) {
    let (x, _, coo_graph) = load_karate_graph();

    let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

    let node_count = x.size()[0];
    let graph_data = CsrGraphStorage::try_from(&coo_graph).unwrap();
    let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();
    let inputs = Tensor::of_slice(&[0_i64, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    c.bench_function("negative_sample_neighbors", |b| b.iter(|| {
        negative_sample_neighbors_homogenous(
            &mut rng,
            &graph,
            (node_count, node_count),
            &inputs,
            10,
            5,
        ).unwrap()
    }));
}

criterion_group!(benches, internal_benchmark);
criterion_main!(benches);
