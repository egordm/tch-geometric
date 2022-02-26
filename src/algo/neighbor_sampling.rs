use rand::{Rng};
use rand::rngs::SmallRng;
use crate::data::graph::{CscGraph, EdgeIndexBuilder};
use crate::utils::{EdgePtr, replacement_sampling_range, reservoir_sampling, reservoir_sampling_weighted};
use crate::utils::types::{NodeIdx, NodePtr};

pub fn neighbor_sampling_homogenous_weighted<
>(
    rng: &mut impl Rng,
    graph: &CscGraph,
    weights: &[f64],
    inputs: &[NodeIdx],
    num_neighbors: &[usize],
) -> (Vec<NodeIdx>, EdgeIndexBuilder, Vec<(NodePtr, EdgePtr)>) {
    // Initialize some data structures for the sampling process
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut layer_offsets: Vec<(NodePtr, EdgePtr)> = Vec::new();
    let mut edge_index = EdgeIndexBuilder::new();

    samples.extend_from_slice(inputs);

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors.iter().cloned() {
        // Initialize temporary arrays for the sampling process
        let tmp0 = (0..num_samples).collect::<Vec<_>>();
        let mut tmp1 = vec![0_usize; num_samples];

        // Add layer offset
        layer_offsets.push((samples.len() as NodePtr, edge_index.len() as EdgePtr));

        for i in begin..end {
            let w = samples[i];
            let neighbors_range = graph.neighbors_range(w);

            if neighbors_range.is_empty() {
                continue;
            }

            let neighbors = graph.neighbors_slice(w);
            let neighbors_weights = &weights[neighbors_range.clone()];

            let sampled_iter = if num_samples > neighbors.len() {
                // If not enough samples, sample all neighbors
                tmp0[0..neighbors.len()].iter()
            } else {
                // Otherwise, sample without replacement
                reservoir_sampling_weighted(
                    rng, (0..neighbors.len()).zip(neighbors_weights.iter().cloned()), &mut tmp1,
                );
                tmp1.iter()
            }.cloned();

            for neighbor_idx in sampled_iter {
                let v = neighbors[neighbor_idx];
                let j = samples.len();
                let edge_ptr = neighbors_range.start + neighbor_idx;

                samples.push(v);
                edge_index.push_edge(j as i64, i as i64, edge_ptr as i64);
            }
        }

        begin = end;
        end = samples.len();
    }

    (
        samples,
        edge_index,
        layer_offsets,
    )
}

pub fn neighbor_sampling_homogenous<
    const REPLACE: bool,
>(
    rng: &mut SmallRng,
    graph: &CscGraph,
    inputs: &[NodeIdx],
    num_neighbors: &[usize],
) -> (Vec<NodeIdx>, EdgeIndexBuilder, Vec<(NodePtr, EdgePtr)>) {
    // Initialize some data structures for the sampling process
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut layer_offsets: Vec<(NodePtr, EdgePtr)> = Vec::new();
    let mut edge_index = EdgeIndexBuilder::new();

    samples.extend_from_slice(inputs);

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors.iter().cloned() {
        // Initialize temporary arrays for the sampling process
        let tmp0 = (0..num_samples).collect::<Vec<_>>();
        let mut tmp1 = vec![0_usize; num_samples];

        // Add layer offset
        layer_offsets.push((samples.len() as NodePtr, edge_index.len() as EdgePtr));

        for i in begin..end {
            let w = samples[i];
            let neighbors_range = graph.neighbors_range(w);

            if neighbors_range.is_empty() {
                continue;
            }

            let neighbors = graph.neighbors_slice(w);

            let sampled_iter = if !REPLACE && num_samples > neighbors.len() {
                // If not enough samples, sample all neighbors
                tmp0[0..neighbors.len()].iter()
            } else {
                // Otherwise, sample without replacement
                if REPLACE {
                    replacement_sampling_range(rng, &(0..neighbors.len()), &mut tmp1);
                } else {
                    reservoir_sampling(rng, 0..neighbors.len(), &mut tmp1);
                }
                tmp1.iter()
            }.cloned();

            for neighbor_idx in sampled_iter {
                let v = neighbors[neighbor_idx];
                let j = samples.len();
                let edge_ptr = neighbors_range.start + neighbor_idx;

                samples.push(v);
                edge_index.push_edge(j as i64, i as i64, edge_ptr as i64);
            }
        }

        begin = end;
        end = samples.len();
    }

    (
        samples,
        edge_index,
        layer_offsets,
    )
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use rand::{Rng, SeedableRng};
    use crate::algo::neighbor_sampling::neighbor_sampling_homogenous;
    use crate::data::{CscGraph, CscGraphData};
    use crate::data::tests::load_karate_graph;

    #[test]
    pub fn test_neighbor_sampling_homogenous_weighted() {
        let (x, _, edge_index) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphData::try_from_edge_index(&edge_index, x.size()[0]).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();
        let weights = (0..graph.edge_count()).map(|_| rng.gen_range(0.2..5.0)).collect::<Vec<f64>>();
        let inputs = vec![0_i64, 1, 4, 5];
        let num_neighbors = vec![4, 3];

        let (samples, edge_index, layer_offsets) = super::neighbor_sampling_homogenous_weighted(
            &mut rng,
            &graph,
            &weights,
            &inputs,
            &num_neighbors
        );

        for (j, i) in edge_index.rows.iter().zip(edge_index.cols.iter()) {
            let v = samples[*j as usize];
            let w = samples[*i as usize];
            assert!(graph.has_edge(v, w));
        }

        let mut counts = vec![0_usize; samples.len()];
        for i in layer_offsets.iter().last().unwrap().0 as usize..samples.len() {
            counts[i] += 1;
        }

        for (j, i) in edge_index.rows.iter().rev().zip(edge_index.cols.iter().rev()) {
            counts[*i as usize] += counts[*j as usize];
        }

        let mut begin = 0;
        for (i, (end, _)) in layer_offsets.iter().cloned().enumerate() {
            let max_neighbors: usize = num_neighbors[0..num_neighbors.len() - i].iter().product();

            for i in begin..end {
                assert!((1..=max_neighbors).contains(&counts[i as usize]));
            }
            begin = end;
        }
    }

    #[test]
    pub fn test_neighbor_sampling_homogenous() {
        let (x, _, edge_index) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphData::try_from_edge_index(&edge_index, x.size()[0]).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();
        let inputs = vec![0_i64, 1, 4, 5];
        let num_neighbors = vec![4, 3];

        let (samples, edge_index, layer_offsets) = neighbor_sampling_homogenous::<true>(
            &mut rng,
            &graph,
            &inputs,
            &num_neighbors
        );

        for (j, i) in edge_index.rows.iter().zip(edge_index.cols.iter()) {
            let v = samples[*j as usize];
            let w = samples[*i as usize];
            assert!(graph.has_edge(v, w));
        }

        let mut counts = vec![0_usize; samples.len()];
        for i in layer_offsets.iter().last().unwrap().0 as usize..samples.len() {
            counts[i] += 1;
        }

        for (j, i) in edge_index.rows.iter().rev().zip(edge_index.cols.iter().rev()) {
            counts[*i as usize] += counts[*j as usize];
        }

        let mut begin = 0;
        for (i, (end, _)) in layer_offsets.iter().cloned().enumerate() {
            let max_neighbors: usize = num_neighbors[0..num_neighbors.len() - i].iter().product();

            for i in begin..end {
                assert!((1..=max_neighbors).contains(&counts[i as usize]));
            }
            begin = end;
        }
    }

}