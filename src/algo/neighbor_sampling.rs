use std::collections::HashMap;
use std::convert::TryInto;
use rand::{Rng, SeedableRng};
use tch::Tensor;
use crate::data::graph::{CscGraph, EdgeIndexBuilder};
use crate::utils::{EdgePtr, replacement_sampling_range, reservoir_sampling, reservoir_sampling_weighted};
use crate::utils::tensor::{TensorResult, try_tensor_to_slice};
use crate::utils::types::{NodeIdx, NodePtr};

pub fn neighbor_sampling_homogenous_weighted<
>(
    rng: &mut impl Rng,
    graph: &CscGraph,
    weights: &[f64],
    inputs: &[NodeIdx],
    num_neighbors: &[usize],
) -> (Vec<NodeIdx>, EdgeIndexBuilder, Vec<(NodePtr, EdgePtr)>) {
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut layer_offsets: Vec<(NodePtr, EdgePtr)> = Vec::new();
    let mut edge_index = EdgeIndexBuilder::new();

    samples.extend_from_slice(inputs);

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors.iter().cloned() {
        layer_offsets.push((samples.len() as NodePtr, edge_index.len() as EdgePtr));

        let tmp0 = (0..num_samples).collect::<Vec<_>>();
        let mut tmp1 = vec![0_usize; num_samples];

        for i in begin..end {
            let w = samples[i];
            let neighbors_range = graph.neighbors_range(w);

            if neighbors_range.is_empty() {
                continue;
            }

            let neighbors = graph.neighbors_slice(w);
            let neighbors_weights = &weights[neighbors_range.clone()];

            let sampled_iter = if num_samples > neighbors.len() {
                tmp0[0..neighbors.len()].iter()
            } else {
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
                edge_index.push_edge(i as i64, j as i64, edge_ptr as i64);
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
    const DIRECTED: bool,
>(
    graph: &CscGraph,
    input_node: Tensor,
    num_neighbors: Vec<i64>,
) -> TensorResult<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
)> {
    let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

    // Initialize some data structures for the sampling process:
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut to_local_node: HashMap<NodeIdx, NodePtr> = HashMap::new();

    let input_node_data = try_tensor_to_slice::<i64>(&input_node)?;

    for (i, v) in input_node_data.iter().enumerate() {
        samples.push(*v);
        to_local_node.insert(*v, i as NodePtr);
    }

    let (mut rows, mut cols, mut edges) = (Vec::<NodeIdx>::new(), Vec::<NodeIdx>::new(), Vec::<NodeIdx>::new());

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors {
        let mut tmp_sample = vec![0; num_samples as usize];

        for i in begin..end {
            let w = samples[i];
            let neighbors_range = graph.neighbors_range(w);
            let neighbors = graph.neighbors_slice(w);

            if neighbors.is_empty() {
                continue;
            }

            if (num_samples < 0) || (!REPLACE && (num_samples > neighbors.len() as i64)) {
                // If num_samples is negative,
                // or if we are not replacing and num_samples is greater than the number of edges,
                // then we sample all the edges

                for (v, offset) in neighbors.iter().cloned().zip(neighbors_range.into_iter()) {
                    // register node in output list
                    let res = to_local_node.insert(v, samples.len() as NodePtr);

                    samples.push(v); // add neighbor to output list
                    if DIRECTED {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset as NodeIdx);
                    }
                }
            } else {
                // Randomly sample num_samples nodes from the neighbors of w
                if REPLACE {
                    replacement_sampling_range(&mut rng, &neighbors_range, &mut tmp_sample);
                } else {
                    reservoir_sampling(&mut rng, neighbors_range, &mut tmp_sample);
                }

                for (v, offset) in tmp_sample.iter().cloned().map(|o| (graph.indices[o], o)) {
                    // register node in output list
                    let res = to_local_node.insert(v, samples.len() as NodePtr);

                    samples.push(v); // add neighbor to output list
                    if DIRECTED {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset as NodeIdx);
                    }
                }
            }
        }
        // set begin to the old end to avoid sampling same nodes twice. and repeat
        begin = end;
        end = samples.len();
    }

    if !DIRECTED {
        // if undirected, we need to add edges in both directions
        for (i, w) in samples.iter().enumerate() {
            // for each sample
            let neighbors_range = graph.neighbors_range(*w);
            let neighbors_range = neighbors_range.start as NodeIdx..neighbors_range.end as NodeIdx;
            let neighbors = graph.neighbors_slice(*w);

            for (v, offset) in neighbors.iter().zip(neighbors_range) {
                let res = to_local_node.get(v); // find neighbor in output list
                if let Some(res) = res.cloned() {
                    // if neighbor is in output list
                    cols.push(i as NodeIdx); // add sample to row
                    rows.push(res); // add edge to output graph
                    edges.push(offset);
                }
            }
        }
    }

    Ok((
        samples.try_into().expect("Cant convert vec into tensor"),
        rows.try_into().expect("Cant convert vec into tensor"),
        cols.try_into().expect("Cant convert vec into tensor"),
        edges.try_into().expect("Cant convert vec into tensor"),
    ))
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use rand::{Rng, SeedableRng};
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

        for (i, j) in edge_index.cols.iter().zip(edge_index.rows.iter()) {
            let v = samples[*i as usize];
            let w = samples[*j as usize];
            assert!(graph.has_edge(v, w));
        }

        let mut counts = vec![0_usize; samples.len()];
        for i in layer_offsets.iter().last().unwrap().0 as usize..samples.len() {
            counts[i] += 1;
        }

        for (i, j) in edge_index.cols.iter().rev().zip(edge_index.rows.iter().rev()) {
            counts[*j as usize] += counts[*i as usize];
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