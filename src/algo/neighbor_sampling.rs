use std::collections::HashMap;
use std::convert::TryInto;
use rand::SeedableRng;
use tch::Tensor;
use crate::data::graph::CsrGraph;
use crate::utils::{replacement_sampling_range, reservoir_sampling};
use crate::utils::tensor::{TensorResult, try_tensor_to_slice};
use crate::utils::types::{NodeIdx, NodePtr};

pub fn neighbor_sampling_homogenous<
    const REPLACE: bool,
    const DIRECTED: bool,
>(
    graph: &CsrGraph,
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
                continue
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
                    reservoir_sampling(&mut rng, neighbors_range.into_iter(), &mut tmp_sample);
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