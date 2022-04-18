use rand::{Rng};
use rand::rngs::SmallRng;
use tch::{Kind, Scalar, Tensor};
use crate::data::EdgeAttr;
use crate::data::graph::CsrGraph;
use crate::utils::reservoir_sampling;
use crate::utils::tensor::{TensorResult, try_tensor_to_slice, try_tensor_to_slice_mut};

#[allow(non_snake_case)]
pub fn random_walk(
    rng: &mut SmallRng,
    graph: &CsrGraph,
    start: &Tensor,
    walk_length: i64,
    p: f32,
    q: f32,
) -> TensorResult<Tensor> {
    let L = (walk_length + 1) as usize;
    let mut walks = Tensor::full(
        &[start.size()[0], L as i64],
        Scalar::from(-1_i64),
        (Kind::Int64, start.device()),
    );

    let start_data = try_tensor_to_slice::<i64>(start)?;
    let walks_data = try_tensor_to_slice_mut::<i64>(&mut walks)?;

    // Normalize the weights to compute rejection probabilities
    let max_prob = vec![1.0 / p, 1.0, 1.0 / q].into_iter()
        .max_by(|a, b| a.partial_cmp(b).expect("p or q may not be 0 or nan")).unwrap();
    // rejection prob for back to the previous node
    let prob0 = 1.0 / p / max_prob;
    // rejection prob for visiting the node with the distance of 1 between the previous node
    let prob1 = 1.0 / max_prob;
    // rejection prob for visiting the node with the distance of 2 between the previous node
    let prob2 = 1.0 / q / max_prob;

    for (i, n) in start_data.iter().enumerate() {
        let mut prev = -1;
        let mut cur = *n;
        walks_data[i * L] = cur;

        for l in 0..walk_length as usize {
            let neighbors = graph.neighbors_slice(cur);
            if neighbors.is_empty() {
                break;
            }

            // TODO: move this logic to a walker state function
            // Neighbor sampling
            let mut next;
            loop {
                next = neighbors[rng.gen_range(0..neighbors.len())];
                let r = rng.gen_range(0.0..1.0);

                if next == prev {
                    // back to the previous node
                    if r < prob0 { break; }
                } else if graph.has_edge(next, prev) {
                    // visiting the node with the distance of 1 between the previous node
                    if r < prob1 { break; }
                } else if r < prob2 {
                    // visiting the node with the distance of 2 between the previous node
                    break;
                }
            }

            prev = cur;
            cur = next;
            walks_data[i * L + l + 1] = cur;
        }
    }

    Ok(walks)
}

const NAN_TIMESTAMP: i64 = -1_i64;

#[allow(non_snake_case)]
pub fn tempo_random_walk(
    rng: &mut SmallRng,
    graph: &CsrGraph,
    node_timestamps: &[i64],
    edge_timestamps: &EdgeAttr<i64>,
    start: &Tensor,
    start_timestamps: &Tensor,
    walk_length: i64,
    window: (i64, i64),
) -> TensorResult<(Tensor, Tensor)> {
    let L = walk_length as usize;
    let mut walks = Tensor::full(
        &[start.size()[0], L as i64],
        Scalar::from(-1_i64),
        (Kind::Int64, start.device()),
    );
    let mut walks_timestamps = Tensor::full(
        &[start.size()[0], L as i64],
        Scalar::from(-1_i64),
        (Kind::Int64, start.device()),
    );

    let start_data = try_tensor_to_slice::<i64>(start)?;
    let start_timestamps = try_tensor_to_slice::<i64>(start_timestamps)?;

    let walks_data = try_tensor_to_slice_mut::<i64>(&mut walks)?;
    let walks_timestamps_data = try_tensor_to_slice_mut::<i64>(&mut walks_timestamps)?;

    for (i, n) in start_data.iter().enumerate() {
        let mut cur = *n;
        let i_timestamp = start_timestamps[i];
        let i_window = i_timestamp + window.0 .. i_timestamp + window.1;

        walks_data[i * L] = cur;
        walks_timestamps_data[i * L] = i_timestamp;


        for l in 0..(walk_length - 1) as usize {
            let neighbors = edge_timestamps.get_range(graph.neighbors_range(cur)).into_iter()
                .zip(graph.neighbors_slice(cur))
                .map(|(&edge_timestamp, &node_idx)| {
                    let node_timestamp = if edge_timestamp != NAN_TIMESTAMP {
                        edge_timestamp
                    } else {
                        node_timestamps[node_idx as usize]
                    };

                    (node_timestamp, node_idx)
                })
                .filter(|(node_timestamp, node_idx)| {
                    if *node_timestamp == NAN_TIMESTAMP || i_timestamp == NAN_TIMESTAMP {
                        return true;
                    }

                    if i_window.contains(&node_timestamp) {
                        return true;
                    }

                    return false;
                });

            let mut next_tmp = [(-1, -1); 1];
            let success = reservoir_sampling(rng, neighbors, &mut next_tmp);

            if success == 0 {
                // Restart random walk
                let restart_i = i * L + rng.gen_range(0..l + 1);
                next_tmp[0] = (walks_timestamps_data[restart_i], walks_data[restart_i]);
            }

            let (next_timestamp, next) = next_tmp[0];
            cur = next;
            walks_data[i * L + l + 1] = cur;
            walks_timestamps_data[i * L + l + 1] = next_timestamp;
        }
    }

    Ok((walks, walks_timestamps))
}

#[cfg(test)]
mod tests {
    use std::convert::{TryFrom};
    use rand::{Rng, SeedableRng};
    use tch::Tensor;
    use crate::algo::random_walk::{random_walk, tempo_random_walk};
    use crate::data::{CsrGraphStorage, CsrGraph, EdgeAttr};
    use crate::data::load_karate_graph;
    use crate::utils::tensor::try_tensor_to_slice;

    #[test]
    fn test_randomwalk() {
        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let (_x, _, coo_graph) = load_karate_graph();

        let graph_data = CsrGraphStorage::try_from(&coo_graph).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let start = Tensor::of_slice(&[0_i64, 1, 2, 3]);

        let walks = random_walk(
            &mut rng,
            &graph,
            &start,
            10,
            1.0,
            1.5,
        ).unwrap();

        // Check validity of the walks
        for (i, head) in Vec::<i64>::from(start.as_ref()).into_iter().enumerate() {
            let slice = walks.select(0, i as i64);
            let walk = try_tensor_to_slice::<i64>(&slice).unwrap();

            assert_eq!(walk[0], head);
            for (prev, curr) in walk.iter().zip(walk.iter().skip(1)) {
                assert!(graph.has_edge(*prev, *curr));
            }
        }
    }

    #[test]
    fn test_tempo_randomwalk() {
        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let (_x, _, coo_graph) = load_karate_graph();

        let graph_data = CsrGraphStorage::try_from(&coo_graph).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();
        let node_timestamps = (0..graph.node_count()).map(|_| rng.gen_range(-1..5) as i64).collect::<Vec<_>>();
        let edge_timestamps = (0..graph.edge_count()).map(|_| rng.gen_range(-1..5) as i64).collect::<Vec<_>>();


        let start = Tensor::of_slice(&[0_i64, 1, 2, 3]);
        let start_timestamps_data = &[0_i64, -1, 2, 3];
        let start_timestamps = Tensor::of_slice(start_timestamps_data);


        let (walks, walk_timestamps) = tempo_random_walk(
            &mut rng,
            &graph,
            &node_timestamps,
            &EdgeAttr::new(&edge_timestamps),
            &start,
            &start_timestamps,
            10,
            (0, 2)
        ).unwrap();

        // Check validity of the walks
        for (i, head) in Vec::<i64>::from(start.as_ref()).into_iter().enumerate() {
            let slice = walks.select(0, i as i64);
            let walk = try_tensor_to_slice::<i64>(&slice).unwrap();
            let slice_timestamps = walk_timestamps.select(0, i as i64);
            let walk_timestamps = try_tensor_to_slice::<i64>(&slice_timestamps).unwrap();

            let head_timestamp = start_timestamps_data[i];

            assert_eq!(walk[0], head);
            for (prev, curr) in walk.iter().zip(walk.iter().skip(1)) {
                // assert!(graph.has_edge(*prev, *curr)); // TODO: there doesn't have to be and edge due to restarts
            }

            for timestamp in walk_timestamps.iter().cloned() {
                if timestamp == -1_i64 || head_timestamp == -1_i64 {
                    continue;
                }

                assert!(timestamp >= head_timestamp + 0 && timestamp < head_timestamp + 2);
            }
        }
    }
}