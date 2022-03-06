use rand::{Rng};
use rand::rngs::SmallRng;
use tch::{Kind, Scalar, Tensor};
use crate::data::graph::CsrGraph;
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

#[cfg(test)]
mod tests {
    use std::convert::{TryFrom};
    use rand::SeedableRng;
    use tch::Tensor;
    use crate::algo::random_walk::random_walk;
    use crate::data::{CsrGraphStorage, CsrGraph};
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
}