use rand::{Rng, SeedableRng};
use tch::{Kind, Scalar, Tensor};
use crate::data::graph::CsrGraph;
use crate::utils::tensor::{TensorResult, try_tensor_to_slice, try_tensor_to_slice_mut};

pub fn node2vec(
    graph: &CsrGraph,
    start: &Tensor,
    walk_length: i64,
    p: f32,
    q: f32,
) -> TensorResult<Tensor> {
    let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

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
        let mut prev = *n;
        let mut cur = *n;
        walks_data[i * L] = cur;

        for l in 0..walk_length as usize {
            let neighbors = graph.neighbors_slice(cur);

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
                } else if next == prev + 2 {
                    // visiting the node with the distance of 2 between the previous node
                    if r < prob2 { break; }
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
    use std::collections::HashMap;
    use std::convert::{TryFrom, TryInto};
    use std::path::PathBuf;
    use tch::Tensor;
    use crate::algo::random_walk::node2vec;
    use crate::data::convert::CsrGraphData;
    use crate::data::graph::CsrGraph;
    use crate::utils::tensor::try_tensor_to_slice;

    #[test]
    fn test_node2vec() {
        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/karate.npz");
        let data: HashMap<_, _> = Tensor::read_npz(&d).unwrap().into_iter().collect();
        let x = &data["x"];
        let edge_index = &data["edge_index"];

        let graph_data = CsrGraphData::try_from_edge_index(edge_index, x.size()[0]).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let start = Tensor::of_slice(&[0 as i64, 1, 2, 3]);

        let walks = node2vec(
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
                assert!(graph.has_edge(*curr, *prev));
            }
        }
    }
}