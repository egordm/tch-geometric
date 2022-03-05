use rand::Rng;
use rand::rngs::SmallRng;
use tch::{ Kind, Tensor};
use rayon::prelude::*;
use crate::data::{ CsrGraph};
use crate::utils::{TensorResult, try_tensor_to_slice, try_tensor_to_slice_mut};
use crate::utils::tensor::{check_device, check_kind, TensorConversionError};

pub fn negative_sample_neighbors(
    rng: &mut SmallRng,
    graph: &CsrGraph,
    graph_size: (i64, i64),
    inputs: &Tensor,
    num_neg: i64,
    try_count: i64,
) -> TensorResult<(Tensor, Vec<usize>)> {
    let node_count = graph_size.1;
    let v_data = inputs.repeat_interleave_self_int(num_neg, None, None);
    let mut w_data = v_data.zeros_like();

    let v = try_tensor_to_slice::<i64>(&v_data)?;
    let w = try_tensor_to_slice_mut::<i64>(&mut w_data)?;

    let mask: Vec<_> = w.into_par_iter().enumerate().map_init(
        || rng.clone(),
        |rng, (i, w)| {
            for _t in 0..try_count {
                *w = rng.gen_range(0..node_count);
                if !graph.has_edge(v[i], *w) {
                    return None;
                }
            }

            return Some(i);
        }
    ).filter_map(|v| v).collect();

    let output = Tensor::vstack(&[v_data, w_data]);
    Ok((output, mask))
}

// TODO: benchmark current impl vs per thread retry

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use rand::SeedableRng;
    use tch::Tensor;
    use crate::data::{CsrGraph, CsrGraphData};
    use crate::data::load_karate_graph;
    use crate::utils::try_tensor_to_slice;

    #[test]
    pub fn test_negative_sample_neighbors() {
        let (x, _, coo_graph) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let node_count = x.size()[0];
        let graph_data = CsrGraphData::try_from(&coo_graph).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let test: Vec<_> = (0..node_count).collect();
        let inputs = Tensor::of_slice(&test);
        let (output, mask) = super::negative_sample_neighbors(
            &mut rng,
            &graph,
            (node_count, node_count),
            &inputs,
            10,
            5,
        ).unwrap();
        let output_size = output.size()[1] as usize;

        let data = try_tensor_to_slice::<i64>(&output).unwrap();
        let v = &data[0..output_size];
        let w = &data[output_size..];

        let _error_count = mask.len();
        for (i, (v, w)) in v.iter().zip(w.iter()).enumerate() {
            if !mask.contains(&i) {
                assert!(!graph.has_edge(*v, *w));
            }
        }
    }
}