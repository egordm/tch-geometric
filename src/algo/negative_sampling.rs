use rand::Rng;
use tch::{Device, Kind, Tensor};
use rayon::prelude::*;
use crate::data::{CscGraph, CsrGraph};
use crate::utils::{NodeIdx, TensorResult, try_tensor_to_slice, try_tensor_to_slice_mut};
use crate::utils::tensor::{check_device, check_kind, TensorConversionError};


pub fn approx_negative_sample_neighbors(
    rng: &mut impl Rng,
    graph: &CsrGraph,
    graph_size: (i64, i64),
    inputs: Tensor,
    num_neg: i64,
    try_count: i64,
) -> TensorResult<(Tensor, Vec<usize>)> {
    let node_count = graph_size.1;
    let v_data = inputs.repeat_interleave_self_int(num_neg, None, None);
    let mut w_data = v_data.randint_like_low_dtype(0, node_count);

    let v = try_tensor_to_slice::<i64>(&v_data)?;
    let w = try_tensor_to_slice_mut::<i64>(&mut w_data)?;

    let mut mask = Vec::new();
    let mut mask_tmp = Vec::new();
    mask.par_extend(
        (0..w.len())
            .into_par_iter()
            .filter(|i| graph.has_edge(v[*i], w[*i]))
    );

    for _t in 0..try_count {
        for i in mask.iter() {
            w[*i] = rng.gen_range(0..node_count);
        }

        std::mem::swap(&mut mask, &mut mask_tmp);
        mask.clear();

        mask.par_extend(
            mask_tmp
                .par_iter().cloned()
                .filter(|i| graph.has_edge(v[*i], w[*i]))
        );
    }

    let output = Tensor::vstack(&[v_data, w_data]);
    Ok((output, mask))
}


#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use rand::SeedableRng;
    use tch::Tensor;
    use crate::data::{CsrGraph, CsrGraphData};
    use crate::data::tests::load_karate_graph;
    use crate::utils::try_tensor_to_slice;

    #[test]
    pub fn test_negative_sample() {
        let (x, _, coo_graph) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let node_count = x.size()[0];
        let graph_data = CsrGraphData::try_from(&coo_graph).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let inputs = Tensor::of_slice(&[0_i64, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let (output, mask) = super::approx_negative_sample_neighbors(
            &mut rng,
            &graph,
            (node_count, node_count),
            inputs,
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