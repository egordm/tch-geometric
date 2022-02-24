use std::ops::Add;
use tch::{Device, IndexOp, Tensor};
use crate::utils::tensor::{check_device, TensorResult, TensorConversionError, try_tensor_to_slice_mut, try_tensor_to_slice};

pub fn ind2ptr(
    ind: &Tensor,
    m: i64,
) -> TensorResult<Tensor> {
    check_device!(ind, Device::Cpu);

    let mut out = Tensor::empty(&[m + 1], (ind.kind(), ind.device()));
    let ind_data = try_tensor_to_slice::<i64>(&ind)?;
    let mut out_data = try_tensor_to_slice_mut::<i64>(&mut out)?;

    let numel = ind.numel();
    if numel == 0 {
        return Ok(out.zero_());
    }

    for i in 0..=ind_data[0] {
        out_data[i as usize] = 0;
    }

    // TODO: parallelize this
    let mut idx = ind_data[0] as usize;
    for i in 0..numel - 1 {
        let next_idx = ind_data[i + 1] as usize;
        for idx in idx..next_idx {
            out_data[idx + 1] = (i + 1) as i64;
        }
        idx = next_idx;
    }

    for i in ind_data[numel - 1] + 1..m + 1 {
        out_data[i as usize] = numel as i64;
    }

    Ok(out)
}

pub struct CscData {
    pub col_ptrs: Tensor,
    pub row_data: Tensor,
    pub perm: Tensor,
}

pub fn to_csc(
    edge_index: &Tensor,
    node_count: i64,
) -> TensorResult<CscData> {
    let row = edge_index.select(0, 0);
    let col = edge_index.select(0, 1);

    let perm = (&col * node_count).add(&row).argsort(0, false);
    let col_ptrs = ind2ptr(&col.i(&perm), node_count)?;

    Ok(CscData {
        col_ptrs,
        row_data: row.i(&perm),
        perm,
    })
}

#[cfg(test)]
mod tests {
    use std::convert::{TryFrom};
    use ndarray::{arr2, Array2};
    use tch::Tensor;
    use crate::CscGraph;
    use crate::data::convert::{ind2ptr, to_csc};

    #[test]
    fn test_ind2ptr() {
        let m = 10;
        let input: Vec<i64> = vec![3, 3, 3, 4, 4, 7, 7, 8, 8];
        let output: Vec<i64> = vec![0, 0, 0, 0, 3, 5, 5, 5, 7, 9, 9];

        let ind = Tensor::of_slice(&input);
        let result = ind2ptr(&ind, m).unwrap();
        let result_data: Vec<i64> = result.into();

        assert_eq!(output, result_data);
    }

    #[test]
    fn test_to_csc() {
        let m = 10;
        let edge_index_data: Array2<i64> = arr2(&[
            [1, 2, 3, 4, 9, 5, 6, 7],
            [0, 0, 0, 1, 4, 1, 2, 2],
        ]);
        let edge_index = Tensor::try_from(edge_index_data).unwrap();

        let result = to_csc(&edge_index, m).unwrap();
        let graph: CscGraph<i64, i64> = CscGraph::try_from(&result).unwrap();

        assert_eq!(graph.in_degree(0), 3);
        assert_eq!(graph.in_degree(1), 2);
        assert_eq!(graph.in_degree(4), 1);
        assert_eq!(graph.in_degree(2), 2);
        assert_eq!(graph.neighbors_slice(0), [1, 2, 3]);
        assert_eq!(graph.neighbors_slice(1), [4, 5]);
    }
}