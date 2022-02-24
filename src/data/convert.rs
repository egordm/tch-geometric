use std::convert::{TryFrom};
use std::ops::Add;
use rayon::prelude::*;
use tch::{Device, IndexOp, Tensor};
use tch::kind::Element;
use crate::data::graph::{Csc, Csr, SparseGraph, SparseGraphType, SparseGraphTypeTrait};
use crate::utils::tensor::{check_device, TensorResult, TensorConversionError, try_tensor_to_slice_mut, try_tensor_to_slice, tensor_to_slice_mut};
use crate::utils::types::IndexType;

pub fn ind2ptr(
    ind: &Tensor,
    m: i64,
) -> TensorResult<Tensor> {
    check_device!(ind, Device::Cpu);

    let mut out = Tensor::empty(&[m + 1], (ind.kind(), ind.device()));
    let ind_data = try_tensor_to_slice::<i64>(ind)?;
    let out_data = try_tensor_to_slice_mut::<i64>(&mut out)?;

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

pub struct SparseGraphData<Ty> {
    pub ptrs: Tensor,
    pub indices: Tensor,
    pub perm: Tensor,
    _phantom: std::marker::PhantomData<Ty>,
}

pub type CscGraphData = SparseGraphData<Csc>;
pub type CsrGraphData = SparseGraphData<Csr>;

impl<Ty> SparseGraphData<Ty> {
    pub fn new(
        ptrs: Tensor,
        indices: Tensor,
        perm: Tensor,
    ) -> SparseGraphData<Ty> {
        SparseGraphData {
            ptrs,
            indices,
            perm,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Ty: SparseGraphTypeTrait> SparseGraphData<Ty> {
    pub fn try_from_edge_index(
        edge_index: &Tensor,
        node_count: i64,
    ) -> TensorResult<Self> {
        let row = edge_index.select(0, 0);
        let col = edge_index.select(0, 1);

        match Ty::get_type() {
            SparseGraphType::Csr => {
                let perm = (&row * node_count).add(&col).argsort(0, false);
                let row_ptrs = ind2ptr(&row.i(&perm), node_count)?;
                let col_indices = col.i(&perm);

                Ok(Self::new(row_ptrs, col_indices, perm))
            }
            SparseGraphType::Csc => {
                let perm = (&col * node_count).add(&row).argsort(0, false);
                let col_ptrs = ind2ptr(&col.i(&perm), node_count)?;
                let row_indices = row.i(&perm);

                Ok(Self::new(col_ptrs, row_indices, perm))
            }
        }
    }
}

impl<
    'a, Ty, Ptr: Element + IndexType, Ix: Element + IndexType
> TryFrom<&'a SparseGraphData<Ty>> for SparseGraph<'a, Ty, Ptr, Ix> {
    type Error = TensorConversionError;

    fn try_from(value: &'a SparseGraphData<Ty>) -> Result<Self, Self::Error> {
        let ptrs = try_tensor_to_slice(&value.ptrs)?;
        let indices = try_tensor_to_slice(&value.indices)?;

        Ok(SparseGraph::new(ptrs, indices))
    }
}

pub fn csc_sort_edges(
    col_ptrs: &Tensor,
    perm: &Tensor,
    row_weights: &Tensor,
    descending: bool,
) -> TensorResult<Tensor> {
    check_device!(col_ptrs, Device::Cpu);

    let new_perm = perm.copy();
    let col_ptrs_data = try_tensor_to_slice::<i64>(col_ptrs)?;

    // TODO: benchmark this implementation. Check if it's runtime is very different than serial or native rust one
    col_ptrs_data.par_iter()
        .zip(col_ptrs_data.par_iter().skip(1))
        .for_each(|(col_start, col_end)| {
            if col_end - col_start <= 1 {
                return;
            }

            let sorted_idx = row_weights
                .slice(0, *col_start, *col_end, 1)
                .argsort(0, descending) + *col_start;
            let slice_perm = perm.i(&sorted_idx);
            new_perm.slice(0, *col_start, *col_end, 1).copy_(&slice_perm)
        });

    Ok(new_perm)
}

pub fn csc_edge_cumsum<T: Element + Add<Output=T> + Default + Copy>(
    col_ptrs: &Tensor,
    row_data: &mut Tensor,
) -> TensorResult<()> {
    check_device!(col_ptrs, Device::Cpu);

    let col_ptrs_data = try_tensor_to_slice::<i64>(col_ptrs)?;
    col_ptrs_data.par_iter()
        .zip(col_ptrs_data.par_iter().skip(1))
        .for_each(|(col_start, col_end)| {
            if col_end - col_start <= 1 {
                return;
            }

            let mut row_slice = row_data.slice(0, *col_start, *col_end, 1);
            let row_data = tensor_to_slice_mut::<T>(&mut row_slice);
            let mut acc = T::default();
            for row_data_val in row_data.iter_mut() {
                acc = acc + *row_data_val;
                *row_data_val = acc;
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::convert::{TryFrom, TryInto};
    use ndarray::{arr2, Array2};
    use tch::Tensor;
    use crate::data::convert::{csc_edge_cumsum, csc_sort_edges, CscGraphData, ind2ptr};
    use crate::data::graph::CscGraph;

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


        let result = CscGraphData::try_from_edge_index(&edge_index, m).unwrap();
        let graph: CscGraph<i64, i64> = (&result).try_into().unwrap();

        assert_eq!(graph.in_degree(0), 3);
        assert_eq!(graph.in_degree(1), 2);
        assert_eq!(graph.in_degree(4), 1);
        assert_eq!(graph.in_degree(2), 2);
        assert_eq!(graph.neighbors_slice(0), [1, 2, 3]);
        assert_eq!(graph.neighbors_slice(1), [4, 5]);
    }

    #[test]
    fn test_csc_sort_edges() {
        let col_ptrs_data: Vec<i64> = vec![0, 0, 0, 0, 3, 5, 5, 5, 7, 9];
        let perm_data: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let row_weights_data: Vec<f64> = vec![9.0, 5.0, 8.0, 9.0, 10.0, 11.0, 1.0, 1.5];

        let col_ptrs = Tensor::of_slice(&col_ptrs_data);
        let perm = Tensor::of_slice(&perm_data);
        let row_weights = Tensor::of_slice(&row_weights_data);

        let result = csc_sort_edges(&col_ptrs, &perm, &row_weights, false).unwrap();
        let result_data: Vec<i64> = result.into();

        assert_eq!(result_data, vec![1, 2, 0, 3, 4, 6, 5, 7]);
    }

    #[test]
    fn test_csc_edge_cumsum() {
        let col_ptrs_data: Vec<i64> = vec![0, 0, 0, 0, 3, 5, 5, 5, 7, 9];
        let row_data_data: Vec<f64> = vec![9.0, 5.0, 8.0, 9.0, 10.0, 11.0, 1.0, 1.5];

        let col_ptrs = Tensor::of_slice(&col_ptrs_data);
        let mut row_data = Tensor::of_slice(&row_data_data);

        csc_edge_cumsum::<f64>(&col_ptrs, &mut row_data).unwrap();

        let result_data: Vec<f64> = row_data.into();

        assert_eq!(result_data, vec![9.0, 14.0, 22.0, 9.0, 19.0, 11.0, 12.0, 1.5]);
    }
}