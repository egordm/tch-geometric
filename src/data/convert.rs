use std::convert::{TryFrom};
use std::ops::Add;
use tch::{Device, IndexOp, Tensor};
use tch::kind::Element;
use crate::data::graph::{Csc, Csr, SparseGraph, SparseGraphType, SparseGraphTypeTrait};
use crate::utils::tensor::{check_device, TensorResult, TensorConversionError, try_tensor_to_slice_mut, try_tensor_to_slice};
use crate::utils::types::IndexType;

pub struct CooGraphData {
    pub row_col: Tensor,
    pub size: (i64, i64),
}

impl CooGraphData {
    pub fn new(row_col: Tensor, size: (i64, i64)) -> Self {
        Self {
            row_col,
            size,
        }
    }

    pub fn row(&self) -> Tensor {
        self.row_col.select(0, 0)
    }

    pub fn col(&self) -> Tensor {
        self.row_col.select(0, 1)
    }
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
    ) -> Self {
        Self {
            ptrs,
            indices,
            perm,
            _phantom: std::marker::PhantomData,
        }
    }
}

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

impl<Ty: SparseGraphTypeTrait> TryFrom<&CooGraphData> for SparseGraphData<Ty> {
    type Error = TensorConversionError;

    fn try_from(value: &CooGraphData) -> Result<Self, Self::Error> {
        let (row, col) = (value.row(), value.col());
        let size = value.size;

        match Ty::get_type() {
            SparseGraphType::Csr => {
                let perm = (&row * size.1).add(&col).argsort(0, false);
                let row_ptrs = ind2ptr(&row.i(&perm), size.0)?;
                let col_indices = col.i(&perm);

                Ok(Self::new(row_ptrs, col_indices, perm))
            }
            SparseGraphType::Csc => {
                let perm = (&col * size.0).add(&row).argsort(0, false);
                let col_ptrs = ind2ptr(&col.i(&perm), size.1)?;
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


#[cfg(test)]
mod tests {
    use std::convert::{TryFrom, TryInto};
    use ndarray::{arr2, Array2};
    use tch::Tensor;
    use crate::data::convert::{CscGraphData, ind2ptr};
    use crate::data::CooGraphData;
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
        let coo_graph_data = CooGraphData::new(edge_index, (m, m) );

        let result = CscGraphData::try_from(&coo_graph_data).unwrap();
        let graph: CscGraph<i64, i64> = (&result).try_into().unwrap();

        assert_eq!(graph.in_degree(0), 3);
        assert_eq!(graph.in_degree(1), 2);
        assert_eq!(graph.in_degree(4), 1);
        assert_eq!(graph.in_degree(2), 2);
        assert_eq!(graph.neighbors_slice(0), [1, 2, 3]);
        assert_eq!(graph.neighbors_slice(1), [4, 5]);
    }
}