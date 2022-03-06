use std::ops::Add;
use rayon::prelude::*;
use tch::{Device, IndexOp, Tensor};
use tch::kind::Element;
use crate::utils::tensor::{check_device, TensorResult, TensorConversionError, try_tensor_to_slice, tensor_to_slice_mut};

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
    use crate::data::graph::CscGraph;
    use crate::data::transform::{csc_edge_cumsum, csc_sort_edges};


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