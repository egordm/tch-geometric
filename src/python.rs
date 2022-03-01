#![allow(clippy::type_complexity)]

use pyo3::prelude::*;


mod random {
    use std::cell::RefCell;
    use std::sync::Mutex;
    use lazy_static::lazy_static;
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    lazy_static! {
        static ref RNG: Mutex<RefCell<SmallRng>> = {
            Mutex::new(RefCell::new(SmallRng::from_entropy()))
        };
    }

    pub fn reseed(seed: [u8; 32]) {
        let rng = RNG.lock().unwrap();
        rng.replace(SmallRng::from_seed(seed));
    }

    pub fn get() -> SmallRng {
        let guard = RNG.lock().unwrap();
        let mut rng = guard.borrow_mut();
        let seed = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let seed = unsafe { std::mem::transmute::<[u64; 4], [u8; 32]>(seed) };
        SmallRng::from_seed(seed)
    }
}

mod data {
    use pyo3::prelude::*;
    use tch::Tensor;
    use crate::data::{CscGraphData, CsrGraphData};

    #[pyfunction]
    pub fn to_csc(
        edge_index: Tensor,
        m: i64,
    ) -> (Tensor, Tensor, Tensor) {
        let CscGraphData {
            ptrs, indices, perm, ..
        } = CscGraphData::try_from_edge_index(&edge_index, m).unwrap();

        (
            ptrs, indices, perm
        )
    }

    #[pyfunction]
    pub fn to_csr(
        edge_index: Tensor,
        m: i64,
    ) -> (Tensor, Tensor, Tensor) {
        let CsrGraphData {
            ptrs, indices, perm, ..
        } = CsrGraphData::try_from_edge_index(&edge_index, m).unwrap();

        (
            ptrs, indices, perm
        )
    }


    pub fn module(py: Python, p: &PyModule) -> PyResult<()> {
        let m = PyModule::new(py, "data")?;
        m.add_function(wrap_pyfunction!(to_csc, m)?)?;
        m.add_function(wrap_pyfunction!(to_csr, m)?)?;
        p.add_submodule(m)?;
        Ok(())
    }
}

mod algo {
    use std::convert::TryInto;
    use pyo3::prelude::*;
    use tch::Tensor;
    use crate::algo::neighbor_sampling::{IdentityFilter, UnweightedSampler, WeightedSampler};
    use crate::data::{CscGraph, CsrGraph, EdgeAttr};
    use crate::utils::{EdgePtr, NodePtr, try_tensor_to_slice};

    #[pyfunction]
    pub fn neighbor_sampling_homogenous(
        col_ptrs: Tensor,
        row_indices: Tensor,
        inputs: Tensor,
        num_neighbors: Vec<usize>,
        replace: bool,
    ) -> PyResult<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Vec<(NodePtr, EdgePtr)>
    )> {
        let mut rng = super::random::get();

        let ptrs = try_tensor_to_slice::<i64>(&col_ptrs)?;
        let indices = try_tensor_to_slice::<i64>(&row_indices)?;
        let graph = CscGraph::new(ptrs, indices);

        let inputs_data = try_tensor_to_slice::<i64>(&inputs)?;

        let (samples, edge_index, layer_offsets) = match replace {
            true => crate::algo::neighbor_sampling::neighbor_sampling_homogenous(
                &mut rng, &graph, &UnweightedSampler::<true>, &IdentityFilter, inputs_data, &num_neighbors
            ),
            false => crate::algo::neighbor_sampling::neighbor_sampling_homogenous(
                &mut rng, &graph, &UnweightedSampler::<false>, &IdentityFilter, inputs_data, &num_neighbors
            ),
        };

        let samples = samples.try_into().expect("Can't convert vec into tensor");
        let rows = edge_index.rows.try_into().expect("Can't convert vec into tensor");
        let cols = edge_index.cols.try_into().expect("Can't convert vec into tensor");
        let edge_index = edge_index.edge_index.try_into().expect("Can't convert vec into tensor");

        Ok((
            samples,
            rows,
            cols,
            edge_index,
            layer_offsets,
        ))
    }

    #[pyfunction]
    pub fn neighbor_sampling_homogenous_weighted(
        col_ptrs: Tensor,
        row_indices: Tensor,
        edge_weights: Tensor,
        inputs: Tensor,
        num_neighbors: Vec<usize>,
    ) -> PyResult<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Vec<(NodePtr, EdgePtr)>
    )>
    {
        let mut rng = super::random::get();

        let ptrs = try_tensor_to_slice::<i64>(&col_ptrs)?;
        let indices = try_tensor_to_slice::<i64>(&row_indices)?;
        let weights = try_tensor_to_slice::<f64>(&edge_weights)?;
        let graph = CscGraph::new(ptrs, indices);
        let weights_attr = EdgeAttr::new(&weights);


        let inputs_data = try_tensor_to_slice::<i64>(&inputs)?;

        let (samples, edge_index, layer_offsets) = crate::algo::neighbor_sampling::neighbor_sampling_homogenous(
            &mut rng, &graph, &WeightedSampler::new(weights_attr), &IdentityFilter, inputs_data, &num_neighbors,
        );

        let samples = samples.try_into().expect("Can't convert vec into tensor");
        let rows = edge_index.rows.try_into().expect("Can't convert vec into tensor");
        let cols = edge_index.cols.try_into().expect("Can't convert vec into tensor");
        let edge_index = edge_index.edge_index.try_into().expect("Can't convert vec into tensor");

        Ok((
            samples,
            rows,
            cols,
            edge_index,
            layer_offsets,
        ))
    }

    #[pyfunction]
    pub fn random_walk(
        row_ptrs: Tensor,
        col_indices: Tensor,
        start: Tensor,
        walk_length: i64,
        p: f32,
        q: f32,
    ) -> PyResult<Tensor> {
        let mut rng = super::random::get();

        let ptrs = try_tensor_to_slice::<i64>(&row_ptrs)?;
        let indices = try_tensor_to_slice::<i64>(&col_indices)?;
        let graph = CsrGraph::new(ptrs, indices);

        let walks = crate::algo::random_walk::random_walk(
            &mut rng,
            &graph,
            &start,
            walk_length,
            p,
            q,
        )?;

        Ok(walks)
    }

    pub fn module(py: Python, p: &PyModule) -> PyResult<()> {
        let m = PyModule::new(py, "algo")?;
        m.add_function(wrap_pyfunction!(neighbor_sampling_homogenous, m)?)?;
        m.add_function(wrap_pyfunction!(neighbor_sampling_homogenous_weighted, m)?)?;
        m.add_function(wrap_pyfunction!(random_walk, m)?)?;
        p.add_submodule(m)?;
        Ok(())
    }
}

#[pymodule]
fn tch_geometric(py: Python, m: &PyModule) -> PyResult<()> {
    data::module(py, m)?;
    algo::module(py, m)?;
    Ok(())
}