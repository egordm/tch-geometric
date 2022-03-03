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
    use std::convert::TryFrom;
    use tch::Tensor;
    use crate::data::{CooGraphData, CscGraphData, CsrGraphData};

    #[derive(FromPyObject)]
    pub enum GraphSize {
        Square(i64),
        Other(i64, i64),
    }

    impl GraphSize {
        pub fn to_tuple(&self) -> (i64, i64) {
            match self {
                GraphSize::Square(n) => (*n, *n),
                GraphSize::Other(m, n) => (*m, *n),
            }
        }
    }

    #[pyfunction]
    pub fn to_csc(
        row_col: Tensor,
        size: GraphSize,
    ) -> PyResult<(Tensor, Tensor, Tensor)> {
        let size = size.to_tuple();
        let coo_graph = CooGraphData::new(row_col, size);
        let CscGraphData {
            ptrs, indices, perm, ..
        } = CscGraphData::try_from(&coo_graph)?;

        Ok((ptrs, indices, perm))
    }

    #[pyfunction]
    pub fn to_csr(
        row_col: Tensor,
        size: GraphSize,
    ) -> PyResult<(Tensor, Tensor, Tensor)> {
        let size = size.to_tuple();
        let coo_graph = CooGraphData::new(row_col, size);
        let CsrGraphData {
            ptrs, indices, perm, ..
        } = CsrGraphData::try_from(&coo_graph)?;

        Ok((ptrs, indices, perm))
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
    use crate::algo::neighbor_sampling as ns;
    use crate::data::{CscGraph, CsrGraph, EdgeAttr, CooGraphBuilder};
    use crate::utils::{EdgePtr, NodeIdx, NodePtr, TensorResult, try_tensor_to_slice};

    #[derive(FromPyObject)]
    pub struct UniformSampler {
        with_replacement: bool,
    }

    #[derive(FromPyObject)]
    pub struct WeightedSampler {
        weights: Tensor,
    }

    #[derive(FromPyObject)]
    pub enum SamplerType {
        Uniform(UniformSampler),
        Weighted(WeightedSampler),
    }

    #[derive(FromPyObject)]
    pub struct TemporalFilter {
        window: (i64, i64),
        timestamps: Tensor,
        initial_state: Tensor,
        forward: bool,
        mode: usize,
    }

    impl TemporalFilter {
        pub fn build<
            const FORWARD: bool, const MODE: usize
        >(&self) -> TensorResult<(ns::TemporalFilter<i64, FORWARD, MODE>, &[i64])> {
            let timestamps_data = try_tensor_to_slice::<i64>(&self.timestamps)?;
            let initial_state_data = try_tensor_to_slice::<i64>(&self.initial_state)?;
            let window = self.window.0..=self.window.1;

            Ok((
                ns::TemporalFilter::new(window, EdgeAttr::new(timestamps_data)),
                initial_state_data,
            ))
        }
    }

    #[derive(FromPyObject)]
    pub enum FilterType {
        TemporalFilter(TemporalFilter),
    }

    macro_rules! match_mixed_return {
        {
            match ($m:expr) {
                $($pattern:pat => $result:expr),*,
            } ==> $ret:expr
        } => {
            match $m {
                $(
                    $pattern => {
                        let result = $result;
                        $ret(result)
                    },
                )*
            }
        }
    }

    #[pyfunction]
    pub fn neighbor_sampling_homogenous(
        col_ptrs: Tensor,
        row_indices: Tensor,
        inputs: Tensor,
        num_neighbors: Vec<usize>,
        sampler: Option<SamplerType>,
        filter: Option<FilterType>,
    ) -> PyResult<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Vec<(NodePtr, EdgePtr, NodePtr)>
    )> {
        let mut rng = super::random::get();

        let ptrs = try_tensor_to_slice::<i64>(&col_ptrs)?;
        let indices = try_tensor_to_slice::<i64>(&row_indices)?;
        let graph = CscGraph::new(ptrs, indices);

        let inputs_data = try_tensor_to_slice::<i64>(&inputs)?;

        let (samples, edge_index, layer_offsets) = match_mixed_return! {
            match (sampler.as_ref()) {
                Some(SamplerType::Uniform(UniformSampler { with_replacement: true })) => ns::UnweightedSampler::<true>,
                Some(SamplerType::Uniform(UniformSampler { with_replacement: false })) => ns::UnweightedSampler::<false>,
                Some(SamplerType::Weighted(WeightedSampler { weights })) => {
                    let weights_data = try_tensor_to_slice::<f64>(weights)?;
                    ns::WeightedSampler::new(EdgeAttr::new(weights_data))
                },
                _ => ns::UnweightedSampler::<false>,
            } ==> |sampler| {
                match_mixed_return! {
                    match (filter.as_ref()) {
                        Some(FilterType::TemporalFilter(ft @ TemporalFilter {
                            mode: ns::TEMPORAL_SAMPLE_STATIC, ..
                        })) => ft.build::<true, {ns::TEMPORAL_SAMPLE_STATIC}>()?,
                        Some(FilterType::TemporalFilter(ft @ TemporalFilter {
                            forward: true, mode: ns::TEMPORAL_SAMPLE_RELATIVE, ..
                        })) => ft.build::<true, {ns::TEMPORAL_SAMPLE_RELATIVE}>()?,
                        Some(FilterType::TemporalFilter(ft @ TemporalFilter {
                            forward: false, mode: ns::TEMPORAL_SAMPLE_RELATIVE, ..
                        })) => ft.build::<false, {ns::TEMPORAL_SAMPLE_RELATIVE}>()?,
                        Some(FilterType::TemporalFilter(ft @ TemporalFilter {
                            forward: true, mode: ns::TEMPORAL_SAMPLE_DYNAMIC, ..
                        })) => ft.build::<true, {ns::TEMPORAL_SAMPLE_DYNAMIC}>()?,
                        Some(FilterType::TemporalFilter(ft @ TemporalFilter {
                            forward: false, mode: ns::TEMPORAL_SAMPLE_DYNAMIC, ..
                        })) => ft.build::<false, {ns::TEMPORAL_SAMPLE_DYNAMIC}>()?,
                        _ => (ns::IdentityFilter, &vec![(); inputs_data.len()][..]),
                    } ==> |(filter, inputs_state)| {
                        Ok(crate::algo::neighbor_sampling::neighbor_sampling_homogenous(
                            &mut rng, &graph, inputs_data, &num_neighbors, &sampler, &filter, inputs_state,
                        )) as TensorResult<(Vec<NodeIdx>, CooGraphBuilder, Vec<ns::LayerOffset>)>
                    }
                }
            }
        }?;

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