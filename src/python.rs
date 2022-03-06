#![allow(clippy::type_complexity)]

use pyo3::prelude::*;


mod data {
    use pyo3::prelude::*;
    use std::convert::TryFrom;
    use tch::Tensor;
    use crate::data::{CooGraphStorage, CscGraphStorage, CsrGraphStorage};

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
    ) -> PyResult<(Tensor, Tensor, Option<Tensor>)> {
        let size = size.to_tuple();
        let coo_graph = CooGraphStorage::new(row_col, size);
        let CscGraphStorage {
            ptrs, indices, perm, ..
        } = CscGraphStorage::try_from(&coo_graph)?;

        Ok((ptrs, indices, perm))
    }

    #[pyfunction]
    pub fn to_csr(
        row_col: Tensor,
        size: GraphSize,
    ) -> PyResult<(Tensor, Tensor, Option<Tensor>)> {
        let size = size.to_tuple();
        let coo_graph = CooGraphStorage::new(row_col, size);
        let CsrGraphStorage {
            ptrs, indices, perm, ..
        } = CsrGraphStorage::try_from(&coo_graph)?;

        Ok((ptrs, indices, perm))
    }


    pub fn module(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(to_csc, m)?)?;
        m.add_function(wrap_pyfunction!(to_csr, m)?)?;
        Ok(())
    }
}

mod algo {
    use std::collections::HashMap;
    use std::convert::TryInto;
    use num_traits::Float;
    use pyo3::prelude::*;
    use rand::distributions::uniform::SampleUniform;
    use tch::kind::Element;
    use tch::Tensor;
    use crate::algo::neighbor_sampling as ns;
    use crate::algo::neighbor_sampling::LayerOffset;
    use crate::data::{CscGraph, CsrGraph, EdgeAttr, CooGraphBuilder, Size};
    use crate::utils::{hashmap_from, EdgeType, NodeIdx, NodeType, RelType, TensorConversionError, TensorResult, try_tensor_to_slice, random};

    #[derive(FromPyObject)]
    pub enum MixedData {
        Homogenous(Tensor),
        Heterogenous(HashMap<String, Tensor>),
    }

    impl MixedData {
        pub fn build_homogenous<T: Element>(&self) -> TensorResult<&[T]> {
            if let MixedData::Homogenous(t) = self {
                try_tensor_to_slice::<T>(t)
            } else {
                Err(TensorConversionError::Unknown("data must be homogenous".to_string()))
            }
        }

        pub fn build_heterogenous<T: Element>(&self) -> TensorResult<HashMap<String, &[T]>> {
            if let MixedData::Heterogenous(t) = self {
                let mut map = HashMap::new();
                for (k, v) in t.iter() {
                    map.insert(k.to_string(), try_tensor_to_slice::<T>(v)?);
                }
                Ok(map)
            } else {
                Err(TensorConversionError::Unknown("data must be heterogenous".to_string()))
            }
        }
    }

    #[derive(FromPyObject)]
    pub struct UniformSampler {
        with_replacement: bool,
    }

    #[derive(FromPyObject)]
    pub struct WeightedSampler {
        weights: MixedData,
    }

    impl WeightedSampler {
        pub fn build_homogenous<T: Float + SampleUniform + Element>(&self) -> TensorResult<ns::WeightedSampler<T>> {
            let weights = self.weights.build_homogenous::<T>()?;
            Ok(ns::WeightedSampler::new(EdgeAttr::new(weights)))
        }

        pub fn build_heterogenous<T: Float + SampleUniform + Element>(&self) -> TensorResult<HashMap<RelType, ns::WeightedSampler<T>>> {
            let weights = self.weights.build_heterogenous::<T>()?;
            Ok(weights.into_iter()
                .map(|(k, v)| (k, ns::WeightedSampler::new(EdgeAttr::new(v))))
                .collect())
        }
    }

    #[derive(FromPyObject)]
    pub enum SamplerType {
        Uniform(UniformSampler),
        Weighted(WeightedSampler),
    }

    #[derive(FromPyObject)]
    pub struct TemporalFilter {
        window: (i64, i64),
        timestamps: MixedData,
        forward: bool,
        mode: usize,
    }

    impl TemporalFilter {
        pub fn build_homogenous<
            const FORWARD: bool, const MODE: usize
        >(&self) -> TensorResult<ns::TemporalFilter<i64, FORWARD, MODE>> {
            let timestamps_data = self.timestamps.build_homogenous::<i64>()?;
            let window = self.window.0..=self.window.1;
            Ok(ns::TemporalFilter::new(window, EdgeAttr::new(timestamps_data)))
        }

        pub fn build_heterogenous<
            const FORWARD: bool, const MODE: usize
        >(&self) -> TensorResult<HashMap<RelType, ns::TemporalFilter<i64, FORWARD, MODE>>> {
            let timestamps_data = self.timestamps.build_heterogenous::<i64>()?;
            let window = self.window.0..=self.window.1;
            Ok(timestamps_data.into_iter()
                .map(|(k, v)| (k, ns::TemporalFilter::new(window.clone(), EdgeAttr::new(v))))
                .collect())
        }
    }

    #[derive(FromPyObject)]
    pub enum FilterType {
        TemporalFilter((TemporalFilter, MixedData)),
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
        Vec<LayerOffset>
    )> {
        let mut rng = random::rng_get();

        let ptrs = try_tensor_to_slice::<i64>(&col_ptrs)?;
        let indices = try_tensor_to_slice::<i64>(&row_indices)?;
        let graph = CscGraph::new(ptrs, indices);

        let inputs_data = try_tensor_to_slice::<i64>(&inputs)?;

        let (samples, edge_index, layer_offsets) = match_mixed_return! {
            match (sampler.as_ref()) {
                Some(SamplerType::Uniform(UniformSampler { with_replacement: true })) => ns::UnweightedSampler::<true>,
                Some(SamplerType::Uniform(UniformSampler { with_replacement: false })) => ns::UnweightedSampler::<false>,
                Some(SamplerType::Weighted(s@WeightedSampler { .. })) => s.build_homogenous::<f64>()?,
                _ => ns::UnweightedSampler::<false>,
            } ==> |sampler| {
                match_mixed_return! {
                    match (filter.as_ref()) {
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            mode: ns::TEMPORAL_SAMPLE_STATIC, ..
                        }, inputs_state))) => (
                            ft.build_homogenous::<true, {ns::TEMPORAL_SAMPLE_STATIC}>()?,
                            inputs_state.build_homogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: true, mode: ns::TEMPORAL_SAMPLE_RELATIVE, ..
                        }, inputs_state))) => (
                            ft.build_homogenous::<true, {ns::TEMPORAL_SAMPLE_RELATIVE}>()?,
                            inputs_state.build_homogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: false, mode: ns::TEMPORAL_SAMPLE_RELATIVE, ..
                        }, inputs_state))) => (
                            ft.build_homogenous::<false, {ns::TEMPORAL_SAMPLE_RELATIVE}>()?,
                            inputs_state.build_homogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: true, mode: ns::TEMPORAL_SAMPLE_DYNAMIC, ..
                        }, inputs_state))) => (
                            ft.build_homogenous::<true, {ns::TEMPORAL_SAMPLE_DYNAMIC}>()?,
                            inputs_state.build_homogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: false, mode: ns::TEMPORAL_SAMPLE_DYNAMIC, ..
                        }, inputs_state))) => (
                            ft.build_homogenous::<false, {ns::TEMPORAL_SAMPLE_DYNAMIC}>()?,
                            inputs_state.build_homogenous::<i64>()?,
                        ),
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

    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    pub fn neighbor_sampling_heterogenous(
        node_types: Vec<NodeType>,
        edge_types: Vec<EdgeType>,
        col_ptrs: HashMap<RelType, Tensor>,
        row_indices: HashMap<RelType, Tensor>,
        inputs: HashMap<NodeType, Tensor>,
        num_neighbors: HashMap<RelType, Vec<usize>>,
        num_hops: usize,
        sampler: Option<SamplerType>,
        filter: Option<FilterType>,
    ) -> PyResult<(
        HashMap<NodeType, Tensor>,
        HashMap<RelType, Tensor>,
        HashMap<RelType, Tensor>,
        HashMap<RelType, Tensor>,
        HashMap<RelType, Vec<LayerOffset>>,
    )> {
        let mut rng = random::rng_get();

        let rel_types = col_ptrs.keys().cloned().collect::<Vec<_>>();
        let mut graphs = HashMap::new();
        for rel_type in rel_types.iter().cloned() {
            let ptrs = try_tensor_to_slice::<i64>(&col_ptrs[&rel_type])?;
            let indices = try_tensor_to_slice::<i64>(&row_indices[&rel_type])?;
            graphs.insert(rel_type, CscGraph::new(ptrs, indices));
        }

        let inputs_data: HashMap<NodeType, &[i64]> = inputs.iter().map(|(node_type, tensor)| {
            let data = try_tensor_to_slice::<i64>(tensor)?;
            Ok((node_type.clone(), data))
        }).collect::<PyResult<_>>()?;

        let mut tmp = HashMap::new();
        let (samples, coo_builders, layer_offsets) = match_mixed_return! {
            match (sampler.as_ref()) {
                Some(SamplerType::Uniform(UniformSampler { with_replacement: true })) => {
                    hashmap_from(rel_types.iter(), |_k| ns::UnweightedSampler::<true>)
                },
                Some(SamplerType::Uniform(UniformSampler { with_replacement: false })) => {
                    hashmap_from(rel_types.iter(), |_k| ns::UnweightedSampler::<false>)
                },
                Some(SamplerType::Weighted(s@WeightedSampler { .. })) => {
                    s.build_heterogenous::<f64>()?
                },
                _ => {
                    hashmap_from(rel_types.iter(), |_k| ns::UnweightedSampler::<false>)
                },
            } ==> |sampler| {
                match_mixed_return! {
                    match (filter.as_ref()) {
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            mode: ns::TEMPORAL_SAMPLE_STATIC, ..
                        }, inputs_state))) => (
                            ft.build_heterogenous::<true, {ns::TEMPORAL_SAMPLE_STATIC}>()?,
                            inputs_state.build_heterogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: true, mode: ns::TEMPORAL_SAMPLE_RELATIVE, ..
                        }, inputs_state))) => (
                            ft.build_heterogenous::<true, {ns::TEMPORAL_SAMPLE_RELATIVE}>()?,
                            inputs_state.build_heterogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: false, mode: ns::TEMPORAL_SAMPLE_RELATIVE, ..
                        }, inputs_state))) => (
                            ft.build_heterogenous::<false, {ns::TEMPORAL_SAMPLE_RELATIVE}>()?,
                            inputs_state.build_heterogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: true, mode: ns::TEMPORAL_SAMPLE_DYNAMIC, ..
                        }, inputs_state))) => (
                            ft.build_heterogenous::<true, {ns::TEMPORAL_SAMPLE_DYNAMIC}>()?,
                            inputs_state.build_heterogenous::<i64>()?,
                        ),
                        Some(FilterType::TemporalFilter((ft @ TemporalFilter {
                            forward: false, mode: ns::TEMPORAL_SAMPLE_DYNAMIC, ..
                        }, inputs_state))) => (
                            ft.build_heterogenous::<false, {ns::TEMPORAL_SAMPLE_DYNAMIC}>()?,
                            inputs_state.build_heterogenous::<i64>()?,
                        ),
                        _ => {
                            tmp = hashmap_from(inputs_data.keys(), |k| vec![(); inputs_data[*k].len()]);
                            (
                                hashmap_from(rel_types.iter(), |_k| ns::IdentityFilter),
                                hashmap_from(inputs_data.keys(), |k| tmp[*k].as_slice()),
                            )
                        },
                    } ==> |(filter, inputs_state)| {
                        Ok(crate::algo::neighbor_sampling::neighbor_sampling_heterogenous(
                            &mut rng, &node_types, &edge_types, &graphs, &inputs_data, &num_neighbors, num_hops, &sampler, &filter, &inputs_state,
                        )) as TensorResult<(
                            HashMap<NodeType, Vec<NodeIdx>>,
                            HashMap<RelType, CooGraphBuilder>,
                            HashMap<RelType, Vec<LayerOffset>>,
                        )>
                    }
                }
            }
        }?;

        let samples: HashMap<NodeType, Tensor> = samples.into_iter().map(|(ty, samples)| {
            (ty, samples.try_into().expect("Can't convert vec into tensor"))
        }).collect();
        let mut rows = HashMap::new();
        let mut cols = HashMap::new();
        let mut edge_indexes = HashMap::new();
        for (rel_type, coo_builder) in coo_builders.into_iter() {
            let (row, col, edge_index) = coo_builder.to_tensor();
            rows.insert(rel_type.clone(), row);
            cols.insert(rel_type.clone(), col);
            edge_indexes.insert(rel_type.clone(), edge_index);
        }

        Ok((
            samples,
            rows,
            cols,
            edge_indexes,
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
        let mut rng = random::rng_get();

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

    #[pyfunction]
    pub fn negative_sample_neighbors_homogenous(
        row_ptrs: Tensor,
        col_indices: Tensor,
        graph_size: Size,
        inputs: Tensor,
        num_neg: i64,
        try_count: i64
    ) -> PyResult<(Tensor, Tensor, Tensor, usize)> {
        let mut rng = random::rng_get();

        let ptrs = try_tensor_to_slice::<i64>(&row_ptrs)?;
        let indices = try_tensor_to_slice::<i64>(&col_indices)?;
        let graph = CsrGraph::new(ptrs, indices);

        let inputs = try_tensor_to_slice::<i64>(&inputs)?;

        let (samples, edge_index, sample_count) = crate::algo::negative_sampling::negative_sample_neighbors_homogenous(
            &mut rng,
            &graph,
            graph_size,
            &inputs,
            num_neg,
            try_count,
        );

        let samples = samples.try_into().expect("Can't convert vec into tensor");
        let rows = edge_index.rows.try_into().expect("Can't convert vec into tensor");
        let cols = edge_index.cols.try_into().expect("Can't convert vec into tensor");

        Ok((samples, rows, cols, sample_count))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    pub fn negative_sample_neighbors_heterogenous(
        node_types: Vec<NodeType>,
        edge_types: Vec<EdgeType>,
        row_ptrs: HashMap<RelType, Tensor>,
        col_indices: HashMap<RelType, Tensor>,
        sizes: HashMap<RelType, Size>,
        inputs: HashMap<NodeType, Tensor>,
        num_neg: i64,
        try_count: i64,
    ) -> PyResult<(
        HashMap<NodeType, Tensor>,
        HashMap<RelType, Tensor>,
        HashMap<RelType, Tensor>,
        HashMap<NodeType, usize>,
    )> {
        let mut rng = random::rng_get();

        let mut graphs = HashMap::new();
        for rel_type in row_ptrs.keys().cloned() {
            let ptrs = try_tensor_to_slice::<i64>(&row_ptrs[&rel_type])?;
            let indices = try_tensor_to_slice::<i64>(&col_indices[&rel_type])?;
            let size = sizes[&rel_type];
            graphs.insert(rel_type, (CsrGraph::new(ptrs, indices), size));
        }

        let inputs_data: HashMap<NodeType, &[i64]> = inputs.iter().map(|(node_type, tensor)| {
            let data = try_tensor_to_slice::<i64>(tensor)?;
            Ok((node_type.clone(), data))
        }).collect::<PyResult<_>>()?;

        let (samples, edge_index, sample_count) = crate::algo::negative_sampling::negative_sample_neighbors_heterogenous(
            &mut rng,
            &node_types,
            &edge_types,
            &graphs,
            &inputs_data,
            num_neg,
            try_count,
        );

        let samples: HashMap<NodeType, Tensor> = samples.into_iter().map(|(ty, samples)| {
            (ty, samples.try_into().expect("Can't convert vec into tensor"))
        }).collect();
        let mut rows = HashMap::new();
        let mut cols = HashMap::new();
        for (rel_type, coo_builder) in edge_index.into_iter() {
            let (row, col, _edge_index) = coo_builder.to_tensor();
            rows.insert(rel_type.clone(), row);
            cols.insert(rel_type.clone(), col);
        }

        Ok((
            samples,
            rows,
            cols,
            sample_count,
        ))
    }

    pub fn module(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(neighbor_sampling_homogenous, m)?)?;
        m.add_function(wrap_pyfunction!(neighbor_sampling_heterogenous, m)?)?;
        m.add_function(wrap_pyfunction!(random_walk, m)?)?;
        m.add_function(wrap_pyfunction!(negative_sample_neighbors_homogenous, m)?)?;
        m.add_function(wrap_pyfunction!(negative_sample_neighbors_heterogenous, m)?)?;
        Ok(())
    }
}

#[pymodule]
fn tch_geometric(py: Python, m: &PyModule) -> PyResult<()> {
    data::module(py, m)?;
    algo::module(py, m)?;
    Ok(())
}