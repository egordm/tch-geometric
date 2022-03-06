#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::ops::{Neg, RangeInclusive, Sub};
use std::slice::Iter;
use num_traits::Float;
use rand::{Rng};
use rand::distributions::uniform::SampleUniform;
use crate::data::EdgeAttr;
use crate::data::graph::{CscGraph, CooGraphBuilder};
use crate::utils::{EdgePtr, EdgeType, NodeType, RelType, replacement_sampling, reservoir_sampling, reservoir_sampling_weighted};
use crate::utils::types::{NodeIdx, NodePtr};

pub trait SamplingFilter {
    type State: Copy;

    fn filter(&self, state: &Self::State, src: NodeIdx, dst: EdgePtr<usize>) -> bool;

    fn mutate(&self, state: &Self::State, src: NodeIdx, dst: EdgePtr<usize>) -> Self::State;
}

pub struct IdentityFilter;

impl SamplingFilter for IdentityFilter {
    type State = ();

    fn filter(&self, _state: &Self::State, _src: NodeIdx, _dst: EdgePtr<usize>) -> bool { true }

    fn mutate(&self, _state: &Self::State, _src: NodeIdx, _dst: EdgePtr<usize>) -> Self::State {}
}

pub const TEMPORAL_SAMPLE_STATIC: usize = 0;
pub const TEMPORAL_SAMPLE_RELATIVE: usize = 1;
pub const TEMPORAL_SAMPLE_DYNAMIC: usize = 2;

pub struct TemporalFilter<'a, T, const FORWARD: bool, const MODE: usize> {
    window: RangeInclusive<T>,
    timestamps: EdgeAttr<'a, T>,
}

impl<'a, T, const FORWARD: bool, const MODE: usize> TemporalFilter<'a, T, FORWARD, MODE> {
    pub fn new(window: RangeInclusive<T>, timestamps: EdgeAttr<'a, T>) -> Self {
        TemporalFilter {
            window,
            timestamps,
        }
    }
}

impl<
    'a, T: Copy + PartialOrd + Neg<Output=T> + Sub<Output=T>, const FORWARD: bool, const MODE: usize
> SamplingFilter for TemporalFilter<'a, T, FORWARD, MODE> {
    type State = T;

    fn filter(&self, state: &Self::State, _src: NodeIdx, dst: EdgePtr<usize>) -> bool {
        let t = self.timestamps.get(dst);
        match MODE {
            TEMPORAL_SAMPLE_STATIC => self.window.contains(t),
            TEMPORAL_SAMPLE_RELATIVE | TEMPORAL_SAMPLE_DYNAMIC => {
                match FORWARD {
                    true => self.window.contains(&(*t - *state)),
                    false => self.window.contains(&(*t - *state).neg()),
                }
            }
            _ => unreachable!(),
        }
    }

    fn mutate(&self, state: &Self::State, _src: NodeIdx, dst: EdgePtr<usize>) -> Self::State {
        match MODE {
            TEMPORAL_SAMPLE_STATIC => *state,
            TEMPORAL_SAMPLE_RELATIVE => *state,
            TEMPORAL_SAMPLE_DYNAMIC => *self.timestamps.get(dst),
            _ => unreachable!(),
        }
    }
}


pub trait Sampler {
    type State;

    fn init(&self, k: usize) -> Self::State;

    fn sample<'a>(
        &self,
        rng: &mut impl Rng,
        state: &'a mut Self::State,
        src: impl Iterator<Item=EdgePtr<usize>>,
    ) -> Iter<'a, EdgePtr<usize>>;
}

pub struct UnweightedSampler<const REPLACE: bool>;

impl<const REPLACE: bool> Sampler for UnweightedSampler<REPLACE> {
    type State = (
        Vec<usize>,
        Vec<usize>,
    );

    fn init(&self, k: usize) -> Self::State {
        (Vec::new(), vec![0; k])
    }

    fn sample<'a>(
        &self,
        rng: &mut impl Rng,
        state: &'a mut Self::State,
        src: impl Iterator<Item=EdgePtr<usize>>,
    ) -> Iter<'a, EdgePtr<usize>> {
        if REPLACE {
            // Collect the iterator into a vector
            state.0.clear();
            for v in src {
                state.0.push(v);
            }

            let n = if !state.0.is_empty() {
                replacement_sampling(rng, &state.0, &mut state.1)
            } else {
                0
            };
            state.1[0..n].iter()
        } else {
            let n = reservoir_sampling(rng, src, &mut state.1);
            state.1[0..n].iter()
        }
    }
}

pub struct WeightedSampler<'w, W: Float + SampleUniform> {
    pub weights: EdgeAttr<'w, W>,
}

impl<'w, W: Float + SampleUniform> WeightedSampler<'w, W> {
    pub fn new(weights: EdgeAttr<'w, W>) -> Self {
        Self { weights }
    }
}

impl<'w, W: Float + SampleUniform> Sampler for WeightedSampler<'w, W> {
    type State = Vec<usize>;

    fn init(&self, k: usize) -> Self::State {
        vec![0; k]
    }

    fn sample<'a>(
        &self,
        rng: &mut impl Rng,
        state: &'a mut Self::State,
        src: impl Iterator<Item=EdgePtr<usize>>,
    ) -> Iter<'a, EdgePtr<usize>> {
        let iter = src.map(|e| (e, *self.weights.get(e)));
        let n = reservoir_sampling_weighted(rng, iter, state);
        state[0..n].iter()
    }
}

pub type LayerOffset = (NodePtr, EdgePtr, NodePtr);

pub fn neighbor_sampling_homogenous<
    F: SamplingFilter
>(
    rng: &mut impl Rng,
    graph: &CscGraph,
    inputs: &[NodeIdx],
    num_neighbors: &[usize],
    sampler: &impl Sampler,
    filter: &F,
    inputs_state: &[F::State],
) -> (
    Vec<NodeIdx>,
    CooGraphBuilder,
    Vec<LayerOffset>
) {
    // Initialize some data structures for the sampling process
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut states: Vec<F::State> = Vec::new();

    let mut layer_offsets: Vec<LayerOffset> = Vec::new();
    let mut edge_index = CooGraphBuilder::new();

    samples.extend_from_slice(inputs);
    states.extend_from_slice(inputs_state);

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors.iter().cloned() {
        // Initialize the states
        let mut sampler_state = sampler.init(num_samples);

        // Add layer offset
        layer_offsets.push((samples.len() as NodePtr, edge_index.len() as EdgePtr, samples.len() as NodePtr));

        for i in begin..end {
            let w = samples[i];
            let w_state = states[i];

            let neighbors_range = graph.neighbors_range(w);
            if neighbors_range.is_empty() {
                continue;
            }

            let samples_filtered = neighbors_range.clone()
                .filter(|edge_ptr| filter.filter(&w_state, w, *edge_ptr));
            let samples_iter = sampler.sample(
                rng, &mut sampler_state, samples_filtered,
            );

            for edge_ptr in samples_iter {
                let v = graph.get_by_ptr(*edge_ptr);
                let j = samples.len();
                let state = filter.mutate(&w_state, w, *edge_ptr);

                samples.push(v);
                states.push(state);
                edge_index.push_edge(j as i64, i as i64, *edge_ptr as i64);
            }
        }

        begin = end;
        end = samples.len();
    }

    (
        samples,
        edge_index,
        layer_offsets,
    )
}


pub fn neighbor_sampling_heterogenous<
    F: SamplingFilter
>(
    rng: &mut impl Rng,
    node_types: &[NodeType],
    edge_types: &[EdgeType],
    graphs: &HashMap<RelType, CscGraph>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    num_neighbors: &HashMap<RelType, Vec<usize>>,
    num_hops: usize,
    sampler: &HashMap<RelType, impl Sampler>,
    filter: &HashMap<RelType, F>,
    inputs_state: &HashMap<NodeType, &[F::State]>,
) -> (
    HashMap<NodeType, Vec<NodeIdx>>,
    HashMap<RelType, CooGraphBuilder>,
    HashMap<RelType, Vec<LayerOffset>>,
) {
    // Create a mapping to convert single string relations to edge type triplets:
    let mut to_edge_types = HashMap::<RelType, EdgeType>::new();
    for e @ (src_node_type, rel_type, dst_node_type) in edge_types {
        to_edge_types.insert(format!("{}__{}__{}", src_node_type, rel_type, dst_node_type), e.clone());
    }

    // Initialize some data structures for the sampling process
    let mut samples: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
    let mut states: HashMap<NodeType, Vec<F::State>> = HashMap::new();

    for node_type in node_types {
        let samples = samples
            .entry(node_type.clone())
            .or_insert_with(Vec::new);
        if let Some(inputs) = inputs.get(node_type) {
            samples.extend_from_slice(inputs);
        }

        let states = states
            .entry(node_type.clone())
            .or_insert_with(Vec::new);
        if let Some(inputs_state) = inputs_state.get(node_type) {
            states.extend_from_slice(inputs_state);
        }
    }

    let mut layer_offsets: HashMap<RelType, Vec<LayerOffset>> = graphs.keys()
        .map(|rel_type| (rel_type.clone(), Vec::new()))
        .collect();
    let mut edge_index: HashMap<RelType, CooGraphBuilder> = graphs.keys()
        .map(|rel_type| (rel_type.clone(), CooGraphBuilder::new()))
        .collect();

    // Maintains begin/end indices for each node type
    let mut slices: HashMap<NodeType, (usize, usize)> = samples.iter()
        .map(|(node_type, samples)| (node_type.clone(), (0, samples.len())))
        .collect();

    for ell in 0..num_hops {
        // Apply sampling for each relation type
        for (rel_type, num_samples) in num_neighbors {
            let num_samples = num_samples[ell];
            let (src_node_type, _, dst_node_type) = &to_edge_types[rel_type];
            let filter = &filter[rel_type];
            let sampler = &sampler[rel_type];

            // Initialize the states
            let mut sampler_state = sampler.init(num_samples);

            // Select data
            let dst_samples = &samples[dst_node_type];
            let dst_states = &states[dst_node_type];
            // This should be safe so long we don't keep references to the samples' value (should copy them immediately)
            let src_samples = unsafe { (&samples[src_node_type] as *const _ as *mut Vec<NodeIdx>).as_mut() }.unwrap();
            let src_states = unsafe { (&states[src_node_type] as *const _ as *mut Vec<F::State>).as_mut() }.unwrap();

            let graph = &graphs[rel_type];
            let edge_index = edge_index.get_mut(rel_type).unwrap();

            // Add layer offset
            layer_offsets.get_mut(rel_type).unwrap()
                .push((src_samples.len() as NodePtr, edge_index.len() as EdgePtr, dst_samples.len() as NodePtr));

            let (begin, end) = slices[dst_node_type];
            for i in begin..end {
                let w = dst_samples[i];
                let w_state = dst_states[i];

                let neighbors_range = graph.neighbors_range(w);
                if neighbors_range.is_empty() {
                    continue;
                }

                let samples_filtered = neighbors_range.clone()
                    .filter(|edge_ptr| filter.filter(&w_state, w, *edge_ptr));
                let samples_iter = sampler.sample(
                    rng, &mut sampler_state, samples_filtered,
                );

                for edge_ptr in samples_iter {
                    let v = graph.get_by_ptr(*edge_ptr);
                    let j = src_samples.len();
                    let state = filter.mutate(&w_state, w, *edge_ptr);

                    src_samples.push(v);
                    src_states.push(state);
                    edge_index.push_edge(j as i64, i as i64, *edge_ptr as i64);
                }
            }
        }

        for (node_type, samples) in samples.iter() {
            let (_, end) = slices[node_type];
            *slices.get_mut(node_type).unwrap() = (end, samples.len());
        }
    }

    (
        samples,
        edge_index,
        layer_offsets,
    )
}


#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::convert::TryFrom;
    use rand::{Rng, SeedableRng};
    use crate::algo::neighbor_sampling::{IdentityFilter, LayerOffset, SamplingFilter, TemporalFilter, UnweightedSampler, WeightedSampler};
    use crate::data::{CscGraph, CscGraphStorage, EdgeAttr, CooGraphBuilder};
    use crate::data::{load_fake_hetero_graph, load_karate_graph};
    use crate::utils::{EdgeType, NodeIdx, NodeType, RelType};
    use super::{TEMPORAL_SAMPLE_STATIC, TEMPORAL_SAMPLE_RELATIVE};

    pub fn validate_neighbor_samples(
        graph: &CscGraph<i64, i64>,
        coo_builder: &CooGraphBuilder,
        samples_src: &[NodeIdx],
        samples_dst: &[NodeIdx],
        layer_offsets: &[LayerOffset],
        num_neighbors: &[usize],
    ) {
        // Validate whether all edges are valid
        for (j, i) in coo_builder.rows.iter().zip(coo_builder.cols.iter()) {
            let v = samples_src[*j as usize];
            let w = samples_dst[*i as usize];
            // Query for dst <- src edge because we are operating on csc graph
            assert!(graph.has_edge(w, v));
        }

        // Validate whether none of the nodes exceed the sampled number of neighbors
        let mut counts = vec![0_usize; samples_dst.len()];
        for (j, i) in coo_builder.rows.iter().zip(coo_builder.cols.iter()) {
            counts[*i as usize] += 1;
        }

        let mut begin = 0;
        for (i, (_, _, dst_end)) in layer_offsets.iter().cloned().enumerate() {
            let max_neighbors = num_neighbors[i];

            for i in begin..dst_end {
                assert!((0..=max_neighbors).contains(&counts[i as usize]));
            }
            begin = dst_end;
        }
    }

    pub fn samples_to_paths(
        coo_builder: &CooGraphBuilder,
        samples: &[NodeIdx],
        inputs: &[NodeIdx],
    ) -> Vec<(Vec<NodeIdx>, Vec<usize>)> {
        let mut paths = inputs.iter().map(|&i| (vec![i], vec![])).collect::<VecDeque<_>>();
        let mut head = vec![-1];
        let mut head_edges = vec![];
        for ((j, i), edge_idx) in coo_builder.rows.iter()
            .zip(coo_builder.cols.iter())
            .zip(0..coo_builder.edge_index.len())
        {
            let v = samples[*j as usize];
            let w = samples[*i as usize];

            while head.is_empty() || w != head[head.len() - 1] {
                if let Some((path, path_edges)) = paths.pop_front() {
                    head = path;
                    head_edges = path_edges;
                } else {
                    break;
                }
            }

            let mut path = head.clone();
            path.push(v);
            let mut path_edges = head_edges.clone();
            path_edges.push(edge_idx);

            paths.push_back((path, path_edges));
        }
        paths.into()
    }

    #[test]
    pub fn test_neighbor_sampling_homogenous() {
        let (_x, _, coo_graph) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphStorage::try_from(&coo_graph).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let inputs = vec![0_i64, 1, 4, 5];
        let inputs_state = vec![(); inputs.len()];

        let num_neighbors = vec![4, 3];

        let (samples, coo_builder, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &inputs,
            &num_neighbors,
            &UnweightedSampler::<true>,
            &IdentityFilter,
            &inputs_state,
        );

        validate_neighbor_samples(
            &graph, &coo_builder, &samples, &samples, &layer_offsets, &num_neighbors
        );
    }

    #[test]
    pub fn test_neighbor_sampling_homogenous_weighted() {
        let (_x, _, coo_graph) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphStorage::try_from(&coo_graph).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let weights_data = (0..graph.edge_count()).map(|_| rng.gen_range(0.2..5.0)).collect::<Vec<f64>>();
        let weights = EdgeAttr::new(&weights_data);

        let inputs = vec![0_i64, 1, 4, 5];
        let inputs_state = vec![(); inputs.len()];
        let num_neighbors = vec![4, 3];

        let (samples, coo_builder, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &inputs,
            &num_neighbors,
            &WeightedSampler::new(weights),
            &IdentityFilter,
            &inputs_state,
        );

        validate_neighbor_samples(
            &graph, &coo_builder, &samples, &samples, &layer_offsets, &num_neighbors
        );
    }

    #[test]
    pub fn test_neighbor_sampling_homogenous_temporal() {
        let (_x, _, coo_graph) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphStorage::try_from(&coo_graph).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let timestamps_data = (0..graph.edge_count()).map(|_| rng.gen_range(0..4)).collect::<Vec<i64>>();
        let timestamps = EdgeAttr::new(&timestamps_data);

        let inputs = vec![0_i64, 1, 4, 5];
        let input_timestamps = vec![0_i64, 1, 2, 3];
        let num_neighbors = vec![4, 3];

        // Tests static window sampling
        let filter = TemporalFilter::<i64, false, TEMPORAL_SAMPLE_STATIC>::new(
            0..=2, timestamps.clone(),
        );
        let (samples, coo_builder, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &inputs,
            &num_neighbors,
            &UnweightedSampler::<false>,
            &filter,
            &input_timestamps,
        );

        validate_neighbor_samples(
            &graph, &coo_builder, &samples, &samples, &layer_offsets, &num_neighbors
        );
        let paths = samples_to_paths(&coo_builder, &samples, &inputs);
        for (_path, edges) in paths {
            for edge_idx in edges {
                let edge_ptr = coo_builder.edge_index[edge_idx];
                let t = timestamps.get(edge_ptr as usize);
                assert!((0..=2).contains(t));
            }
        }

        // Tests relative window sampling backward in time
        let filter = TemporalFilter::<i64, false, TEMPORAL_SAMPLE_RELATIVE>::new(
            0..=2, timestamps.clone(),
        );
        let (samples, coo_builder, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &inputs,
            &num_neighbors,
            &UnweightedSampler::<false>,
            &filter,
            &input_timestamps,
        );

        validate_neighbor_samples(
            &graph, &coo_builder, &samples, &samples, &layer_offsets, &num_neighbors
        );
        let paths = samples_to_paths(&coo_builder, &samples, &inputs);
        for (_path, edges) in paths {
            if let Some(edge_idx) = edges.first().cloned() {
                let to_idx = coo_builder.cols[edge_idx];
                let start_t = input_timestamps[to_idx as usize];

                for edge_idx in edges {
                    let edge_ptr = coo_builder.edge_index[edge_idx];
                    let t = *timestamps.get(edge_ptr as usize);

                    assert!(((start_t - 2)..=start_t).contains(&t));
                }
            }
        }
    }

    #[test]
    pub fn test_neighbor_sampling_heterogenous() {
        let (xs, coo_graphs) = load_fake_hetero_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let node_types: Vec<NodeType> = xs.keys().cloned().collect();
        let edge_types: Vec<EdgeType> = coo_graphs.keys().cloned().collect();

        let mut to_edge_types = HashMap::<RelType, EdgeType>::new();
        for e @ (src_node_type, rel_type, dst_node_type) in edge_types.iter() {
            to_edge_types.insert(format!("{}__{}__{}", src_node_type, rel_type, dst_node_type), e.clone());
        }

        let graph_data: HashMap<RelType, CscGraphStorage> = coo_graphs.iter().map(|((src, rel, dst), coo_graph)| {
            let graph_data = CscGraphStorage::try_from(coo_graph).unwrap();
            (format!("{}__{}__{}", src, rel, dst), graph_data)
        }).collect();
        let graphs: HashMap<RelType, CscGraph> = graph_data.iter().map(|(rel_type, graph_data)| {
            let graph = CscGraph::<i64, i64>::try_from(graph_data).unwrap();
            (rel_type.clone(), graph)
        }).collect();

        let rel_types: Vec<RelType> = graphs.keys().cloned().collect();

        let inputs_data: HashMap<NodeType, Vec<NodeIdx>> = node_types.iter().map(|node_type| {
            (node_type.clone(), vec![0_i64, 1, 4, 5])
        }).collect();
        let inputs: HashMap<NodeType, &[NodeIdx]> = inputs_data.iter().map(|(node_type, inputs)| {
            (node_type.clone(), &inputs[..])
        }).collect();

        let inputs_state_data: HashMap<NodeType, Vec<<IdentityFilter as SamplingFilter>::State>> = inputs.iter().map(|(node_type, inputs)| {
            (node_type.clone(), vec![(); inputs.len()])
        }).collect();
        let inputs_state: HashMap<NodeType, &[<IdentityFilter as SamplingFilter>::State]> = inputs_state_data.iter().map(|(node_type, input_state)| {
            (node_type.clone(), &input_state[..])
        }).collect();

        let num_neighbors: HashMap<RelType, Vec<usize>> = rel_types.iter().cloned().map(|rel_type| {
            (rel_type, vec![4, 3])
        }).collect();
        let num_hops = 2;

        let sampler = rel_types.iter().cloned().map(|rel_type| {
            (rel_type, UnweightedSampler::<false>)
        }).collect();
        let filter = rel_types.iter().cloned().map(|rel_type| {
            (rel_type, IdentityFilter)
        }).collect();

        let (samples, coo_builders, layer_offsets) = super::neighbor_sampling_heterogenous(
            &mut rng,
            &node_types,
            &edge_types,
            &graphs,
            &inputs,
            &num_neighbors,
            num_hops,
            &sampler,
            &filter,
            &inputs_state,
        );

        for rel_type in coo_builders.keys() {
            let (src_node_type, _, dst_node_type) = &to_edge_types[rel_type];

            validate_neighbor_samples(
                &graphs[rel_type],
                &coo_builders[rel_type],
                &samples[src_node_type],
                &samples[dst_node_type],
                &layer_offsets[rel_type],
                &num_neighbors[rel_type],
            );
        }
    }
}