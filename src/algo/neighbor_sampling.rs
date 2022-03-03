#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::ops::{Neg, RangeInclusive, Sub};
use std::slice::Iter;
use num_traits::Float;
use rand::{Rng};
use rand::distributions::uniform::SampleUniform;
use crate::data::EdgeAttr;
use crate::data::graph::{CscGraph, EdgeIndexBuilder};
use crate::utils::{EdgePtr, EdgeType, NodeType, RelType, replacement_sampling, reservoir_sampling, reservoir_sampling_weighted};
use crate::utils::types::{NodeIdx, NodePtr};

pub trait SamplingFilter {
    type State: Copy;

    fn init(&self, samples: &[NodeIdx], s: &mut Vec<Self::State>);

    fn filter(&self, state: &Self::State, src: NodeIdx, dst: EdgePtr<usize>) -> bool;

    fn mutate(&self, state: &Self::State, src: NodeIdx, dst: EdgePtr<usize>) -> Self::State;
}

pub struct IdentityFilter;

impl SamplingFilter for IdentityFilter {
    type State = ();

    fn init(&self, samples: &[NodeIdx], s: &mut Vec<Self::State>) {
        s.resize(samples.len(), ());
    }

    fn filter(&self, state: &Self::State, src: NodeIdx, dst: EdgePtr<usize>) -> bool { true }

    fn mutate(&self, state: &Self::State, src: NodeIdx, dst: EdgePtr<usize>) -> Self::State {}
}

pub const TEMPORAL_SAMPLE_STATIC: usize = 0;
pub const TEMPORAL_SAMPLE_RELATIVE: usize = 1;
pub const TEMPORAL_SAMPLE_DYNAMIC: usize = 2;

pub struct TemporalFilter<'a, T, const FORWARD: bool, const MODE: usize> {
    window: RangeInclusive<T>,
    timestamps: EdgeAttr<'a, T>,
    initial_state: &'a [T],
}

impl<
    'a, T, const FORWARD: bool, const MODE: usize
> TemporalFilter<'a, T, FORWARD, MODE> {
    pub fn new(window: RangeInclusive<T>, timestamps: EdgeAttr<'a, T>, initial_state: &'a [T]) -> Self {
        TemporalFilter {
            window,
            timestamps,
            initial_state,
        }
    }
}

impl<
    'a, T: Copy + PartialOrd + Neg<Output=T> + Sub<Output=T>, const FORWARD: bool, const MODE: usize
> SamplingFilter for TemporalFilter<'a, T, FORWARD, MODE> {
    type State = T;

    fn init(&self, _samples: &[NodeIdx], s: &mut Vec<Self::State>) {
        s.extend_from_slice(self.initial_state);
    }

    fn filter(&self, state: &Self::State, _src: NodeIdx, dst: EdgePtr<usize>) -> bool {
        let t = self.timestamps.get(dst);
        match MODE {
            TEMPORAL_SAMPLE_STATIC => self.window.contains(t),
            TEMPORAL_SAMPLE_RELATIVE | TEMPORAL_SAMPLE_DYNAMIC => {
                match FORWARD {
                    true => self.window.contains(&(*t - *state)),
                    false => self.window.contains(&(*t - *state).neg()),
                }
            },
            _ => unreachable!(),
        }
    }

    fn mutate(&self, state: &Self::State, _src: NodeIdx, dst: EdgePtr<usize>) -> Self::State {
        match MODE {
            TEMPORAL_SAMPLE_STATIC => *state,
            TEMPORAL_SAMPLE_RELATIVE =>  *state,
            TEMPORAL_SAMPLE_DYNAMIC =>  *self.timestamps.get(dst),
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

pub fn neighbor_sampling_homogenous<
    F: SamplingFilter
>(
    rng: &mut impl Rng,
    graph: &CscGraph,
    sampler: &impl Sampler,
    filter: &F,
    inputs: &[NodeIdx],
    num_neighbors: &[usize],
) -> (Vec<NodeIdx>, EdgeIndexBuilder, Vec<(NodePtr, EdgePtr)>) {
    // Initialize some data structures for the sampling process
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut states: Vec<F::State> = Vec::new();

    let mut layer_offsets: Vec<(NodePtr, EdgePtr)> = Vec::new();
    let mut edge_index = EdgeIndexBuilder::new();

    samples.extend_from_slice(inputs);
    filter.init(&samples, &mut states);

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors.iter().cloned() {
        // Initialize the states
        let mut sampler_state = sampler.init(num_samples);

        // Add layer offset
        layer_offsets.push((samples.len() as NodePtr, edge_index.len() as EdgePtr));

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
    graphs: &HashMap<RelType, &CscGraph>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    num_neighbors: &HashMap<NodeType, &[usize]>,
    num_hops: usize,
    sampler: &HashMap<RelType, &impl Sampler>,
    filter: &HashMap<RelType, &F>,
) -> (HashMap<NodeType, Vec<NodeIdx>>, HashMap<RelType, EdgeIndexBuilder>) {

    // Create a mapping to convert single string relations to edge type triplets:
    let mut to_edge_types = HashMap::<RelType, EdgeType>::new();
    for e@(src_node_type, rel_type, dst_node_type) in edge_types {
        to_edge_types.insert(format!("{}__{}__{}", src_node_type, rel_type, dst_node_type), e.clone());
    }

    // Initialize some data structures for the sampling process
    let mut samples: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
    let mut states: HashMap<NodeType, Vec<F::State>> = HashMap::new();

    for node_type in node_types {
        samples
            .entry(node_type.clone())
            .or_insert_with(Vec::new)
            .extend_from_slice(inputs[node_type]);

        let states = states.entry(node_type.clone()).or_insert_with(Vec::new);
        filter[node_type].init(&samples[node_type], states);
    }

    let mut edge_index: HashMap<RelType, EdgeIndexBuilder> = graphs.keys()
        .map(|rel_type| (rel_type.clone(), EdgeIndexBuilder::new()))
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
            let filter = filter[rel_type];
            let sampler = sampler[rel_type];

            // Initialize the states
            let mut sampler_state = sampler.init(num_samples);

            // Select data
            let dst_samples = &samples[dst_node_type];
            let dst_states = &states[dst_node_type];
            // This should be safe so long we don't keep references to the samples' value (should copy them immediately)
            let src_samples = unsafe {(&samples[src_node_type] as *const _ as *mut Vec<NodeIdx>).as_mut()}.unwrap();
            let src_states = unsafe {(&states[src_node_type] as *const _ as *mut Vec<F::State>).as_mut()}.unwrap();

            let graph = graphs[rel_type];
            let edge_index = edge_index.get_mut(rel_type).unwrap();

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
    )
}


#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::TryFrom;
    use rand::{Rng, SeedableRng};
    use crate::algo::neighbor_sampling::{IdentityFilter, TemporalFilter, UnweightedSampler, WeightedSampler};
    use crate::data::{CscGraph, CscGraphData, EdgeAttr, EdgeIndexBuilder};
    use crate::data::tests::load_karate_graph;
    use crate::utils::{EdgePtr, NodeIdx, NodePtr};
    use super::{TEMPORAL_SAMPLE_STATIC, TEMPORAL_SAMPLE_RELATIVE};

    pub fn validate_neighbor_samples(
        graph: &CscGraph<i64, i64>,
        edge_index: &EdgeIndexBuilder,
        samples: &[NodeIdx],
        layer_offsets: &[(NodePtr, EdgePtr)],
        num_neighbors: &[usize],
    ) {
        for (j, i) in edge_index.rows.iter().zip(edge_index.cols.iter()) {
            let v = samples[*j as usize];
            let w = samples[*i as usize];
            assert!(graph.has_edge(v, w));
        }

        let mut counts = vec![0_usize; samples.len()];
        for i in layer_offsets.iter().last().unwrap().0 as usize..samples.len() {
            counts[i] += 1;
        }

        for (j, i) in edge_index.rows.iter().rev().zip(edge_index.cols.iter().rev()) {
            counts[*i as usize] += counts[*j as usize];
        }

        let mut begin = 0;
        for (i, (end, _)) in layer_offsets.iter().cloned().enumerate() {
            let max_neighbors: usize = num_neighbors[0..num_neighbors.len() - i].iter().product();

            for i in begin..end {
                assert!((0..=max_neighbors).contains(&counts[i as usize]));
            }
            begin = end;
        }
    }

    pub fn samples_to_paths(
        edge_index: &EdgeIndexBuilder,
        samples: &[NodeIdx],
        inputs: &[NodeIdx],
    ) -> Vec<(Vec<NodeIdx>, Vec<usize>)> {
        let mut paths = inputs.iter().map(|&i| (vec![i], vec![])).collect::<VecDeque<_>>();
        let mut head = vec![-1];
        let mut head_edges = vec![];
        for ((j, i), edge_idx) in edge_index.rows.iter()
            .zip(edge_index.cols.iter())
            .zip(0..edge_index.edge_index.len())
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
        let (x, _, edge_index) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphData::try_from_edge_index(&edge_index, x.size()[0]).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();
        let inputs = vec![0_i64, 1, 4, 5];
        let num_neighbors = vec![4, 3];

        let (samples, edge_index, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &UnweightedSampler::<true>,
            &IdentityFilter,
            &inputs,
            &num_neighbors,
        );

        validate_neighbor_samples(&graph, &edge_index, &samples, &layer_offsets, &num_neighbors);
    }

    #[test]
    pub fn test_neighbor_sampling_homogenous_weighted() {
        let (x, _, edge_index) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphData::try_from_edge_index(&edge_index, x.size()[0]).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let weights_data = (0..graph.edge_count()).map(|_| rng.gen_range(0.2..5.0)).collect::<Vec<f64>>();
        let weights = EdgeAttr::new(&weights_data);

        let inputs = vec![0_i64, 1, 4, 5];
        let num_neighbors = vec![4, 3];

        let (samples, edge_index, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &WeightedSampler::new(weights),
            &IdentityFilter,
            &inputs,
            &num_neighbors,
        );

        validate_neighbor_samples(&graph, &edge_index, &samples, &layer_offsets, &num_neighbors);
    }

    #[test]
    pub fn test_neighbor_sampling_homogenous_temporal() {
        let (x, _, edge_index) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let graph_data = CscGraphData::try_from_edge_index(&edge_index, x.size()[0]).unwrap();
        let graph = CscGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let timestamps_data = (0..graph.edge_count()).map(|_| rng.gen_range(0..4)).collect::<Vec<i64>>();
        let timestamps = EdgeAttr::new(&timestamps_data);

        let inputs = vec![0_i64, 1, 4, 5];
        let input_timestamps = vec![0_i64, 1, 2, 3];
        let num_neighbors = vec![4, 3];

        // Tests static window sampling
        let filter = TemporalFilter::<i64, false, TEMPORAL_SAMPLE_STATIC>::new(
            0..=2, timestamps.clone(), &input_timestamps
        );
        let (samples, edge_index, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &UnweightedSampler::<false>,
            &filter,
            &inputs,
            &num_neighbors,
        );

        validate_neighbor_samples(&graph, &edge_index, &samples, &layer_offsets, &num_neighbors);
        let paths = samples_to_paths(&edge_index, &samples, &inputs);
        for (_path, edges) in paths {
            for edge_idx in edges {
                let edge_ptr = edge_index.edge_index[edge_idx];
                let t = timestamps.get(edge_ptr as usize);
                assert!((0..=2).contains(t));
            }
        }

        // Tests relative window sampling backward in time
        let filter = TemporalFilter::<i64, false, TEMPORAL_SAMPLE_RELATIVE>::new(
            0..=2, timestamps.clone(), &input_timestamps
        );
        let (samples, edge_index, layer_offsets) = super::neighbor_sampling_homogenous(
            &mut rng,
            &graph,
            &UnweightedSampler::<false>,
            &filter,
            &inputs,
            &num_neighbors,
        );

        validate_neighbor_samples(&graph, &edge_index, &samples, &layer_offsets, &num_neighbors);
        let paths = samples_to_paths(&edge_index, &samples, &inputs);
        for (_path, edges) in paths {
            if let Some(edge_idx) = edges.first().cloned() {
                let to_idx = edge_index.cols[edge_idx];
                let start_t = input_timestamps[to_idx as usize];

                for edge_idx in edges {
                    let edge_ptr = edge_index.edge_index[edge_idx];
                    let t = *timestamps.get(edge_ptr as usize);

                    assert!(((start_t - 2)..=start_t).contains(&t));
                }
            }
        }
    }
}