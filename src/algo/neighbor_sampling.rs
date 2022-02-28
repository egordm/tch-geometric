use std::slice::Iter;
use num_traits::Float;
use rand::{Rng};
use rand::distributions::uniform::SampleUniform;
use crate::data::EdgeAttr;
use crate::data::graph::{CscGraph, EdgeIndexBuilder};
use crate::utils::{EdgeIdx, EdgePtr, replacement_sampling, reservoir_sampling, reservoir_sampling_weighted};
use crate::utils::types::{NodeIdx, NodePtr};

pub trait SamplingFilter {
    type State;

    fn init(&self, n: NodeIdx) -> Self::State;

    fn mutate(&self, state: Self::State, src: NodeIdx, e: EdgeIdx, dst: NodeIdx) -> Self::State;

    fn filter(&self, state: Self::State, src: NodeIdx, e: EdgeIdx, dst: NodeIdx) -> bool;
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

            let n = replacement_sampling(rng, &state.0, &mut state.1);
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
>(
    rng: &mut impl Rng,
    graph: &CscGraph,
    sampler: &impl Sampler,
    inputs: &[NodeIdx],
    num_neighbors: &[usize],
) -> (Vec<NodeIdx>, EdgeIndexBuilder, Vec<(NodePtr, EdgePtr)>) {
    // Initialize some data structures for the sampling process
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut layer_offsets: Vec<(NodePtr, EdgePtr)> = Vec::new();
    let mut edge_index = EdgeIndexBuilder::new();

    samples.extend_from_slice(inputs);

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors.iter().cloned() {
        // Initialize the sampler state
        let mut sampler_state = sampler.init(num_samples);

        // Add layer offset
        layer_offsets.push((samples.len() as NodePtr, edge_index.len() as EdgePtr));

        for i in begin..end {
            let w = samples[i];

            let neighbors_range = graph.neighbors_range(w);
            if neighbors_range.is_empty() {
                continue;
            }

            let samples_iter = sampler.sample(
                rng, &mut sampler_state, neighbors_range.clone()
            );

            for edge_ptr in samples_iter {
                let v = graph.get_by_ptr(*edge_ptr);
                let j = samples.len();

                samples.push(v);
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

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use rand::{Rng, SeedableRng};
    use crate::algo::neighbor_sampling::{UnweightedSampler, WeightedSampler};
    use crate::data::{CscGraph, CscGraphData, EdgeAttr};
    use crate::data::tests::load_karate_graph;

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
            &inputs,
            &num_neighbors
        );

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
                assert!((1..=max_neighbors).contains(&counts[i as usize]));
            }
            begin = end;
        }
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
            &inputs,
            &num_neighbors
        );

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
                assert!((1..=max_neighbors).contains(&counts[i as usize]));
            }
            begin = end;
        }
    }

}