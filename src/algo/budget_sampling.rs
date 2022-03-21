use std::collections::HashMap;
use std::ops::Range;
use rand::Rng;
use crate::algo::neighbor_sampling::{LayerOffset, SamplingFilter};
use crate::data::{CooGraphBuilder, CscGraph, EdgeAttr};
use crate::utils::{EdgePtr, EdgeType, IndexOpt, NodeIdx, NodePtr, NodeType, RelType, reservoir_sampling};

pub type Timestamp = i64;

const MAX_NEIGHBORS: usize = 50;
const NAN_TIMESTAMP: Timestamp = -1;

#[derive(Clone)]
struct BudgetValue {
    edge: (RelType, EdgePtr, usize),
    timestamp: Timestamp,
}

impl Default for BudgetValue {
    fn default() -> Self {
        Self {
            edge: ("".to_string(), 0, 0), // TODO: make this ints
            timestamp: NAN_TIMESTAMP,
        }
    }
}


#[derive(Default, Clone)]
struct Budget {
    data: HashMap<(NodeType, NodeIdx), BudgetValue>, // TODO: Make node type an int
}

impl Budget {
    fn update(
        rng: &mut impl Rng,
        node_type: &NodeType,
        nodes: &[NodeIdx],
        nodes_timestamps: &[Timestamp],
        to_edge_types: &HashMap<RelType, EdgeType>,
        graphs: &HashMap<RelType, (CscGraph, Option<EdgeAttr<Timestamp>>)>,
        window: &Option<Range<Timestamp>>,
    ) -> Vec<Budget> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Initialize some helper structures
        let mut indices = [0_usize; MAX_NEIGHBORS];
        let mut budgets = vec![Budget::default(); nodes.len()];

        for (rel_type, (graph, timestamps)) in graphs.iter() {
            let (src, _, dst) = &to_edge_types[rel_type];

            if node_type != dst {
                continue;
            }

            for (j, w) in nodes.iter().enumerate() {
                let neighbors_range = graph.neighbors_range(*w);
                if neighbors_range.is_empty() {
                    continue;
                }

                let w_t = nodes_timestamps[j];
                let w_budget = &mut budgets[j];

                let neighbors = graph.neighbors_slice(*w);
                let neigbor_timestamps = timestamps.as_ref().map(|ts| ts.get_range(neighbors_range.clone()));

                let neighbor_count = reservoir_sampling(rng, 0..neighbors.len().min(MAX_NEIGHBORS), &mut indices);
                for i in indices[..neighbor_count].iter() {
                    let v = neighbors[*i];
                    let mut v_t = neigbor_timestamps.get(i).cloned().unwrap_or(NAN_TIMESTAMP);
                    if v_t == NAN_TIMESTAMP {
                        v_t = w_t;
                    }

                    // TODO: add relative windowing here (instead of absolute)
                    // if let Some(window) = window {
                    //     if v_t != NAN_TIMESTAMP && !window.contains(&v_t) {
                    //         continue;
                    //     }
                    // }

                    let budget = w_budget.data.entry((src.clone(), v)).or_default();
                    let admit = if budget.edge.2 > 0 { rng.gen_range(0..budget.edge.2) } else { 0 };
                    if admit < 1 { // Reservoir sample the parallel edge
                        budget.edge = (rel_type.clone(), *i as EdgePtr, budget.edge.2 + 1);
                        budget.timestamp = v_t;
                    }
                }
            }
        }

        budgets
    }

    fn sample(
        rng: &mut impl Rng,
        j: NodePtr,
        budget: &Budget,
        nodes: &mut HashMap<NodeType, Vec<NodeIdx>>,
        nodes_timestamps: &mut HashMap<NodeType, Vec<Timestamp>>,
        edges: &mut HashMap<RelType, CooGraphBuilder>,
        num_samples: usize,
    ) {
        let mut idx = vec![0_usize; num_samples];
        let sample_count = reservoir_sampling(rng, 0..budget.data.len(), &mut idx);

        for (_, ((node_type, v), bv)) in budget.data
            .iter().enumerate()
            .filter(|(id, _)| idx[..sample_count].contains(id)) {
            let BudgetValue { edge: (rel_type, edge_ptr, _), timestamp } = bv;

            let i = nodes[node_type].len() as NodePtr;
            nodes.get_mut(node_type).unwrap().push(*v);
            nodes_timestamps.get_mut(node_type).unwrap().push(*timestamp);
            edges.get_mut(rel_type).unwrap().push_edge(i, j, *edge_ptr);
        }
    }
}

pub fn budget_neighbor_sampling_heterogenous(
    rng: &mut impl Rng,
    node_types: &[NodeType],
    edge_types: &[EdgeType],
    graphs: &HashMap<RelType, (CscGraph, Option<EdgeAttr<Timestamp>>)>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    inputs_timestamps: &HashMap<NodeType, &[Timestamp]>,
    num_neighbors: &HashMap<NodeType, Vec<usize>>,
    num_hops: usize,
    window: &Option<Range<Timestamp>>,
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
    let mut samples_timestamps: HashMap<NodeType, Vec<Timestamp>> = HashMap::new();

    for node_type in node_types {
        let samples = samples
            .entry(node_type.clone())
            .or_insert_with(Vec::new);
        if let Some(inputs) = inputs.get(node_type) {
            samples.extend_from_slice(inputs);
        }

        let samples_timestamps = samples_timestamps
            .entry(node_type.clone())
            .or_insert_with(Vec::new);
        if let Some(inputs_state) = inputs_timestamps.get(node_type) {
            samples_timestamps.extend_from_slice(inputs_state);
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


    let mut budgets: HashMap<NodeType, Vec<Budget>> = HashMap::new();
    for (node_type, nodes) in samples.iter() {
        let type_budgets = Budget::update(
            rng, node_type, nodes, &samples_timestamps[node_type],
            &to_edge_types, graphs, &None, // TODO: specify timerange
        );

        budgets.insert(node_type.clone(), type_budgets);
    }


    for layer in 0..num_hops {
        // Sample neighbors for each input node
        for node_type in node_types {
            let num_samples = num_neighbors[node_type][layer];
            let budgets = &budgets[node_type];
            let (begin, end) = slices[node_type];

            for (j, budget) in (begin..end).zip(budgets.iter()) {
                Budget::sample(
                    rng, j as NodePtr, budget,
                    &mut samples, &mut samples_timestamps, &mut edge_index,
                    num_samples,
                );
            }
        }

        // Shift input ranges
        for (node_type, samples) in samples.iter() {
            let (_, end) = slices[node_type];
            *slices.get_mut(node_type).unwrap() = (end, samples.len());
        }

        if layer < num_hops - 1 {
            // Fill budgets for input nodes
            for (node_type, nodes) in samples.iter() {
                let (begin, end) = slices[node_type];
                let type_budgets = Budget::update(
                    rng, node_type, &nodes[begin..end], &samples_timestamps[node_type][begin..end],
                    &to_edge_types, graphs, &None, // TODO: specify timerange
                );

                budgets.insert(node_type.clone(), type_budgets);
            }
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
    use crate::algo::hgt_sampling::Timestamp;
    use crate::algo::neighbor_sampling::{IdentityFilter, LayerOffset, SamplingFilter, TemporalFilter, UnweightedSampler, WeightedSampler};
    use crate::data::{CscGraph, CscGraphStorage, EdgeAttr, CooGraphBuilder};
    use crate::data::{load_fake_hetero_graph, load_karate_graph};
    use crate::utils::{EdgeType, NodeIdx, NodeType, RelType};

    pub fn validate_neighbor_samples(
        graph: &CscGraph<i64, i64>,
        coo_builder: &CooGraphBuilder,
        samples_src: &[NodeIdx],
        samples_dst: &[NodeIdx],
        layer_offsets: &[LayerOffset],
        num_neighbors: &[usize],
    ) {
        // Validate whether all edges are valid
        for (j, i) in coo_builder.iter_edges() {
            let v = samples_src[j as usize];
            let w = samples_dst[i as usize];
            // Query for dst <- src edge because we are operating on csc graph
            assert!(graph.has_edge(w, v));
        }

        // Validate whether none of the nodes exceed the sampled number of neighbors
        let mut counts = vec![0_usize; samples_dst.len()];
        for (_j, i) in coo_builder.iter_edges() {
            counts[i as usize] += 1;
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
        let graphs: HashMap<RelType, (CscGraph, _)> = graph_data.iter().map(|(rel_type, graph_data)| {
            let graph = CscGraph::<i64, i64>::try_from(graph_data).unwrap();
            (rel_type.clone(), (graph, None))
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

        let inputs_timestamp_data: HashMap<NodeType, Vec<Timestamp>> = inputs.iter().map(|(node_type, inputs)| {
            (node_type.clone(), vec![0; inputs.len()])
        }).collect();
        let inputs_timestamp: HashMap<NodeType, &[Timestamp]> = inputs_timestamp_data.iter().map(|(node_type, input_state)| {
            (node_type.clone(), &input_state[..])
        }).collect();

        let num_neighbors: HashMap<NodeType, Vec<usize>> = node_types.iter().cloned().map(|node_types| {
            (node_types, vec![20, 15])
        }).collect();
        let num_hops = 2;

        let (samples, coo_builders, layer_offsets) = super::budget_neighbor_sampling_heterogenous(
            &mut rng,
            &node_types,
            &edge_types,
            &graphs,
            &inputs,
            &inputs_timestamp,
            &num_neighbors,
            num_hops,
            &None
        );

        let graphs: HashMap<RelType, CscGraph> = graph_data.iter().map(|(rel_type, graph_data)| {
            let graph = CscGraph::<i64, i64>::try_from(graph_data).unwrap();
            (rel_type.clone(), (graph))
        }).collect();

        for rel_type in coo_builders.keys() {
            let (src_node_type, _, dst_node_type) = &to_edge_types[rel_type];

            validate_neighbor_samples(
                &graphs[rel_type],
                &coo_builders[rel_type],
                &samples[src_node_type],
                &samples[dst_node_type],
                &layer_offsets[rel_type],
                &num_neighbors[src_node_type],
            );
        }
    }
}