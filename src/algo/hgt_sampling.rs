use std::collections::HashMap;
use std::ops::Range;
use rand::Rng;
use crate::data::{CooGraphBuilder, CscGraph, EdgeAttr};
use crate::utils::{EdgePtr, EdgeType, IndexOpt, NodeIdx, NodePtr, NodeType, RelType, reservoir_sampling, reservoir_sampling_weighted};

type Score = f64;
pub type Timestamp = i64;

const MAX_NEIGHBORS: usize = 50;
const NAN_TIMESTAMP: Timestamp = -1;

type NodeBudget = HashMap<NodeIdx, BudgetValue>;

#[derive(Default)]
struct BudgetDict {
    budget_dict: HashMap<NodeType, NodeBudget>,
}

#[derive(Debug, Clone, Default)]
struct BudgetValue {
    score: Score,
    timestamp: Timestamp,
}

impl BudgetDict {
    pub fn update_budget(
        &mut self,
        rng: &mut impl Rng,
        node_type: &NodeType,
        samples: &[NodeIdx],
        samples_timestamps: &[Timestamp],
        to_local_node_dict: &HashMap<NodeType, HashMap<NodeIdx, NodePtr<usize>>>,
        to_edge_types: &HashMap::<RelType, EdgeType>,
        graphs: &HashMap<RelType, (CscGraph, Option<EdgeAttr<Timestamp>>)>,
        timerange: &Option<Range<Timestamp>>
    ) {
        if samples.is_empty() {
            return;
        }

        // Initialize some helper structures
        let tmp = HashMap::new();
        let mut indices = [0_usize; MAX_NEIGHBORS];

        // Line 1: for each node type and its adjacent edge type, update the budget
        for (rel_type, (graph, timestamps)) in graphs.iter() {
            let (src, _, dst) = to_edge_types.get(rel_type).unwrap();

            if node_type != dst {
                continue;
            }

            let to_local_src_node = to_local_node_dict.get(src).unwrap_or(&tmp);
            let src_budget = self.budget_dict.entry(src.clone()).or_default();

            // Line 1: for each target node (t = w)
            for (j, w) in samples.iter().enumerate() {
                let neighbors_range = graph.neighbors_range(*w);
                if neighbors_range.is_empty() {
                    continue;
                }

                let w_timestamp = samples_timestamps[j];
                let neighbors = graph.neighbors_slice(*w);
                let neigbor_timestamps = timestamps.as_ref().map(|ts| ts.get_range(neighbors_range.clone()));

                // Line 2: Calculate normalized degree
                // There might be same neighbors with large neighborhood sizes.
                // In order to prevent that we fill our budget with many values of low
                // probability, we instead sample a fixed amount without replacement:
                let neighbor_count = reservoir_sampling(rng, 0..neighbors.len().min(MAX_NEIGHBORS), &mut indices);
                let inv_deg = 1.0 / neighbor_count as Score;

                // Line 3: for each source node (s = v)
                for i in indices[..neighbor_count].iter() {
                    let v = neighbors[*i];

                    // Line 4: Only add the neighbor in case we have not yet seen/sampled it before:
                    if !to_local_src_node.contains_key(&v) {
                        // Line 5: use source timestamp or inductively inherit from target timestamp
                        let mut v_timestamp = neigbor_timestamps.map(|ts| ts[*i]).unwrap_or(NAN_TIMESTAMP);
                        if v_timestamp == NAN_TIMESTAMP {
                            v_timestamp = w_timestamp;
                        }

                        // Check whether node is within timerange (if timerange is specified)
                        if let Some(timerange) = timerange {
                            if v_timestamp != NAN_TIMESTAMP && !timerange.contains(&v_timestamp) {
                                continue;
                            }
                        }

                        // Line 8: Update budget
                        let budget = src_budget.entry(v).or_default();
                        budget.score += inv_deg;
                        budget.timestamp = v_timestamp;
                    }
                }
            }
        }
    }

    pub fn sample_from(
        rng: &mut impl Rng,
        budget: &NodeBudget,
        num_samples: usize,
    ) -> (Vec<NodeIdx>, Vec<Timestamp>) {
        let indices = budget.iter()
            .map(|(_, budget)| budget.score * budget.score)
            .enumerate();

        let mut sampled_indices: Vec<NodePtr<usize>> = vec![0; num_samples];
        let count = reservoir_sampling_weighted(rng, indices, &mut sampled_indices);

        let mut nodes: Vec<NodeIdx> = Vec::new();
        let mut timestamps: Vec<Timestamp> = Vec::new();
        nodes.reserve(count);
        timestamps.reserve(count);

        for i in sampled_indices[..count].iter() {
            let v = *budget.keys().nth(*i).unwrap();
            nodes.push(v);
            timestamps.push(budget[&v].timestamp);
        }

        let nodes = sampled_indices[..count].iter()
            .map(|i| *budget.keys().nth(*i).unwrap())
            .collect();
        let timestamps = sampled_indices[..count].iter()
            .map(|i| budget.get(budget.keys().nth(*i).unwrap()).unwrap().timestamp)
            .collect();

        (nodes, timestamps)
    }
}

pub fn hgt_sampling(
    rng: &mut impl Rng,
    _node_types: &[NodeType],
    edge_types: &[EdgeType],
    graphs: &HashMap<RelType, (CscGraph, Option<EdgeAttr<Timestamp>>)>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    input_timestamps: Option<&HashMap<NodeType, &[Timestamp]>>,
    num_samples: &HashMap<NodeType, Vec<usize>>,
    num_hops: usize,
    timerange: &Option<Range<Timestamp>>
) -> (
    HashMap<NodeType, Vec<NodeIdx>>,
    HashMap<NodeType, Vec<Timestamp>>,
    HashMap<RelType, CooGraphBuilder>,
    // HashMap<RelType, Vec<LayerOffset>>,
) {
    // Create a mapping to convert single string relations to edge type triplets:
    let mut to_edge_types: HashMap::<RelType, EdgeType> = HashMap::new();
    for e @ (src_node_type, rel_type, dst_node_type) in edge_types {
        to_edge_types.insert(format!("{}__{}__{}", src_node_type, rel_type, dst_node_type), e.clone());
    }

    // Initialize some data structures for the sampling process
    let mut nodes_dict: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
    let mut nodes_timestamps_dict: HashMap<NodeType, Vec<Timestamp>> = HashMap::new();
    let mut to_local_node_dict: HashMap<NodeType, HashMap<NodeIdx, NodePtr<usize>>> = HashMap::new();
    let mut budget_dict: BudgetDict = BudgetDict::default();

    // Add the input nodes to the sampled output nodes (line 1):
    for (node_type, inputs) in inputs {
        let nodes = nodes_dict.entry(node_type.clone()).or_default();
        let nodes_timestamps = nodes_timestamps_dict.entry(node_type.clone()).or_default();
        let to_local_node = to_local_node_dict.entry(node_type.clone()).or_default();

        let inputs_timestamps = input_timestamps.map(|ts| *ts.get(node_type).unwrap());
        for (i, v) in inputs.iter().enumerate() {
            to_local_node.insert(*v, nodes.len());
            nodes.push(*v);
            nodes_timestamps.push(
                inputs_timestamps.get(&i).cloned().unwrap_or(NAN_TIMESTAMP)
            );
        }
    }

    // Update the budget based on the initial input set (line 3-5):
    for (node_type, inputs) in nodes_dict.iter() {
        let inputs_timestamps = &nodes_timestamps_dict[node_type];

        budget_dict.update_budget(
            rng,
            node_type,
            inputs,
            inputs_timestamps,
            &to_local_node_dict,
            &to_edge_types,
            graphs,
            timerange,
        );
    }

    for layer in 0..num_hops {
        let mut samples_dict: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
        let mut samples_timestamps_dict: HashMap<NodeType, Vec<Timestamp>> = HashMap::new();
        for (node_type, budget) in budget_dict.budget_dict.iter_mut() {
            let num_samples = num_samples[node_type][layer];

            // Sample `num_samples` nodes, according to the budget (line 9-11):
            let (samples, timestamps) = BudgetDict::sample_from(rng, budget, num_samples);
            samples_dict.insert(node_type.clone(), samples);
            let samples = &samples_dict[node_type];
            samples_timestamps_dict.insert(node_type.clone(), timestamps);
            let timestamps = &samples_timestamps_dict[node_type];

            // Add samples to the sampled output nodes, and erase them from the budget
            // (line 13/15):
            let nodes = nodes_dict.entry(node_type.clone()).or_default();
            let nodes_timestamps = nodes_timestamps_dict.entry(node_type.clone()).or_default();
            let to_local_node = to_local_node_dict.entry(node_type.clone()).or_default();
            for (i, v) in samples.iter().enumerate() {
                to_local_node.insert(*v, nodes.len());
                nodes.push(*v);
                nodes_timestamps.push(timestamps[i]);
                budget.remove(v);
            }
        }

        if layer < num_hops - 1 {
            // Add neighbors of newly sampled nodes to the budget (line 14):
            // Note that we do not need to update the budget in the last iteration.
            for (node_type, samples) in samples_dict.iter() {
                let samples_timestamps = &samples_timestamps_dict[node_type];

                budget_dict.update_budget(
                    rng,
                    node_type,
                    samples,
                    samples_timestamps,
                    &to_local_node_dict,
                    &to_edge_types,
                    graphs,
                    timerange,
                );
            }
        }
    }

    // Reconstruct the sampled adjacency matrix among the sampled nodes (line 19):
    let mut out_edge_list_dict: HashMap<RelType, CooGraphBuilder> = HashMap::new();

    for (rel_type, (graph, _)) in graphs {
        let (src, _, dst) = &to_edge_types[rel_type];

        let out_edge_list = out_edge_list_dict.entry(rel_type.clone()).or_default();
        let dst_nodes = nodes_dict.entry(dst.clone()).or_default();
        let to_local_src_node = to_local_node_dict.entry(src.clone()).or_default();

        for (i, w) in dst_nodes.iter().enumerate() {
            let neighbors_range = graph.neighbors_range(*w);
            let neighbors = graph.neighbors_slice(*w);

            let mut indices: Vec<EdgePtr<usize>> = vec![0; neighbors.len().min(MAX_NEIGHBORS)];
            reservoir_sampling(rng, neighbors_range.clone(), &mut indices[..]);

            for edge_ptr in indices {
                let v = neighbors[edge_ptr - neighbors_range.start];
                if let Some(j) = to_local_src_node.get(&v) {
                    out_edge_list.push_edge(*j as NodeIdx, i as NodeIdx, edge_ptr as NodePtr);
                }
            }
        }
    }

    let out_node_dict: HashMap<NodeType, Vec<NodeIdx>> = nodes_dict;
    let out_node_timestamps_dict: HashMap<NodeType, Vec<Timestamp>> = nodes_timestamps_dict;

    (
        out_node_dict,
        out_node_timestamps_dict,
        out_edge_list_dict,
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

    pub fn validate_neighbor_samples(
        graph: &CscGraph<i64, i64>,
        coo_builder: &CooGraphBuilder,
        samples_src: &[NodeIdx],
        samples_dst: &[NodeIdx],
        // layer_offsets: &[LayerOffset],
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

        // let mut begin = 0;
        // TODO: must count per node type not per relation
        for i in 0..5 {
            assert!((0..=20).contains(&counts[i as usize]));
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
    pub fn test_hgt_sampling() {
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

        let num_samples: HashMap<NodeType, Vec<usize>> = node_types.iter().cloned().map(|node_types| {
            (node_types, vec![20, 15])
        }).collect();
        let num_hops = 2;

        let (samples, sample_timestamps, coo_builders) = super::hgt_sampling(
            &mut rng,
            &node_types,
            &edge_types,
            &graphs,
            &inputs,
            None,
            &num_samples,
            num_hops,
            &None,
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
                // &layer_offsets[rel_type],
                &num_samples[dst_node_type],
            );
        }
    }
}