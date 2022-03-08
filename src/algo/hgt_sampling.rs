use std::collections::HashMap;
use std::hash::Hash;
use std::usize::MAX;
use pyo3::iter::iter;
use rand::Rng;
use crate::algo::neighbor_sampling::{LayerOffset, Sampler, SamplingFilter};
use crate::data::{CooGraphBuilder, CscGraph};
use crate::utils::{EdgePtr, EdgeType, NodeIdx, NodePtr, NodeType, RelType, reservoir_sampling, reservoir_sampling_weighted};

type Score = f64;
type Weight = f64;
type Timestamp = f64;

const MAX_NEIGHBORS: usize = 50;

type NodeBudget = HashMap<NodeIdx, BudgetValue>;

struct BudgetDict {
    budget_dict: HashMap<NodeType, NodeBudget>,
}

impl Default for BudgetDict {
    fn default() -> Self {
        BudgetDict {
            budget_dict: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct BudgetValue {
    score: Score,
    timestamp: Option<Timestamp>,
}

impl Default for BudgetValue {
    fn default() -> Self {
        BudgetValue {
            score: 0.0,
            timestamp: None,
        }
    }
}

impl BudgetDict {
    pub fn update_budget(
        &mut self,
        rng: &mut impl Rng,
        node_type: &NodeType,
        samples: &[NodeIdx],
        to_local_node_dict: &HashMap<NodeType, HashMap<NodeIdx, NodePtr<usize>>>,
        to_edge_types: &HashMap::<RelType, EdgeType>,
        graphs: &HashMap<RelType, CscGraph>,
    ) {
        if samples.is_empty() {
            return;
        }

        let tmp = HashMap::new();
        let mut indices = [0 as NodeIdx; MAX_NEIGHBORS];
        for (rel_type, graph) in graphs.iter() {
            let (src, _, dst) = to_edge_types.get(rel_type).unwrap();

            if node_type != dst {
                continue;
            }

            let to_local_src_node = to_local_node_dict.get(src).unwrap_or(&tmp);
            let src_budget = self.budget_dict.entry(src.clone()).or_insert_with(HashMap::new);

            for w in samples {
                let neighbors = graph.neighbors_slice(*w);
                if neighbors.is_empty() {
                    continue;
                }

                let mut inv_deg;
                let iter = if neighbors.len() > MAX_NEIGHBORS {
                    // There might be same neighbors with large neighborhood sizes.
                    // In order to prevent that we fill our budget with many values of low
                    // probability, we instead sample a fixed amount without replacement:
                    reservoir_sampling(rng, neighbors.iter().cloned(), &mut indices);

                    inv_deg = 1.0 / indices.len() as Score;
                    indices.iter()
                } else {
                    inv_deg = 1.0 / neighbors.len() as Score;
                    neighbors.iter()
                };

                for v in iter {
                    // Only add the neighbor in case we have not yet seen it before:
                    if !to_local_src_node.contains_key(v) {
                        let budget = src_budget.entry(*v).or_default();
                        budget.score += inv_deg;
                        // TODO: set timestamp
                    }
                }
            }
        }
    }

    pub fn sample_from(
        rng: &mut impl Rng,
        budget: &NodeBudget,
        num_samples: usize,
    ) -> Vec<NodeIdx> {
        let mut indices: Vec<(NodeIdx, Weight)> = Vec::new();
        indices.reserve(budget.len());

        for (node, budget_value) in budget.iter() {
            let weight = budget_value.score * budget_value.score;
            indices.push((*node, weight));
        }

        let mut samples = vec![0 as NodeIdx; num_samples];
        let count = reservoir_sampling_weighted(rng, indices.into_iter(), &mut samples);

        samples[..count].to_vec()
    }
}

pub fn hgt_sampling(
    rng: &mut impl Rng,
    node_types: &[NodeType],
    edge_types: &[EdgeType],
    graphs: &HashMap<RelType, CscGraph>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    num_samples: &HashMap<NodeType, Vec<usize>>,
    num_hops: usize,
) -> (
    HashMap<NodeType, Vec<NodeIdx>>,
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
    let mut to_local_node_dict: HashMap<NodeType, HashMap<NodeIdx, NodePtr<usize>>> = HashMap::new();
    let mut budget_dict: BudgetDict = BudgetDict::default();

    // Add the input nodes to the sampled output nodes (line 1):
    for (node_type, inputs) in inputs {
        let mut nodes = nodes_dict.entry(node_type.clone()).or_insert_with(Vec::new);
        let to_local_node = to_local_node_dict.entry(node_type.clone()).or_insert_with(HashMap::new);

        for v in *inputs {
            to_local_node.insert(*v, nodes.len());
            nodes.push(*v);
        }
    }

    // Update the budget based on the initial input set (line 3-5):
    for (node_type, inputs) in nodes_dict.iter() {
        budget_dict.update_budget(
            rng,
            node_type,
            inputs,
            &to_local_node_dict,
            &to_edge_types,
            &graphs,
        );
    }

    for layer in 0..num_hops {
        let mut samples_dict: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
        for (node_type, budget) in budget_dict.budget_dict.iter_mut() {
            let num_samples = num_samples[node_type][layer];

            // Sample `num_samples` nodes, according to the budget (line 9-11):
            let samples = BudgetDict::sample_from(rng, budget, num_samples);
            samples_dict.insert(node_type.clone(), samples);
            let samples = &samples_dict[node_type];

            // Add samples to the sampled output nodes, and erase them from the budget
            // (line 13/15):
            let nodes = nodes_dict.entry(node_type.clone()).or_default();
            let to_local_node = to_local_node_dict.entry(node_type.clone()).or_default();
            for v in samples {
                to_local_node.insert(*v, nodes.len());
                nodes.push(*v);
                budget.remove(v);
            }
        }

        if layer < num_hops - 1 {
            // Add neighbors of newly sampled nodes to the budget (line 14):
            // Note that we do not need to update the budget in the last iteration.
            for (node_type, samples) in samples_dict.iter() {
                budget_dict.update_budget(
                    rng,
                    node_type,
                    samples,
                    &to_local_node_dict,
                    &to_edge_types,
                    &graphs,
                );
            }
        }
    }

    // Reconstruct the sampled adjacency matrix among the sampled nodes (line 19):
    let mut out_node_dict: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
    let mut out_edge_list_dict: HashMap<RelType, CooGraphBuilder> = HashMap::new();

    for (rel_type, graph) in graphs {
        let (src, _, dst) = &to_edge_types[rel_type];

        let mut out_edge_list = out_edge_list_dict.entry(rel_type.clone()).or_default();
        let dst_nodes = nodes_dict.entry(dst.clone()).or_default();
        let to_local_src_node = to_local_node_dict.entry(src.clone()).or_default();

        for (i, w) in dst_nodes.iter().enumerate() {
            let neighbors_range = graph.neighbors_range(*w);
            let neighbors = graph.neighbors_slice(*w);

            let mut indices = vec![0 as EdgePtr<usize>; neighbors.len().min(MAX_NEIGHBORS)];
            reservoir_sampling(rng, neighbors_range.clone(), &mut indices[..]);

            for edge_ptr in indices {
                let v = neighbors[edge_ptr - neighbors_range.start];
                if let Some(j) = to_local_src_node.get(&v) {
                    out_edge_list.push_edge(*j as NodeIdx, i as NodeIdx, edge_ptr as NodePtr);
                }
            }
        }
    }

    for (node_type, nodes) in nodes_dict {
        if !nodes.is_empty() {
            out_node_dict.insert(node_type.clone(), nodes);
        }
    }

    (
        out_node_dict,
        out_edge_list_dict,
    )
}