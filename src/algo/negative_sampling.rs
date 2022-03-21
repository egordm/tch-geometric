use std::collections::HashMap;
use rand::{Rng};
use crate::data::{CooGraphBuilder, CsrGraph, SparseGraph, SparseGraphType, SparseGraphTypeTrait};
use crate::utils::{EdgeType, NodeIdx, NodePtr, NodeType, RelType};

pub fn negative_sample_neighbors_homogenous(
    rng: &mut impl Rng,
    graph: &CsrGraph,
    graph_size: (i64, i64),
    inputs: &[NodeIdx],
    num_neg: i64,
    try_count: i64,
) -> (
    Vec<NodeIdx>,
    CooGraphBuilder,
    usize,
) {

    // Initialize some data structures for the sampling process
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut samples_mapping: HashMap<NodeIdx, NodePtr<usize>> = HashMap::new();

    let mut edge_index = CooGraphBuilder::default();

    samples.extend_from_slice(inputs);
    samples_mapping.extend(samples.iter().enumerate().map(|(i, &s)| (s, i)));

    let node_count = graph_size.1;
    let sample_count = samples.len();

    for (i, &v) in inputs.iter().enumerate() {
        for _ in 0..num_neg {
            for _t in 0..try_count {
                let w = rng.gen_range(0..node_count);
                if !graph.has_edge(v, w) && v != w {
                    let j = *samples_mapping.entry(w).or_insert_with(|| {
                        samples.push(w);
                        samples.len() - 1
                    });
                    edge_index.push_edge(i as i64, j as i64, -1);
                    break;
                }
            }
        }
    }

    (samples, edge_index, sample_count)
}

pub fn negative_sample_neighbors_heterogenous(
    rng: &mut impl Rng,
    node_types: &[NodeType],
    edge_types: &[EdgeType],
    graphs: &HashMap<RelType, (CsrGraph, (i64, i64))>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    num_neg: i64,
    try_count: i64,
    inbound: bool,
) -> (
    HashMap<NodeType, Vec<NodeIdx>>,
    HashMap<RelType, CooGraphBuilder>,
    HashMap<NodeType, usize>,
) {
    // Create node to reltype mapping
    let mut node_rels: HashMap<NodeType, Vec<(RelType, NodeType)>> = HashMap::default();
    for (src, rel, dst) in edge_types {
        let rel_type = format!("{}__{}__{}", src, rel, dst);
        node_rels.entry(src.clone())
            .or_insert_with(Vec::new)
            .push((rel_type, dst.clone()));
    }

    // Initialize some data structures for the sampling process
    let mut samples: HashMap<NodeType, Vec<NodeIdx>> = HashMap::new();
    let mut samples_mapping: HashMap<NodeType, HashMap<NodeIdx, NodePtr<usize>>> = HashMap::new();

    for node_type in node_types {
        let samples = samples
            .entry(node_type.clone())
            .or_insert_with(Vec::new);
        if let Some(inputs) = inputs.get(node_type) {
            samples.extend_from_slice(inputs);
        }

        let samples_mapping = samples_mapping
            .entry(node_type.clone())
            .or_insert_with(HashMap::new);
        if inputs.contains_key(node_type) {
            samples_mapping.extend(samples.iter().enumerate().map(|(i, &s)| (s, i)));
        }
    }

    let mut edge_index: HashMap<RelType, CooGraphBuilder> = graphs.keys()
        .map(|rel_type| (rel_type.clone(), CooGraphBuilder::new()))
        .collect();
    let sample_count: HashMap<NodeType, usize> = samples.iter().map(|(t, v)| (t.clone(), v.len())).collect();

    // Maintains begin/end indices for each node type
    for (node_type, inputs) in inputs {
        let node_rels = &node_rels[node_type];

        for (i, &v) in inputs.iter().enumerate() {
            for _ in 0..num_neg {
                let (rel_type, dst_type) = &node_rels[rng.gen_range(0..node_rels.len())];
                let (graph, (src_node_count, node_count)) = &graphs[rel_type];
                let samples_mapping = samples_mapping.get_mut(dst_type).unwrap();
                let samples = samples.get_mut(dst_type).unwrap();
                let edge_index = edge_index.get_mut(rel_type).unwrap();

                for _t in 0..try_count {
                    let w = rng.gen_range(0..*node_count);
                    let has_edge = match inbound {
                        true => graph.has_edge(w, v),
                        false => graph.has_edge(v, w),
                    };

                    if !has_edge && v != w {
                        let j = *samples_mapping.entry(w).or_insert_with(|| {
                            samples.push(w);
                            samples.len() - 1
                        });
                        edge_index.push_edge(i as i64, j as i64, -1);
                        break;
                    }
                }
            }
        }
    }

    (samples, edge_index, sample_count)
}

// TODO: benchmark current impl vs per thread retry

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::convert::TryFrom;
    use rand::SeedableRng;
    use tch::Tensor;
    use crate::algo::negative_sampling::negative_sample_neighbors_heterogenous;
    use crate::data::{CscGraph, CscGraphStorage, CsrGraph, CsrGraphStorage, load_fake_hetero_graph, Size};
    use crate::data::load_karate_graph;
    use crate::utils::{EdgeType, NodeIdx, NodeType, RelType, try_tensor_to_slice};

    #[test]
    pub fn test_negative_sample_neighbors() {
        let (x, _, coo_graph) = load_karate_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let node_count = x.size()[0];
        let graph_data = CsrGraphStorage::try_from(&coo_graph).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let inputs: Vec<_> = (0..node_count).collect();

        let (samples, edge_index, _sample_count) = super::negative_sample_neighbors_homogenous(
            &mut rng,
            &graph,
            (node_count, node_count),
            &inputs,
            10,
            5,
        );

        for (i, j) in edge_index.iter_edges() {
            let (v, w) = (samples[i as usize], samples[j as usize]);
            assert!(!graph.has_edge(v, w));
        }
    }

    #[test]
    pub fn test_negative_sample_neighbors_heterogenous() {
        let (xs, coo_graphs) = load_fake_hetero_graph();

        let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

        let node_types: Vec<NodeType> = xs.keys().cloned().collect();
        let edge_types: Vec<EdgeType> = coo_graphs.keys().cloned().collect();

        let mut to_edge_types = HashMap::<RelType, EdgeType>::new();
        for e @ (src_node_type, rel_type, dst_node_type) in edge_types.iter() {
            to_edge_types.insert(format!("{}__{}__{}", src_node_type, rel_type, dst_node_type), e.clone());
        }

        let graph_data: HashMap<RelType, CsrGraphStorage> = coo_graphs.iter().map(|((src, rel, dst), coo_graph)| {
            let graph_data = CsrGraphStorage::try_from(coo_graph).unwrap();
            (format!("{}__{}__{}", src, rel, dst), graph_data)
        }).collect();
        let graph_size: HashMap<RelType, Size> = to_edge_types.iter().map(|(rel_type, (src, _, dst))| {
            (rel_type.clone(), (xs[src].size()[0], xs[dst].size()[0]))
        }).collect();

        let graphs: HashMap<RelType, (CsrGraph, Size)> = graph_data.iter().map(|(rel_type, graph_data)| {
            let graph = CsrGraph::<i64, i64>::try_from(graph_data).unwrap();
            (rel_type.clone(), (graph, graph_size[rel_type]))
        }).collect();

        let inputs_data: HashMap<NodeType, Vec<NodeIdx>> = node_types.iter().map(|node_type| {
            (node_type.clone(), vec![0_i64, 1, 4, 5])
        }).collect();
        let inputs: HashMap<NodeType, &[NodeIdx]> = inputs_data.iter().map(|(node_type, inputs)| {
            (node_type.clone(), &inputs[..])
        }).collect();

        let mut node_rels = HashMap::new();
        for (rel_type, (src, _, _)) in to_edge_types.iter() {
            node_rels.entry(src.clone()).or_insert(vec![]).push(rel_type.clone());
        }

        let (samples, edge_index, _sample_count) = negative_sample_neighbors_heterogenous(
            &mut rng,
            &node_types,
            &edge_types,
            &graphs,
            &inputs,
            3,
            10,
            false,
        );

        for (rel_type, edge_index) in edge_index {
            let (graph, _) = &graphs[&rel_type];
            let (src, _, dst) = &to_edge_types[&rel_type];
            let (src_samples, dst_samples) = (&samples[src], &samples[dst]);

            for (i, j) in edge_index.iter_edges() {
                let (v, w) = (src_samples[i as usize], dst_samples[j as usize]);
                assert!(!graph.has_edge(v, w));
            }
        }
    }
}