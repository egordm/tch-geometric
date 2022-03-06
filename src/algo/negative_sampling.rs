use std::collections::HashMap;
use rand::{Rng};
use tch::{ Tensor};
use rayon::prelude::*;
use crate::data::{CooGraphBuilder, CsrGraph};
use crate::utils::{NodeIdx, NodeType, random, RelType, TensorResult, try_tensor_to_slice, try_tensor_to_slice_mut};

pub fn negative_sample_neighbors(
    graph: &CsrGraph,
    graph_size: (i64, i64),
    inputs: &Tensor,
    num_neg: i64,
    try_count: i64,
) -> TensorResult<(Tensor, Vec<usize>)> {
    let node_count = graph_size.1;
    let v_data = inputs.repeat_interleave_self_int(num_neg, None, None);
    let mut w_data = v_data.zeros_like();

    let v = try_tensor_to_slice::<i64>(&v_data)?;
    let w = try_tensor_to_slice_mut::<i64>(&mut w_data)?;

    let mask: Vec<_> = w.into_par_iter().enumerate().map_init(
        random::rng_get,
        |rng, (i, w)| {
            for _t in 0..try_count {
                *w = rng.gen_range(0..node_count);
                if !graph.has_edge(v[i], *w) {
                    return None;
                }
            }

            Some(i)
        }
    ).filter_map(|v| v).collect();

    let output = Tensor::vstack(&[v_data, w_data]);
    Ok((output, mask))
}


pub fn negative_sample_neighbors_heterogenous(
    rng: &mut impl Rng,
    graphs: &HashMap<RelType, (CsrGraph, (i64, i64))>,
    node_rels: &HashMap<NodeType, Vec<RelType>>,
    inputs: &HashMap<NodeType, &[NodeIdx]>,
    num_neg: i64,
    try_count: i64,
) -> TensorResult<HashMap<RelType, CooGraphBuilder>> {

    let mut result: HashMap<RelType, CooGraphBuilder> = graphs.keys()
        .map(|k| (k.clone(), CooGraphBuilder::new()))
        .collect();

    for (node_type, inputs) in inputs{
        let node_rels = &node_rels[node_type];

        for v in inputs.iter().cloned() {
            for _ in 0..num_neg {
                let rel_type = &node_rels[rng.gen_range(0..node_rels.len())];
                let (graph, size) = &graphs[rel_type];
                let result = result.get_mut(rel_type).unwrap();

                for _t in 0..try_count {
                    let w = rng.gen_range(0..size.1);
                    if !graph.has_edge(v, w) {
                        result.push_edge(v, w, -1);
                        break;
                    }
                }
            }
        }
    }

    Ok(result)
}

// TODO: benchmark current impl vs per thread retry

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::convert::TryFrom;
    use rand::SeedableRng;
    use tch::Tensor;
    use crate::algo::negative_sampling::negative_sample_neighbors_heterogenous;
    use crate::data::{CscGraph, CscGraphData, CsrGraph, CsrGraphData, load_fake_hetero_graph, Size};
    use crate::data::load_karate_graph;
    use crate::utils::{EdgeType, NodeIdx, NodeType, RelType, try_tensor_to_slice};

    #[test]
    pub fn test_negative_sample_neighbors() {
        let (x, _, coo_graph) = load_karate_graph();

        let node_count = x.size()[0];
        let graph_data = CsrGraphData::try_from(&coo_graph).unwrap();
        let graph = CsrGraph::<i64, i64>::try_from(&graph_data).unwrap();

        let test: Vec<_> = (0..node_count).collect();
        let inputs = Tensor::of_slice(&test);
        let (output, mask) = super::negative_sample_neighbors(
            &graph,
            (node_count, node_count),
            &inputs,
            10,
            5,
        ).unwrap();
        let output_size = output.size()[1] as usize;

        let data = try_tensor_to_slice::<i64>(&output).unwrap();
        let v = &data[0..output_size];
        let w = &data[output_size..];

        let _error_count = mask.len();
        for (i, (v, w)) in v.iter().zip(w.iter()).enumerate() {
            if !mask.contains(&i) {
                assert!(!graph.has_edge(*v, *w));
            }
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

        let graph_data: HashMap<RelType, CsrGraphData> = coo_graphs.iter().map(|((src, rel, dst), coo_graph)| {
            let graph_data = CsrGraphData::try_from(coo_graph).unwrap();
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

        let result = negative_sample_neighbors_heterogenous(
            &mut rng,
            &graphs,
            &node_rels,
            &inputs,
            3,
            10,
        ).unwrap();

        for (rel_type, coo_builder) in result {
            let (graph, _) = &graphs[&rel_type];

            for i in coo_builder.rows.iter().zip(coo_builder.cols.iter()) {
                let (v, w) = i;
                assert!(!graph.has_edge(*v, *w));
            }
        }

    }
}