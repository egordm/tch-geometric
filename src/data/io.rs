use std::collections::HashMap;
use std::path::PathBuf;
use tch::Tensor;
use crate::data::CooGraphStorage;
use crate::utils::{EdgeType, NodeType};

pub fn load_karate_graph() -> (Tensor, Tensor, CooGraphStorage) {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/karate.npz");
    let data: HashMap<_, _> = Tensor::read_npz(&d).unwrap().into_iter().collect();
    let x = data["x"].shallow_clone();
    let y = data["y"].shallow_clone();
    let edge_index = data["edge_index"].shallow_clone();

    let coo_graph_data = CooGraphStorage::new(
        edge_index, (x.size()[0], x.size()[0]),
    );

    (x, y, coo_graph_data)
}

pub fn load_fake_hetero_graph() -> (HashMap<NodeType, Tensor>, HashMap<EdgeType, CooGraphStorage>) {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fakeheterodataset.npz");
    let data: HashMap<_, _> = Tensor::read_npz(&d).unwrap().into_iter().collect();

    let mut xs: HashMap<NodeType, Tensor> = HashMap::new();
    let mut edge_indexes: HashMap<EdgeType, Tensor> = HashMap::new();

    for (k, v) in data.into_iter() {
        if k.starts_with("node_") {
            let tokens: Vec<_> = k.split('_').collect();

            if tokens[2] == "x" {
                let node_type: NodeType = tokens[1].to_string();
                xs.insert(node_type, v);
            }
        } else if k.starts_with("edge_") {
            let tokens: Vec<_> = k.split('_').collect();
            assert_eq!(tokens[2], "edge");
            assert_eq!(tokens[3], "index");
            let edge_type_tokens: Vec<_> = tokens[1].split('-').collect();
            let edge_type: EdgeType = (
                edge_type_tokens[0].to_string(),
                edge_type_tokens[1].to_string(),
                edge_type_tokens[2].to_string(),
            );

            edge_indexes.insert(edge_type, v);
        } else {
            panic!("Unknown key: {}", k);
        }
    }

    let mut coo_graph_data = HashMap::new();
    for ((src, rel, dst), edge_index) in edge_indexes {
        let coo_graph_data_ = CooGraphStorage::new(
            edge_index, (xs[&src].size()[0], xs[&dst].size()[0]),
        );
        coo_graph_data.insert((src, rel, dst), coo_graph_data_);
    }

    (
        xs,
        coo_graph_data,
    )
}
