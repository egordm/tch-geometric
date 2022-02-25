

#[cfg(test)]
pub mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use tch::Tensor;

    pub fn load_karate_graph() -> (Tensor, Tensor, Tensor) {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/karate.npz");
        let data: HashMap<_, _> = Tensor::read_npz(&d).unwrap().into_iter().collect();
        let x = data["x"].shallow_clone();
        let y = data["y"].shallow_clone();
        let edge_index = data["edge_index"].shallow_clone();

        (x, y, edge_index)
    }
}