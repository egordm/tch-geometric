use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use tch::{Kind, Tensor};
use data::graph::CscGraph;
use crate::utils::sampling::{ReplacementSampling, Sampling};
use crate::utils::utils::{replacement_sampling_range, reservoir_sampling_range, TensorConversionError, try_tensor_to_slice};

mod utils;
mod data;
mod algo;

#[pyfunction]
fn sum_as_string(a: Tensor, b: Tensor) -> PyResult<Tensor> {
    Ok(a + &b)
}

type NodeType = String;
type RelType = String;
type EdgeType = (NodeType, RelType, NodeType);

type NodeIdx = i64;
type OutputNodeIdx = i64;
type EdgeIdx = i64;


fn sample_own<
    const REPLACE: bool,
    const DIRECTED: bool,
>(
    t_col_ptrs: Tensor,
    t_row_data: Tensor,
    input_node: Tensor,
    num_neighbors: Vec<i64>,
) -> PyResult<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
)> {
    let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

    // Initialize some data structures for the sampling process:
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut to_local_node: HashMap<NodeIdx, OutputNodeIdx> = HashMap::new();


    let colptr_data = try_tensor_to_slice::<i64>(&t_col_ptrs)
        .map_err(<TensorConversionError as Into<PyErr>>::into)?;
    let row_data = try_tensor_to_slice::<i64>(&t_row_data)
        .map_err(<TensorConversionError as Into<PyErr>>::into)?;

    let graph = CscGraph {
        col_ptrs: colptr_data,
        row_data: row_data
    };

    let input_node_data = try_tensor_to_slice::<i64>(&input_node)
        .map_err(<TensorConversionError as Into<PyErr>>::into)?;


    for i in 0..input_node_data.len() {
        let v = input_node_data[i];
        samples.push(v);
        to_local_node.insert(v, i as OutputNodeIdx);
    }

    let (mut rows, mut cols, mut edges) = (Vec::<NodeIdx>::new(), Vec::<NodeIdx>::new(), Vec::<NodeIdx>::new());

    let (mut begin, mut end) = (0, samples.len());
    for num_samples in num_neighbors {
        let mut tmp_sample = vec![0; num_samples as usize];

        for i in begin..end {
            let w = samples[i];
            let neighbors_range = graph.neighbors_range(w);
            let neighbors = graph.neighbors_slice(w);

            if neighbors.len() == 0 {
                continue
            }

            if (num_samples < 0) || (!REPLACE && (num_samples > neighbors.len() as i64)) {
                // If num_samples is negative,
                // or if we are not replacing and num_samples is greater than the number of edges,
                // then we sample all the edges

                for (v, offset) in neighbors.iter().cloned().zip(neighbors_range.into_iter()) {
                    // register node in output list
                    let res = to_local_node.insert(v, samples.len() as OutputNodeIdx);

                    samples.push(v); // add neighbor to output list
                    if DIRECTED {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset as NodeIdx);
                    }
                }
            } else {
                // Randomly sample num_samples nodes from the neighbors of w
                if REPLACE {
                    neighbors_range.replacement_sample(&mut rng, &mut tmp_sample);
                } else {
                    neighbors_range.sample(&mut rng, &mut tmp_sample);
                }

                for (v, offset) in tmp_sample.iter().cloned().map(|o| (graph.row_data[o], o)) {
                    let res = to_local_node.insert(v, samples.len() as OutputNodeIdx); // register node in output list

                    samples.push(v); // add neighbor to output list
                    if DIRECTED {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset as NodeIdx);
                    }
                }
            }
        }
        // set begin to the old end to avoid sampling same nodes twice. and repeat
        begin = end;
        end = samples.len();
    }

    if !DIRECTED {
        // if undirected, we need to add edges in both directions
        for (i, w) in samples.iter().enumerate() {
            // for each sample
            let neighbors_range = graph.neighbors_range(*w);
            let neighbors_range = neighbors_range.start as NodeIdx..neighbors_range.end as NodeIdx;
            let neighbors = graph.neighbors_slice(*w);

            for (v, offset) in neighbors.iter().zip(neighbors_range) {
                let res = to_local_node.get(&v); // find neighbor in output list
                if let Some(res) = res.cloned() {
                    // if neighbor is in output list
                    cols.push(i as NodeIdx); // add sample to row
                    rows.push(res); // add edge to output graph
                    edges.push(offset);
                }
            }
        }
    }

    return Ok((
        samples.try_into().expect("Cant convert vec into tensor"),
        rows.try_into().expect("Cant convert vec into tensor"),
        cols.try_into().expect("Cant convert vec into tensor"),
        edges.try_into().expect("Cant convert vec into tensor"),
    ));
}

#[pyfunction]
fn sample_own_custom(
    colptr: Tensor,
    row: Tensor,
    input_node: Tensor,
    num_neighbors: Vec<i64>,
    replace: bool,
    directed: bool,
) -> PyResult<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
)> {
    match (replace, directed) {
        (true, true) => sample_own::<true, true>(colptr, row, input_node, num_neighbors),
        (true, false) => sample_own::<true, false>(colptr, row, input_node, num_neighbors),
        (false, true) => sample_own::<false, true>(colptr, row, input_node, num_neighbors),
        (false, false) => sample_own::<false, false>(colptr, row, input_node, num_neighbors),
    }
}

#[pyfunction]
fn sample(
    colptr: Tensor,
    row: Tensor,
    input_node: Tensor,
    num_neighbors: Vec<i64>,
    replace: bool,
    directed: bool,
) -> PyResult<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
)> {
    let mut rng = rand::rngs::SmallRng::from_seed([0; 32]);

    // Initialize some data structures for the sampling process:
    let mut samples: Vec<NodeIdx> = Vec::new();
    let mut to_local_node: HashMap<NodeIdx, OutputNodeIdx> = HashMap::new();

    let colptr_data = if colptr.kind() == Kind::Int64 {
        unsafe { std::slice::from_raw_parts(colptr.data_ptr() as *const NodeIdx, colptr.size()[0] as usize) }
    } else {
        return Err(PyTypeError::new_err("colptr must be int64"));
    };
    let row_data = if row.kind() == Kind::Int64 {
        unsafe { std::slice::from_raw_parts(row.data_ptr() as *const NodeIdx, row.size()[0] as usize) }
    } else {
        return Err(PyTypeError::new_err("row must be int64"));
    };
    let input_node_data = if input_node.kind() == Kind::Int64 {
        unsafe { std::slice::from_raw_parts(input_node.data_ptr() as *const NodeIdx, input_node.size()[0] as usize) }
    } else {
        return Err(PyTypeError::new_err("input_node must be int64"));
    };

    for i in 0..input_node_data.len() {
        let v = input_node_data[i];
        samples.push(v);
        to_local_node.insert(v, i as OutputNodeIdx);
    }

    let (mut rows, mut cols, mut edges) = (Vec::<NodeIdx>::new(), Vec::<NodeIdx>::new(), Vec::<NodeIdx>::new());

    let (mut begin, mut end) = (0, samples.len());
    for ell in 0..num_neighbors.len() {
        // foreach layer, sample num_samples neighbors
        let num_samples = num_neighbors[ell];

        for i in begin..end {
            // foreach node in layer (in samples stack), sample num_samples neighbors
            let w = samples[i]; // node id
            let wu = w as usize;
            let col_start = colptr_data[wu]; // Col index value
            let col_end = colptr_data[(wu + 1)]; // Next Col index value
            let col_count = col_end - col_start; // Amount of edges for given node

            if col_count == 0 {
                // If no edges, skip
                continue;
            }

            if (num_samples < 0) || (!replace && (num_samples > col_count)) {
                // If num_samples is negative,
                // or if we are not replacing and num_samples is greater than the number of edges,
                // then we sample all the edges

                for offset in col_start as usize..col_end as usize {
                    // for each neighbor of sample

                    let v = row_data[offset]; // neighbor node id
                    let res = to_local_node.insert(v, samples.len() as OutputNodeIdx); // register node in output list

                    samples.push(v); // add neighbor to output list
                    if directed {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset as EdgeIdx);
                    }
                }
            } else if replace {
                // sample neighbors with replacement

                for j in 0..num_samples {
                    let offset = col_start + rng.gen_range(0..col_count); // random neighbor offset in matrix
                    let v = row_data[offset as usize]; // neighbor node id
                    let res = to_local_node.insert(v, samples.len() as OutputNodeIdx); // register node in output list

                    samples.push(v); // add neighbor to output list
                    if directed {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset);
                    }
                }
            } else {
                // sample neighbors without replacement

                let mut rnd_indices = HashSet::new();
                for j in col_count - num_samples..col_count {
                    // start at first col_count - num_samples and go to last col_count
                    // (adding an additional possible node each time)
                    let mut rnd = rng.gen_range(0..j);
                    if !rnd_indices.insert(rnd) {
                        // if it was already picked, then instead add j
                        rnd = j;
                        rnd_indices.insert(j);
                    }

                    let offset = col_start + rnd; // random neighbor offset in matrix
                    let v = row_data[offset as usize]; // neighbor node id
                    let res = to_local_node.insert(v, samples.len() as OutputNodeIdx); // register node in output list

                    samples.push(v); // add neighbor to output list
                    if directed {
                        // if directed, add edge to output graph (because it matters)
                        cols.push(i as NodeIdx);
                        rows.push(res.unwrap());
                        edges.push(offset);
                    }
                }
            }
        }
        // set begin to the old end to avoid sampling same nodes twice. and repeat
        begin = end;
        end = samples.len();
    }

    if !directed {
        // if undirected, we need to add edges in both directions
        for i in 0..samples.len() as OutputNodeIdx {
            // for each sample

            let w = samples[i as usize]; // sample node id
            let col_start = colptr_data[w as usize]; // Col index value
            let col_end = colptr_data[(w + 1) as usize]; // Next Col index value

            for offset in col_start..col_end {
                // for each neighbor of sample

                let v = row_data[offset as usize]; // neighbor node id
                let res = to_local_node.get(&v); // find neighbor in output list
                if let Some(res) = res.cloned() {
                    // if neighbor is in output list
                    cols.push(i); // add sample to row
                    rows.push(res); // add edge to output graph
                    edges.push(offset);
                }
            }
        }
    }


    return Ok((
        samples.try_into().expect("Cant convert vec into tensor"),
        rows.try_into().expect("Cant convert vec into tensor"),
        cols.try_into().expect("Cant convert vec into tensor"),
        edges.try_into().expect("Cant convert vec into tensor"),
    ));
}


/*#[pyfunction]
fn hetero_sample<'a>(
    node_types: Vec<&'a NodeType>,
    edge_types: Vec<&'a EdgeType>,
    colptr_dict: HashMap<&'a RelType, Tensor>,
    row_dict: HashMap<&'a RelType, Tensor>,
    input_node_dict: HashMap<&'a NodeType, Tensor>,
    num_neighbors_dict: HashMap<&'a RelType, Vec<i64>>,
    num_hops: i64,
) -> (
    HashMap<&'a NodeType, Tensor>,
    HashMap<&'a RelType, Tensor>,
    HashMap<&'a RelType, Tensor>,
    HashMap<&'a RelType, Tensor>,
) {

    unimplemented!()
}*/


#[pymodule]
fn tch_geometric(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    // m.add_function(wrap_pyfunction!(hetero_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_own_custom, m)?)?;
    Ok(())
}