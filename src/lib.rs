use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use tch::{Kind, Tensor};
use data::graph::SparseGraph;
use crate::utils::sampling::{ReplacementSampling, Sampling};
use crate::utils::tensor::{TensorConversionError, try_tensor_to_slice};

mod utils;
mod data;
mod algo;
#[cfg(feature = "extension-module")]
mod python;