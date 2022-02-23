use std::ops::{Add, Range, Sub};
use pyo3::exceptions::PyValueError;
use pyo3::{PyErr};
use rand::distributions::uniform::{SampleRange, SampleUniform};
use rand::Rng;
use thiserror::Error;
use tch::kind::Element;
use tch::{Device, Kind, Tensor};

#[derive(Error, Debug)]
pub enum TensorConversionError {
    #[error("Tensor must be on {0:?} device")]
    InvalidDevice(Device),
    #[error("Tensor must be a is of invalid type. Expected {0:?} but got {1:?}")]
    InvalidDType(Kind, Kind),
    #[error("Tensor must be of rank {0:?}")]
    InvalidShape(Option<String>),
}

impl Into<PyErr> for TensorConversionError {
    fn into(self) -> PyErr {
        PyValueError::new_err(format!("{}", &self))
    }
}

pub fn try_tensor_to_slice<T: Element>(tensor: &Tensor) -> Result<&[T], TensorConversionError> {
    if tensor.device() != Device::Cpu {
        return Err(TensorConversionError::InvalidDevice(Device::Cpu));
    }
    if tensor.kind() != T::KIND {
        return Err(TensorConversionError::InvalidDType(T::KIND, tensor.kind()));
    }

    let length = tensor.size().into_iter()
        .reduce(|acc, x| acc * x)
        .ok_or(TensorConversionError::InvalidShape(None))?;
    Ok(
        unsafe { std::slice::from_raw_parts(tensor.data_ptr() as *const T, length as usize) }
    )
}

pub fn reservoir_sampling<T: Copy>(
    rng: &mut impl Rng,
    src: &[T],
    dst: &mut [T],
) {
    dst.copy_from_slice(&src[0..dst.len()]);

    for i in dst.len()..src.len() {
        let j = rng.gen_range(0..i);
        if j < dst.len() {
            dst[j] = src[i];
        }
    }
}

pub fn reservoir_sampling_range<
    T: Copy + Add<Output=T> + Sub<Output = T> + From<usize> + Into<usize>
>(
    rng: &mut impl Rng,
    src: Range<T>,
    dst: &mut [T],
) {
    let src_len = (src.end - src.start).into();

    for i in 0..dst.len() {
        dst[i] = src.start + T::from(i);
    }

    for i in dst.len()..src_len {
        let j = rng.gen_range(0..i);
        if j < dst.len() {
            dst[j] = src.start + T::from(i);
        }
    }
}

pub fn replacement_sampling<T: Copy>(
    rng: &mut impl Rng,
    src: &[T],
    dst: &mut [T],
) {
    for i in 0..dst.len() {
        let j = rng.gen_range(0..src.len());
        dst[i] = src[j];
    }
}

pub fn replacement_sampling_range<
    T: SampleUniform,
    R: SampleRange<T> + Clone
>(
    rng: &mut impl Rng,
    src: R,
    dst: &mut [T],
) {
    for i in 0..dst.len() {
        dst[i] = rng.gen_range(src.clone());
    }
}