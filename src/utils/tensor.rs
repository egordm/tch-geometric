use std::ops::{Add, Range, Sub};
use pyo3::exceptions::PyValueError;
use pyo3::{PyErr};
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

pub type TensorResult<T> = Result<T, TensorConversionError>;

macro_rules! check_device {
    ($tensor:expr, $device:expr) => {
        if $tensor.device() != $device {
            return Err(TensorConversionError::InvalidDevice($device).into());
        }
    };
}

macro_rules! check_kind {
    ($tensor:expr, $kind:expr) => {
        if $tensor.kind() != $kind {
            return Err(TensorConversionError::InvalidDType($kind, $tensor.kind()).into());
        }
    };
}

pub(crate) use check_device;
pub(crate) use check_kind;

pub fn try_tensor_to_slice<T: Element>(tensor: &Tensor) -> Result<&[T], TensorConversionError> {
    check_device!(tensor, Device::Cpu);
    check_kind!(tensor, T::KIND);

    Ok(tensor_to_slice(tensor))
}

pub fn tensor_to_slice<T: Element>(tensor: &Tensor) -> &[T] {
    unsafe { std::slice::from_raw_parts(tensor.data_ptr() as *const T, tensor.numel()) }
}

pub fn try_tensor_to_slice_mut<T: Element>(tensor: &mut Tensor) -> Result<&mut [T], TensorConversionError> {
    check_device!(tensor, Device::Cpu);
    check_kind!(tensor, T::KIND);

    Ok(tensor_to_slice_mut(tensor))
}

pub fn tensor_to_slice_mut<T: Element>(tensor: &mut Tensor) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut T, tensor.numel()) }
}