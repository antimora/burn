use crate::kernel::index::slice;
use crate::ops::numeric::{empty_device_dtype, zeros};
use crate::{CubeRuntime, tensor::CubeTensor};
use burn_backend::{DType, TensorMetadata};
use burn_std::{Shape, Slice};
use cubecl::prelude::*;
use cubek::fft::{irfft_launch, rfft_launch};

// Materializes a padded tensor (allocate + copy) because rfft_launch/irfft_launch
// in the external cubek crate don't support virtual padding via a length parameter.
// See: https://github.com/tracel-ai/cubek/issues/194
fn pad_to_length<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    target: usize,
) -> CubeTensor<R> {
    let shape = tensor.shape();
    let current = shape[dim];
    if current == target {
        return tensor;
    }
    if current > target {
        let ranges: Vec<_> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == dim { 0..target } else { 0..s })
            .collect();
        return slice(tensor, &ranges);
    }
    let mut padded_shape = shape.clone();
    padded_shape[dim] = target;
    let padded = zeros::<R>(tensor.device.clone(), padded_shape, tensor.dtype);
    let slices: Vec<Slice> = shape
        .iter()
        .enumerate()
        .map(|(i, &s)| Slice::from(if i == dim { 0..s } else { 0..s }))
        .collect();
    crate::kernel::index::slice_assign::<R>(padded, &slices, tensor)
}

/// Launch the rfft kernel with optional padding for non-power-of-two sizes.
pub fn rfft<R: CubeRuntime>(
    signal: CubeTensor<R>,
    dim: usize,
    n: Option<usize>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let dtype = match signal.dtype {
        DType::F64 => f64::as_type_native_unchecked().storage_type(),
        DType::F32 => f32::as_type_native_unchecked().storage_type(),
        _ => panic!("Unsupported type {:?}", signal.dtype),
    };

    let requested_n = n.unwrap_or(signal.shape()[dim]);
    let fft_size = requested_n.next_power_of_two();

    let signal = pad_to_length(signal, dim, fft_size);

    let signal_shape = signal.shape();
    let mut output_shape = signal_shape.clone();
    output_shape[dim] = fft_size / 2 + 1;

    let output_re = empty_device_dtype(
        signal.client.clone(),
        signal.device.clone(),
        output_shape.clone(),
        signal.dtype,
    );
    let output_im = empty_device_dtype(
        signal.client.clone(),
        signal.device.clone(),
        output_shape.clone(),
        signal.dtype,
    );

    rfft_launch(
        &signal.client.clone(),
        signal.binding(),
        output_re.clone().binding(),
        output_im.clone().binding(),
        dim,
        dtype,
    )
    .expect("rfft kernel launch failed");

    let n_out = requested_n / 2 + 1;
    let fft_out = fft_size / 2 + 1;
    if fft_out > n_out {
        (
            pad_to_length(output_re, dim, n_out),
            pad_to_length(output_im, dim, n_out),
        )
    } else {
        (output_re, output_im)
    }
}

/// Launch the irfft kernel with optional padding for non-power-of-two sizes.
pub fn irfft<R: CubeRuntime>(
    spectrum_re: CubeTensor<R>,
    spectrum_im: CubeTensor<R>,
    dim: usize,
    n: Option<usize>,
) -> CubeTensor<R> {
    assert!(
        spectrum_re.shape() == spectrum_im.shape(),
        "irfft: spectrum_re and spectrum_im shapes must match"
    );

    let dtype = match spectrum_re.dtype {
        DType::F64 => f64::as_type_native_unchecked().storage_type(),
        DType::F32 => f32::as_type_native_unchecked().storage_type(),
        _ => panic!("Unsupported type {:?}", spectrum_re.dtype),
    };

    let requested_n = n.unwrap_or((spectrum_re.shape()[dim] - 1) * 2);
    if requested_n == 0 {
        return spectrum_re;
    }
    let fft_size = requested_n.next_power_of_two();
    let half_fft = fft_size / 2 + 1;

    let spectrum_re = pad_to_length(spectrum_re, dim, half_fft);
    let spectrum_im = pad_to_length(spectrum_im, dim, half_fft);

    let mut signal_shape = spectrum_re.shape().clone();
    signal_shape[dim] = fft_size;

    let signal = empty_device_dtype(
        spectrum_re.client.clone(),
        spectrum_re.device.clone(),
        signal_shape,
        spectrum_re.dtype,
    );

    irfft_launch(
        &spectrum_re.client.clone(),
        spectrum_re.binding(),
        spectrum_im.binding(),
        signal.clone().binding(),
        dim,
        dtype,
    )
    .expect("irfft kernel launch failed");

    if fft_size > requested_n {
        pad_to_length(signal, dim, requested_n)
    } else {
        signal
    }
}
