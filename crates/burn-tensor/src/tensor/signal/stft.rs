use alloc::vec;
use core::ops::Range;

use burn_backend::Backend;

use crate::Tensor;
use crate::ops::PadMode;

use super::{irfft, rfft};

/// Computes the Short-Time Fourier Transform (STFT).
///
/// Splits the signal into overlapping windowed frames and computes the FFT on each.
///
/// # Arguments
///
/// * `signal` - Real-valued input tensor of shape `[batch, signal_length]`.
/// * `n_fft` - Size of each FFT frame (must be >= 1).
/// * `hop_length` - Stride between successive frames (must be >= 1).
/// * `win_length` - Window length. If `Some(w)`, the window is center-padded to `n_fft`.
///   Defaults to `n_fft`.
/// * `window` - Optional 1-D window tensor. Defaults to a rectangular (all-ones) window.
/// * `center` - If `true`, the signal is reflect-padded by `n_fft / 2` on both sides before
///   framing.
/// * `onesided` - If `true` (typical for real input), returns only the first `n_fft/2 + 1`
///   frequency bins. If `false`, returns all `n_fft` bins.
///
/// # Returns
///
/// A tensor of shape `[batch, n_frames, n_freqs, 2]` where the last dimension holds
/// `[real, imaginary]`.
///
/// # Panics
///
/// * If `n_fft == 0` or `hop_length == 0`.
/// * If `win_length > n_fft`.
/// * If `window` length != `win_length` (or `n_fft` when `win_length` is `None`).
pub fn stft<B: Backend>(
    signal: Tensor<B, 2>,
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    window: Option<Tensor<B, 1>>,
    center: bool,
    onesided: bool,
) -> Tensor<B, 4> {
    assert!(n_fft >= 1, "n_fft must be >= 1, got {n_fft}");
    assert!(
        hop_length >= 1,
        "hop_length must be >= 1, got {hop_length}"
    );

    let win_len = win_length.unwrap_or(n_fft);
    assert!(
        win_len <= n_fft,
        "win_length ({win_len}) must be <= n_fft ({n_fft})"
    );

    let device = signal.device();

    let window = match window {
        Some(w) => {
            assert_eq!(
                w.dims()[0],
                win_len,
                "window length ({}) must match win_length ({win_len})",
                w.dims()[0],
            );
            w
        }
        None => Tensor::ones([win_len], &device),
    };

    let window = pad_window_to_n_fft(window, win_len, n_fft);

    let signal = if center {
        let pad_amount = n_fft / 2;
        signal.pad([(0, 0), (pad_amount, pad_amount)], PadMode::Reflect)
    } else {
        signal
    };

    let [batch, sig_len] = signal.dims();
    assert!(
        sig_len >= n_fft,
        "signal length ({sig_len}) must be >= n_fft ({n_fft}) after padding"
    );

    let n_frames = 1 + (sig_len - n_fft) / hop_length;

    // Extract overlapping frames: [batch, n_frames, n_fft]
    let frames: Tensor<B, 3> = signal.unfold(1, n_fft, hop_length);

    // Apply window (broadcast over batch and n_frames)
    let window_3d: Tensor<B, 3> = window.reshape([1, 1, n_fft]);
    let windowed = frames.mul(window_3d);

    // Flatten to [batch * n_frames, n_fft] for rfft_n
    let flat: Tensor<B, 2> = windowed.reshape([batch * n_frames, n_fft]);

    let (re, im) = rfft(flat, 1, Some(n_fft));

    let n_freqs_onesided = n_fft / 2 + 1;
    let (re, im, n_freqs) = if onesided {
        (re, im, n_freqs_onesided)
    } else {
        let (re_full, im_full) = reconstruct_full_spectrum(re, im, n_fft);
        (re_full, im_full, n_fft)
    };

    // Reshape to [batch, n_frames, n_freqs] then stack real/imag
    let re: Tensor<B, 3> = re.reshape([batch, n_frames, n_freqs]);
    let im: Tensor<B, 3> = im.reshape([batch, n_frames, n_freqs]);

    Tensor::stack::<4>(vec![re, im], 3)
}

/// Reconstruct the full N-point spectrum from the onesided rfft output using Hermitian symmetry.
fn reconstruct_full_spectrum<B: Backend>(
    re: Tensor<B, 2>,
    im: Tensor<B, 2>,
    n_fft: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    if n_fft <= 1 {
        return (re, im);
    }

    let n_freqs_onesided = n_fft / 2 + 1;
    let n_neg = n_fft - n_freqs_onesided;
    if n_neg == 0 {
        return (re, im);
    }

    // Negative frequencies = conjugate-reversed copies of bins 1..n_fft/2
    let re_neg = re.clone().narrow(1, 1, n_neg).flip([1]);
    let im_neg = im.clone().narrow(1, 1, n_neg).flip([1]).neg();

    (
        Tensor::cat(vec![re, re_neg], 1),
        Tensor::cat(vec![im, im_neg], 1),
    )
}

/// Center-pad window from `win_len` to `n_fft` with zeros.
fn pad_window_to_n_fft<B: Backend>(
    window: Tensor<B, 1>,
    win_len: usize,
    n_fft: usize,
) -> Tensor<B, 1> {
    if win_len < n_fft {
        let pad_left = (n_fft - win_len) / 2;
        let pad_right = n_fft - win_len - pad_left;
        window.pad([(pad_left, pad_right)], PadMode::Constant(0.0))
    } else {
        window
    }
}

/// Computes the inverse Short-Time Fourier Transform (ISTFT).
///
/// Reconstructs a time-domain signal from its STFT representation using overlap-add.
///
/// # Arguments
///
/// * `stft_matrix` - Complex STFT tensor of shape `[batch, n_frames, n_freqs, 2]`.
/// * `n_fft` - FFT size used in the forward STFT.
/// * `hop_length` - Hop length used in the forward STFT.
/// * `win_length` - Window length used in the forward STFT. Defaults to `n_fft`.
/// * `window` - Window tensor used in the forward STFT. Defaults to rectangular.
/// * `center` - Whether centering was used in the forward STFT.
/// * `onesided` - Whether the STFT is onesided.
/// * `length` - Optional output signal length. If `None`, the length is inferred.
///
/// # Returns
///
/// A real-valued tensor of shape `[batch, signal_length]`.
pub fn istft<B: Backend>(
    stft_matrix: Tensor<B, 4>,
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    window: Option<Tensor<B, 1>>,
    center: bool,
    onesided: bool,
    length: Option<usize>,
) -> Tensor<B, 2> {
    let [batch, n_frames, _n_freqs, two] = stft_matrix.dims();
    assert_eq!(two, 2, "last dimension of stft_matrix must be 2 (real, imag)");

    let win_len = win_length.unwrap_or(n_fft);
    let device = stft_matrix.device();

    let window = match window {
        Some(w) => w,
        None => Tensor::ones([win_len], &device),
    };
    let window = pad_window_to_n_fft(window, win_len, n_fft);

    // Split real and imaginary: each [batch, n_frames, n_freqs]
    let re: Tensor<B, 3> = stft_matrix.clone().narrow(3, 0, 1).squeeze_dim(3);
    let im: Tensor<B, 3> = stft_matrix.narrow(3, 1, 1).squeeze_dim(3);

    let n_freqs = re.dims()[2];
    let re: Tensor<B, 2> = re.reshape([batch * n_frames, n_freqs]);
    let im: Tensor<B, 2> = im.reshape([batch * n_frames, n_freqs]);

    // Extract onesided spectrum for irfft
    let (re_half, im_half) = if onesided {
        (re, im)
    } else {
        let half = n_fft / 2 + 1;
        (re.narrow(1, 0, half), im.narrow(1, 0, half))
    };

    let frames = irfft(re_half, im_half, 1, Some(n_fft));

    // Reshape to [batch, n_frames, n_fft]
    let frames: Tensor<B, 3> = frames.reshape([batch, n_frames, n_fft]);

    // Apply window to each frame
    let window_3d: Tensor<B, 3> = window.clone().reshape([1, 1, n_fft]);
    let windowed = frames.mul(window_3d.clone());

    // Overlap-add
    let expected_len = n_fft + (n_frames - 1) * hop_length;
    let mut output = Tensor::<B, 2>::zeros([batch, expected_len], &device);
    let mut window_sum = Tensor::<B, 2>::zeros([batch, expected_len], &device);

    let window_sq: Tensor<B, 3> = window_3d.clone().mul(window_3d.clone());
    let window_sq_batch: Tensor<B, 3> =
        window_sq.expand([batch as i64, 1, n_fft as i64]);

    for f in 0..n_frames {
        let start = f * hop_length;
        let frame: Tensor<B, 2> = windowed.clone().narrow(1, f, 1).squeeze_dim(1);
        let win_frame: Tensor<B, 2> = window_sq_batch.clone().narrow(1, 0, 1).squeeze_dim(1);

        // Overlap-ADD: read current slice, add frame, write back
        let ranges: [Range<usize>; 2] = [0..batch, start..start + n_fft];
        let current = output.clone().slice(ranges.clone());
        output = output.clone().slice_assign(ranges.clone(), current.add(frame));
        let current_win = window_sum.clone().slice(ranges.clone());
        window_sum = window_sum.clone().slice_assign(ranges, current_win.add(win_frame));
    }

    // Normalize by window sum (avoid division by zero)
    let window_sum = window_sum.add_scalar(1e-10);
    output = output.div(window_sum);

    // Remove center padding
    let output = if center {
        let pad_amount = n_fft / 2;
        let trimmed_len = expected_len - 2 * pad_amount;
        output.narrow(1, pad_amount, trimmed_len)
    } else {
        output
    };

    match length {
        Some(len) => {
            let current_len = output.dims()[1];
            if len <= current_len {
                output.narrow(1, 0, len)
            } else {
                output.pad([(0, 0), (0, len - current_len)], PadMode::Constant(0.0))
            }
        }
        None => output,
    }
}
