use alloc::vec;

use burn_backend::Backend;

use crate::Tensor;
use crate::ops::PadMode;

use super::{irfft, rfft};

/// Configuration shared by [`stft`] and [`istft`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StftOptions {
    /// Size of each FFT frame (must be >= 1).
    pub n_fft: usize,
    /// Stride between successive frames (must be >= 1).
    pub hop_length: usize,
    /// Window length. If `Some(w)`, the window is center-padded to `n_fft`. Defaults to `n_fft`.
    pub win_length: Option<usize>,
    /// If `true`, the signal is reflect-padded by `n_fft / 2` on both sides before framing.
    pub center: bool,
    /// If `true` (typical for real input), uses only the first `n_fft/2 + 1` frequency bins.
    pub onesided: bool,
}

impl StftOptions {
    fn effective_win_length(&self) -> usize {
        self.win_length.unwrap_or(self.n_fft)
    }
}

/// Computes the Short-Time Fourier Transform (STFT).
///
/// Splits the signal into overlapping windowed frames and computes the FFT on each.
///
/// # Returns
///
/// A tensor of shape `[batch, n_frames, n_freqs, 2]` where the last dimension holds
/// `[real, imaginary]`.
pub fn stft<B: Backend>(
    signal: Tensor<B, 2>,
    window: Option<Tensor<B, 1>>,
    options: StftOptions,
) -> Tensor<B, 4> {
    let n_fft = options.n_fft;
    let hop_length = options.hop_length;
    let center = options.center;
    let onesided = options.onesided;
    assert!(n_fft >= 1, "n_fft must be >= 1, got {n_fft}");
    assert!(hop_length >= 1, "hop_length must be >= 1, got {hop_length}");

    let win_len = options.effective_win_length();
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

    // Flatten to [batch * n_frames, n_fft] for rfft
    let flat: Tensor<B, 2> = windowed.reshape([batch * n_frames, n_fft]);

    let (re, im) = rfft(flat, 1, Some(n_fft));

    let n_freqs_actual = re.dims()[1];
    let (re, im, n_freqs) = if onesided {
        (re, im, n_freqs_actual)
    } else {
        let half = n_fft / 2 + 1;
        let re = if n_freqs_actual > half {
            re.narrow(1, 0, half)
        } else {
            re
        };
        let im = if n_freqs_actual > half {
            im.narrow(1, 0, half)
        } else {
            im
        };
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
/// * `window` - Window tensor used in the forward STFT. Defaults to rectangular.
/// * `length` - Optional output signal length. If `None`, the length is inferred.
/// * `options` - STFT configuration (must match the forward STFT).
///
/// # Returns
///
/// A real-valued tensor of shape `[batch, signal_length]`.
pub fn istft<B: Backend>(
    stft_matrix: Tensor<B, 4>,
    window: Option<Tensor<B, 1>>,
    length: Option<usize>,
    options: StftOptions,
) -> Tensor<B, 2> {
    let n_fft = options.n_fft;
    let hop_length = options.hop_length;
    let center = options.center;
    let onesided = options.onesided;
    assert!(n_fft >= 1, "n_fft must be >= 1, got {n_fft}");
    assert!(hop_length >= 1, "hop_length must be >= 1, got {hop_length}");
    let [batch, n_frames, _n_freqs, two] = stft_matrix.dims();
    assert_eq!(
        two, 2,
        "last dimension of stft_matrix must be 2 (real, imag)"
    );

    let win_len = options.effective_win_length();
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
    let window_3d: Tensor<B, 3> = window.reshape([1, 1, n_fft]);
    let windowed = frames.mul(window_3d.clone());

    let expected_len = n_fft + (n_frames - 1) * hop_length;
    let mut output = Tensor::<B, 2>::zeros([batch, expected_len], &device);
    let mut window_sum = Tensor::<B, 2>::zeros([batch, expected_len], &device);

    let window_1d: Tensor<B, 1> = window_3d.clone().reshape([n_fft]);
    let window_sq_1d: Tensor<B, 1> = window_1d.clone().mul(window_1d);

    for f in 0..n_frames {
        let start = f * hop_length;
        let right_pad = expected_len - n_fft - start;
        let frame: Tensor<B, 2> = windowed.clone().narrow(1, f, 1).squeeze_dim(1);
        let frame_full = frame.pad([(0, 0), (start, right_pad)], PadMode::Constant(0.0));
        output = output.add(frame_full);

        let win_full: Tensor<B, 1> = window_sq_1d
            .clone()
            .pad([(start, right_pad)], PadMode::Constant(0.0));
        let win_full: Tensor<B, 2> = win_full.unsqueeze_dim(0);
        window_sum = window_sum.add(win_full);
    }

    let window_sum = window_sum.clamp_min(1e-10);
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
