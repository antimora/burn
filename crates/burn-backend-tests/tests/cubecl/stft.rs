use super::*;
use burn_tensor::signal::{StftOptions, hann_window, istft, stft};
use burn_tensor::Tolerance;

fn opts(n_fft: usize, hop_length: usize, center: bool, onesided: bool) -> StftOptions {
    StftOptions {
        n_fft,
        hop_length,
        win_length: None,
        center,
        onesided,
    }
}

#[test]
fn stft_constant_signal_rectangular_window() {
    // Constant signal with rectangular window: only DC bin should be non-zero
    let signal = TestTensor::<2>::from([[1.0, 1.0, 1.0, 1.0]]);
    let result = stft(signal, None, opts(4, 1, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 1); // (4 - 4) / 1 + 1 = 1
    assert_eq!(n_freqs, 3); // 4/2 + 1 = 3
    assert_eq!(two, 2);

    // DC bin should be 4.0 + 0i (sum of ones)
    let data = result.into_data();
    let values = data.to_vec::<f32>().unwrap();
    // [batch=0, frame=0, freq=0, re] = 4.0
    assert!((values[0] - 4.0).abs() < 1e-4, "DC real should be 4.0, got {}", values[0]);
    // [batch=0, frame=0, freq=0, im] = 0.0
    assert!(values[1].abs() < 1e-4, "DC imag should be 0.0, got {}", values[1]);
}

#[test]
fn stft_output_shape_onesided() {
    let signal = TestTensor::<2>::from([[1.0; 16]]);
    let result = stft(signal, None, opts(8, 4, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3); // (16 - 8) / 4 + 1 = 3
    assert_eq!(n_freqs, 5); // 8/2 + 1 = 5
    assert_eq!(two, 2);
}

#[test]
fn stft_output_shape_twosided() {
    let signal = TestTensor::<2>::from([[1.0; 16]]);
    let result = stft(signal, None, opts(8, 4, false, false));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3);
    assert_eq!(n_freqs, 8); // full spectrum
    assert_eq!(two, 2);
}

#[test]
fn stft_center_padding() {
    // With center=true, signal is padded by n_fft/2 on both sides
    let signal = TestTensor::<2>::from([[1.0; 8]]);
    let result = stft(signal, None, opts(4, 2, true, true));

    // After padding: 2 + 8 + 2 = 12 samples
    // n_frames = (12 - 4) / 2 + 1 = 5
    let [_, n_frames, _, _] = result.dims();
    assert_eq!(n_frames, 5);
}

#[test]
fn stft_with_hann_window() {
    let signal = TestTensor::<2>::from([[1.0; 8]]);
    let window: TestTensor<1> = hann_window(4, true, &Default::default());
    let result = stft(signal, Some(window), opts(4, 2, false, true));

    let [batch, n_frames, n_freqs, two] = result.dims();
    assert_eq!(batch, 1);
    assert_eq!(n_frames, 3);
    assert_eq!(n_freqs, 3);
    assert_eq!(two, 2);
}

#[test]
fn stft_batch_dimension() {
    let signal = TestTensor::<2>::from([[1.0; 8], [2.0; 8]]);
    let result = stft(signal, None, opts(4, 2, false, true));

    let [batch, _, _, _] = result.dims();
    assert_eq!(batch, 2);
}

#[test]
fn stft_istft_roundtrip_rectangular() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, false, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_centered() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, true, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_hann_window() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 1;

    let window: TestTensor<1> = hann_window(4, true, &Default::default());
    let o = opts(n_fft, hop_length, true, true);

    // center=true is required with Hann windows to avoid edge effects from window zeros
    let spectrum = stft(original.clone(), Some(window.clone()), o);
    let reconstructed = istft(spectrum, Some(window), Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-2));
}

#[test]
fn stft_istft_roundtrip_twosided() {
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, false, false);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_batch() {
    let original = TestTensor::<2>::from([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ]);
    let n_fft = 4;
    let hop_length = 2;

    let o = opts(n_fft, hop_length, false, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(8), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}

#[test]
fn stft_istft_roundtrip_non_power_of_two_nfft() {
    // n_fft=5 exercises non-power-of-two virtual padding path
    let original = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]);
    let n_fft = 5;
    let hop_length = 1;

    let o = opts(n_fft, hop_length, false, true);
    let spectrum = stft(original.clone(), None, o);
    let reconstructed = istft(spectrum, None, Some(10), o);

    reconstructed
        .into_data()
        .assert_approx_eq::<FloatElem>(&original.into_data(), Tolerance::absolute(1e-3));
}
