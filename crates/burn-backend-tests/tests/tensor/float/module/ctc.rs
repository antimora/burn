use super::*;
use burn_tensor::Tolerance;
use burn_tensor::activation::log_softmax;
use burn_tensor::module::ctc_loss;

#[test]
fn test_ctc_loss_uniform() {
    // T=3, N=1, C=2, blank=0, target=[1, 1], uniform P = 1/2.
    // Only valid path is (1, 0, 1) with prob (1/2)^3.
    // Loss = -ln(1/8) = 3 * ln(2) ~ 2.0794
    let log_probs = TestTensor::<3>::full([3, 1, 2], (0.5f32).ln(), &Default::default());
    let targets = TestTensorInt::<2>::from([[1, 1]]);
    let input_lengths = TestTensorInt::<1>::from([3]);
    let target_lengths = TestTensorInt::<1>::from([2]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);

    let expected = burn_tensor::TensorData::from([3.0f32 * 2.0f32.ln()]);
    loss.into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}

#[test]
fn test_ctc_loss_matches_pytorch() {
    // T=5, N=3, C=4, deterministic logits via sin((t*7 + n*13 + c*3) * 0.1).
    // Expected per-sample losses computed by PyTorch's nn.functional.ctc_loss.
    let t_size: usize = 5;
    let n_size: usize = 3;
    let c_size: usize = 4;

    let mut data = Vec::with_capacity(t_size * n_size * c_size);
    for t in 0..t_size {
        for n in 0..n_size {
            for c in 0..c_size {
                data.push(((t * 7 + n * 13 + c * 3) as f32 * 0.1).sin());
            }
        }
    }
    let logits =
        TestTensor::<3>::from(burn_tensor::TensorData::new(data, [t_size, n_size, c_size]));
    let log_probs = log_softmax(logits, 2);

    let targets = TestTensorInt::<2>::from([[1, 2, 0], [1, 0, 0], [3, 2, 1]]);
    let input_lengths = TestTensorInt::<1>::from([5, 5, 5]);
    let target_lengths = TestTensorInt::<1>::from([2, 1, 3]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);

    let expected = burn_tensor::TensorData::from([
        3.5236570835113525f32,
        3.495313882827759,
        4.262677192687988,
    ]);
    loss.into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}
