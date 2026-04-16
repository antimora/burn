use super::*;
use burn_tensor::Tolerance;
use burn_tensor::activation::log_softmax;
use burn_tensor::module::ctc_loss;

#[test]
fn test_ctc_loss_unreachable_target_is_inf() {
    // T=2, N=1, C=3, target=[1, 1]. CTC requires at least `target_length
    // + repeats = 3` steps to produce this target, but input_length is 2,
    // so no valid alignment exists and the loss must be +inf. burn-nn's
    // `zero_infinity` relies on `is_inf` to detect this case, so fused
    // backend kernels must preserve the +inf (not collapse to a large
    // finite number).
    let device = Default::default();
    let log_probs = TestTensor::<3>::full([2, 1, 3], (1.0f32 / 3.0).ln(), &device);
    let targets = TestTensorInt::<2>::from([[1_i64, 1]]);
    let input_lengths = TestTensorInt::<1>::from([2_i64]);
    let target_lengths = TestTensorInt::<1>::from([2_i64]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);
    let loss_data = loss.into_data();
    let value = loss_data.iter::<f32>().next().unwrap();
    assert!(
        value.is_infinite() && value > 0.0,
        "Expected +inf for an unreachable target, got {value}"
    );
}

/// Regression for NaN gradient contamination on unreachable targets via the
/// default `ctc_loss_backward` path (the one used by burn-cubecl's fused
/// kernel via `ctc_grad_from_alpha_beta_default`). `log_alpha + log_beta -
/// log_probs + nll` evaluates to `(-inf) + (+inf) = NaN` for nll = +inf
/// samples, and `NaN * 0 = NaN` under IEEE 754 would defeat any outer
/// zero_infinity masking. The fix masks gradient entries for is_inf(nll)
/// samples directly inside `ctc_grad_from_alpha_beta_default`.
#[test]
fn test_ctc_loss_backward_unreachable_is_finite() {
    use burn_tensor::ops::ModuleOps;

    let device: <TestBackend as burn_tensor::backend::Backend>::Device = Default::default();

    // T=2, target=[1, 1] requires three steps, so nll = +inf for this sample.
    let log_probs = TestTensor::<3>::full([2, 1, 3], (1.0f32 / 3.0).ln(), &device);
    let targets =
        TestTensorInt::<2>::from_data(burn_tensor::TensorData::from([[1_i64, 1]]), &device);
    let input_lengths =
        TestTensorInt::<1>::from_data(burn_tensor::TensorData::from([2_i64]), &device);
    let target_lengths =
        TestTensorInt::<1>::from_data(burn_tensor::TensorData::from([2_i64]), &device);
    let grad_loss = TestTensor::<1>::ones([1], &device);

    let grad = <TestBackend as ModuleOps<TestBackend>>::ctc_loss_backward(
        log_probs.into_primitive().tensor(),
        targets.into_primitive(),
        input_lengths.into_primitive(),
        target_lengths.into_primitive(),
        grad_loss.into_primitive().tensor(),
        0,
    );
    let grad_tensor = TestTensor::<3>::from_primitive(burn_tensor::TensorPrimitive::Float(grad));
    let grad_data = grad_tensor.into_data();
    for v in grad_data.iter::<f32>() {
        assert!(
            v.is_finite(),
            "gradient for unreachable target must be finite, got {v}"
        );
        assert_eq!(
            v, 0.0f32,
            "gradient for unreachable target must be zero, got {v}"
        );
    }
}

#[test]
fn test_ctc_loss_empty_target() {
    // T=3, N=1, C=2, blank=0, target_length=0, uniform P = 1/2.
    // With an empty target, the only valid alignment is all blanks.
    // Loss = -ln((1/2)^3) = 3 * ln(2) ~ 2.0794.
    //
    // Uses a [1, 1] targets shape rather than [1, 0] since not every backend
    // supports zero-sized dimensions. The target slot at [0, 0] is never read
    // because target_lengths[0] == 0.
    let log_probs = TestTensor::<3>::full([3, 1, 2], (0.5f32).ln(), &Default::default());
    let targets = TestTensorInt::<2>::from([[0]]);
    let input_lengths = TestTensorInt::<1>::from([3]);
    let target_lengths = TestTensorInt::<1>::from([0]);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0);

    let expected = burn_tensor::TensorData::from([3.0f32 * 2.0f32.ln()]);
    loss.into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-3, 1e-3));
}

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
