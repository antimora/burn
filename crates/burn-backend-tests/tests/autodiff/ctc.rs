use super::*;
use burn_tensor::{TensorData, Tolerance, activation::log_softmax, module::ctc_loss};

/// Verifies that gradients flow through `ctc_loss` on every backend, including
/// those with fused native overrides (cubecl, tch). The autodiff backend should
/// route around the fused kernel and decompose into primitives so gradients can
/// be recorded.
///
/// Reference values from PyTorch's `nn.functional.ctc_loss(reduction='sum').backward()`.
#[test]
fn test_ctc_loss_grad() {
    // Same fixture as the forward cross-backend test (T=5, N=3, C=4) so the
    // expected loss matches PyTorch's reference exactly.
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

    let device = AutodiffDevice::new();
    let logits =
        TestTensor::<3>::from_data(TensorData::new(data, [t_size, n_size, c_size]), &device)
            .require_grad();
    let log_probs = log_softmax(logits.clone(), 2);

    let targets =
        TestTensorInt::<2>::from_data(TensorData::from([[1, 2, 0], [1, 0, 0], [3, 2, 1]]), &device);
    let input_lengths = TestTensorInt::<1>::from_data(TensorData::from([5, 5, 5]), &device);
    let target_lengths = TestTensorInt::<1>::from_data(TensorData::from([2, 1, 3]), &device);

    let loss = ctc_loss(log_probs, targets, input_lengths, target_lengths, 0).sum();
    let grads = loss.backward();
    let logits_grad = logits.grad(&grads).expect(
        "logits should receive a gradient - if this fails on a backend with a fused ctc_loss \
         kernel, the autodiff override needs to skip the fused path",
    );

    // First-row sanity check against PyTorch reference (full grid is in burn-nn tests).
    let expected_first_row =
        TensorData::from([-0.1679008007_f32, -0.4595540464, 0.2795598209, 0.3478950262]);

    let logits_grad_data = logits_grad.into_data();
    let first_row: Vec<f32> = logits_grad_data.iter::<f32>().take(4).collect();
    TensorData::from(first_row.as_slice())
        .assert_approx_eq::<FloatElem>(&expected_first_row, Tolerance::rel_abs(1e-3, 1e-3));
}
