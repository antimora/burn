use cubecl::prelude::*;

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{Shape, TensorMetadata};

/// CTC alpha-recursion kernel.
///
/// Each cube handles one batch element; each thread within a cube owns one `s`
/// position of the modified label sequence `l'` (length `2 * target_len + 1`).
/// `alpha` lives in shared memory and the time loop runs sequentially inside
/// the kernel with `sync_cube()` between steps so threads see each other's
/// writes. This collapses what would otherwise be ~40 * T host-side dispatches
/// into a single kernel launch.
///
/// Note on -inf: we use `F::min_value()` (the most negative finite value) to
/// represent "impossible alignment" rather than true `-inf`. The standard
/// log-sum-exp formulation then works without explicit -inf masking because
/// `exp(min_value - finite)` underflows to 0 cleanly (no NaN from -inf - -inf).
#[cube(launch)]
fn ctc_loss_kernel<F: Float, I: Numeric>(
    log_probs: &Tensor<F>,      // [T, N, C]
    targets: &Tensor<I>,        // [N, S_max]
    input_lengths: &Tensor<I>,  // [N]
    target_lengths: &Tensor<I>, // [N]
    output: &mut Tensor<F>,     // [N]
    blank: u32,
    #[comptime] max_l_prime: u32,
) {
    let n = CUBE_POS_X as usize;
    let s = UNIT_POS_X as usize;
    let max_lp = max_l_prime as usize;
    let blank_u = blank as usize;

    let target_len = u32::cast_from(target_lengths[n]) as usize;
    let input_len = u32::cast_from(input_lengths[n]) as usize;
    let l_prime_len = 2 * target_len + 1;
    let active = s < l_prime_len;

    let lp_t = log_probs.stride(0);
    let lp_n = log_probs.stride(1);
    let lp_c = log_probs.stride(2);
    let tgt_n = targets.stride(0);
    let tgt_s = targets.stride(1);

    // Compute l'[s] class
    let mut l_class = blank_u;
    if active && s % 2 == 1 {
        l_class = u32::cast_from(targets[n * tgt_n + ((s - 1) / 2) * tgt_s]) as usize;
    }

    // Compute l'[s-2] class (only meaningful for s >= 2)
    let mut l_class_m2 = blank_u;
    if active && s >= 2 && (s - 2) % 2 == 1 {
        l_class_m2 = u32::cast_from(targets[n * tgt_n + ((s - 2 - 1) / 2) * tgt_s]) as usize;
    }
    let skip_allowed = active && s >= 2 && l_class != blank_u && l_class != l_class_m2;

    let mut alpha = SharedMemory::<F>::new(max_lp);
    let neg_inf = F::min_value();
    let one = F::new(1.0);

    // Initialize alpha at t = 0
    let mut init = neg_inf;
    if active && s == 0 {
        init = log_probs[n * lp_n + blank_u * lp_c];
    } else if active && s == 1 {
        init = log_probs[n * lp_n + l_class * lp_c];
    }
    if s < max_lp {
        alpha[s] = init;
    }
    sync_cube();

    // Sequential time loop
    for t in 1..input_len {
        let mut new_alpha = neg_inf;
        if active {
            let log_p = log_probs[t * lp_t + n * lp_n + l_class * lp_c];

            let a_s = alpha[s];
            let mut a_s_m1 = neg_inf;
            if s >= 1 {
                a_s_m1 = alpha[s - 1];
            }
            let mut a_s_m2 = neg_inf;
            if s >= 2 {
                a_s_m2 = alpha[s - 2];
            }

            // log_sum_exp(a_s, a_s_m1) - using min_value avoids -inf NaN issues
            let mut mx01 = a_s;
            let mut mn01 = a_s_m1;
            if a_s_m1 > a_s {
                mx01 = a_s_m1;
                mn01 = a_s;
            }
            let lse_01 = mx01 + (one + (mn01 - mx01).exp()).ln();

            // Optionally combine with a_s_m2
            let mut combined = lse_01;
            if skip_allowed {
                let mut mx2 = lse_01;
                let mut mn2 = a_s_m2;
                if a_s_m2 > lse_01 {
                    mx2 = a_s_m2;
                    mn2 = lse_01;
                }
                combined = mx2 + (one + (mn2 - mx2).exp()).ln();
            }
            new_alpha = log_p + combined;
        }

        sync_cube();
        if active {
            alpha[s] = new_alpha;
        }
        sync_cube();
    }

    // Reduce: only thread 0 writes the output for this batch element
    if s == 0 {
        let last_blank = alpha[2 * target_len];
        let mut last_label = neg_inf;
        if target_len > 0 {
            last_label = alpha[2 * target_len - 1];
        }

        let mut mx = last_blank;
        let mut mn = last_label;
        if last_label > last_blank {
            mx = last_label;
            mn = last_blank;
        }
        let log_lik = mx + (one + (mn - mx).exp()).ln();

        output[n] = F::new(0.0) - log_lik;
    }
}

/// Maximum cube_dim that the fused kernel supports. Inputs whose
/// `2 * max_target_len + 1` exceeds this fall back to the default impl.
pub const CTC_MAX_L_PRIME: u32 = 1024;

/// Fused CTC loss for burn-cubecl. Returns `None` when the input shape does not
/// fit the single-cube design, in which case the caller should fall back.
pub fn ctc_loss<R: CubeRuntime>(
    log_probs: CubeTensor<R>,
    targets: CubeTensor<R>,
    input_lengths: CubeTensor<R>,
    target_lengths: CubeTensor<R>,
    blank: usize,
) -> Option<CubeTensor<R>> {
    let log_probs_shape = log_probs.shape();
    let [_t, batch_size, _c] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime = (2 * max_target_len + 1) as u32;

    if max_l_prime > CTC_MAX_L_PRIME {
        return None;
    }

    let client = log_probs.client.clone();
    let device = log_probs.device.clone();
    let output = empty_device_dtype::<R>(
        client.clone(),
        device,
        Shape::new([batch_size]),
        log_probs.dtype,
    );

    let cube_count = CubeCount::Static(batch_size as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(max_l_prime);

    ctc_loss_kernel::launch::<f32, i32, R>(
        &client,
        cube_count,
        cube_dim,
        log_probs.into_tensor_arg(),
        targets.into_tensor_arg(),
        input_lengths.into_tensor_arg(),
        target_lengths.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        blank as u32,
        max_l_prime,
    );

    Some(output)
}
