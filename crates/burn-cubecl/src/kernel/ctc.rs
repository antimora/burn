use cubecl::prelude::*;

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{Shape, TensorMetadata};

/// Maximum `2 * max_target_len + 1` the kernel supports. The alpha row lives in
/// shared memory; this size in `f32` consumes 32 KB, comfortably within the per-block
/// shared-memory budget on every consumer GPU we target. Inputs exceeding this
/// will panic with a clear message rather than silently degrading.
const SHARED_ALPHA_CAPACITY: u32 = 8192;

/// CTC alpha-recursion kernel.
///
/// Each cube handles one batch element. `cube_dim.x` is fixed at launch time
/// (capped to the runtime's hardware limit); each thread strides over the `s`
/// positions of the modified label sequence `l'` (length `2 * target_len + 1`),
/// covering arbitrary target lengths up to `SHARED_ALPHA_CAPACITY`. `alpha` is
/// kept in shared memory and the time loop runs sequentially inside the kernel,
/// using two `sync_cube()` barriers per iteration: one to fence reads of
/// `alpha[t-1]` before any thread writes `alpha[t]`, one to publish the new row
/// before the next iteration. This collapses what would otherwise be roughly
/// `40 * T` host-side dispatches into a single kernel launch.
///
/// Note on -inf: we use `F::min_value()` (the most negative finite value) to
/// represent "impossible alignment" rather than true `-inf`. The standard
/// log-sum-exp formulation then works without explicit -inf masking because
/// `exp(min_value - finite)` underflows to 0 cleanly (no NaN from
/// `-inf - -inf`).
#[cube(launch)]
fn ctc_loss_kernel<F: Float, I: Numeric>(
    log_probs: &Tensor<F>,      // [T, N, C]
    targets: &Tensor<I>,        // [N, S_max]
    input_lengths: &Tensor<I>,  // [N]
    target_lengths: &Tensor<I>, // [N]
    output: &mut Tensor<F>,     // [N]
    blank: u32,
    #[comptime] alpha_capacity: u32,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    let n = CUBE_POS_X as usize;
    let cube_dim = CUBE_DIM_X as usize;
    let alpha_cap = alpha_capacity as usize;
    let blank_u = blank as usize;

    let target_len = u32::cast_from(target_lengths[n]) as usize;
    let input_len = u32::cast_from(input_lengths[n]) as usize;
    let l_prime_len = 2 * target_len + 1;

    let lp_t = log_probs.stride(0);
    let lp_n = log_probs.stride(1);
    let lp_c = log_probs.stride(2);
    let tgt_n = targets.stride(0);
    let tgt_s = targets.stride(1);

    // Two adjacent regions: alpha[0..alpha_cap] is the active row, the second
    // half [alpha_cap..2*alpha_cap] is a write scratch buffer that we copy back
    // to the active region after a sync. This avoids RAW hazards across stride
    // batches in the t-loop (a thread writing alpha[s] races with another
    // thread still reading alpha[s-1] or alpha[s-2] for its own s).
    let mut alpha = SharedMemory::<F>::new(2 * alpha_cap);
    let neg_inf = F::min_value();
    let one = F::new(1.0);

    // Initialize alpha at t = 0. Each thread strides over its assigned s positions.
    let mut s = UNIT_POS_X as usize;
    while s < alpha_cap {
        let mut init = neg_inf;
        if s < l_prime_len {
            // l'[s] class for s in {0, 1}
            if s == 0 {
                init = log_probs[n * lp_n + blank_u * lp_c];
            } else if s == 1 {
                let l1 = u32::cast_from(targets[n * tgt_n]) as usize;
                init = log_probs[n * lp_n + l1 * lp_c];
            }
        }
        alpha[s] = init;
        s += cube_dim;
    }
    sync_cube();

    // Sequential time loop. Each iteration re-strides over s positions to
    // compute alpha[t, s] from alpha[t-1, *] and writes back to the same
    // shared memory after a full read fence.
    for t in 1..input_len {
        // First pass: compute new alpha values into local registers.
        // Note: cubecl does not let us hold a per-thread Vec of new values, so
        // we serialize - read, sync, write. Two passes per iteration.
        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            // Compute l'[s] class
            let l_class = if s % 2 == 1 {
                u32::cast_from(targets[n * tgt_n + ((s - 1) / 2) * tgt_s]) as usize
            } else {
                blank_u
            };
            let log_p = log_probs[t * lp_t + n * lp_n + l_class * lp_c];

            // l'[s-2] class - check skip transition eligibility
            let mut l_class_m2 = blank_u;
            if s >= 2 && (s - 2) % 2 == 1 {
                l_class_m2 =
                    u32::cast_from(targets[n * tgt_n + ((s - 2 - 1) / 2) * tgt_s]) as usize;
            }
            let skip_allowed = s >= 2 && l_class != blank_u && l_class != l_class_m2;

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
            // We can write directly to alpha[s] only after every other thread
            // has finished reading alpha[s-1] / alpha[s-2] for *its* s. Since
            // a thread's writes only conflict with neighbors at most 2 away,
            // and we're processing in stride-cube_dim batches, conflicts can
            // happen across stride iterations. Force a full sync below.
            //
            // Stash the new value in shared mem at offset alpha_cap + s as a
            // scratch buffer? Simpler: gather everything then sync then write.
            // Use a second shared buffer.
            alpha[alpha_cap + s] = log_p + combined;
            s += cube_dim;
        }
        sync_cube();

        // Second pass: copy scratch back into the active alpha slots.
        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            alpha[s] = alpha[alpha_cap + s];
            s += cube_dim;
        }
        sync_cube();
    }

    // Reduce: only thread 0 writes the output for this batch element.
    if UNIT_POS_X == 0 {
        let last_blank = alpha[2 * target_len];
        // Guard target_len = 0: index 2*0 - 1 underflows. Pick last_blank itself
        // so log_sum_exp reduces to just last_blank.
        let mut last_label = last_blank;
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

/// Fused CTC loss for burn-cubecl. Single kernel launch covers the entire
/// alpha recursion across all timesteps.
///
/// Panics if `2 * max_target_len + 1` exceeds `SHARED_ALPHA_CAPACITY` (8192).
pub fn ctc_loss<R: CubeRuntime>(
    log_probs: CubeTensor<R>,
    targets: CubeTensor<R>,
    input_lengths: CubeTensor<R>,
    target_lengths: CubeTensor<R>,
    blank: usize,
) -> CubeTensor<R> {
    let log_probs_shape = log_probs.shape();
    let [_t, batch_size, _c] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime = 2 * max_target_len + 1;

    assert!(
        max_l_prime as u32 <= SHARED_ALPHA_CAPACITY,
        "ctc_loss: 2 * max_target_len + 1 = {} exceeds the kernel's shared-memory \
         alpha capacity ({}). Reduce target length or raise SHARED_ALPHA_CAPACITY.",
        max_l_prime,
        SHARED_ALPHA_CAPACITY,
    );

    // Pick a thread count that fits the runtime's per-cube limit. We don't
    // need one thread per s position - threads stride over s.
    let hw_max = log_probs.client.properties().hardware.max_cube_dim.0;
    let cube_dim_x = (max_l_prime as u32).min(hw_max).min(256);

    let client = log_probs.client.clone();
    let device = log_probs.device.clone();
    let f_dtype = log_probs.dtype;
    let i_dtype = targets.dtype;
    let output = empty_device_dtype::<R>(client.clone(), device, Shape::new([batch_size]), f_dtype);

    let cube_count = CubeCount::Static(batch_size as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(cube_dim_x);

    ctc_loss_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        log_probs.into_tensor_arg(),
        targets.into_tensor_arg(),
        input_lengths.into_tensor_arg(),
        target_lengths.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        blank as u32,
        SHARED_ALPHA_CAPACITY,
        [f_dtype.into(), i_dtype.into()],
    );

    output
}
