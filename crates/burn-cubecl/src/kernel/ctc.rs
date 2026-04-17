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
/// Impossible alignments are represented as `F::NEG_INFINITY` so that when an
/// entire sequence has no valid alignment (e.g. `target_length > input_length`)
/// the forward loss comes out as `+inf`, which is what `zero_infinity` masking
/// in `burn-nn` detects via `is_inf`. Each `log_sum_exp` guards `mx == -inf`
/// explicitly to avoid the `-inf - -inf = NaN` case that would otherwise
/// propagate through the recursion.
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
    let neg_inf = F::new(f32::NEG_INFINITY);
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

            // log_sum_exp(a_s, a_s_m1). Guarding `mx == -inf` avoids the
            // `-inf - -inf = NaN` case; when both terms are -inf the sum is
            // -inf too.
            let mut mx01 = a_s;
            let mut mn01 = a_s_m1;
            if a_s_m1 > a_s {
                mx01 = a_s_m1;
                mn01 = a_s;
            }
            let lse_01 = if mx01 == neg_inf {
                mx01
            } else {
                mx01 + (one + (mn01 - mx01).exp()).ln()
            };

            let mut combined = lse_01;
            if skip_allowed {
                let mut mx2 = lse_01;
                let mut mn2 = a_s_m2;
                if a_s_m2 > lse_01 {
                    mx2 = a_s_m2;
                    mn2 = lse_01;
                }
                combined = if mx2 == neg_inf {
                    mx2
                } else {
                    mx2 + (one + (mn2 - mx2).exp()).ln()
                };
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
        // Guard target_len = 0: index 2*0 - 1 underflows. Use -inf so
        // log_sum_exp(last_blank, -inf) = last_blank (log_sum_exp(x, x) = x+ln2
        // would be wrong here).
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
        let log_lik = if mx == neg_inf {
            mx
        } else {
            mx + (one + (mn - mx).exp()).ln()
        };

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

    // Pass the actual max_l_prime (not the static capacity) so shared memory
    // is sized to what we need. Metal limits threadgroup memory to 32 KB;
    // allocating 2 * 8192 * sizeof(f32) = 64 KB would silently corrupt on
    // Apple GPUs. Different max_l_prime values trigger separate kernel
    // compilations (it's a comptime param), but that's fine: target lengths
    // are stable within a dataset.
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
        max_l_prime as u32,
        [f_dtype.into(), i_dtype.into()],
    );

    output
}

/// Fused CTC alpha + beta recursion kernel.
///
/// Runs the full forward alpha recursion and reverse beta recursion for one
/// batch element per cube, reusing the same shared-memory layout twice.
/// Writes `alpha_out[T, N, 2S+1]`, `beta_out[T, N, 2S+1]` and the per-sample
/// negative log-likelihood `nll_out[N]`. The three outputs are everything the
/// default CTC gradient-composition helper needs, so the caller can finish the
/// backward pass with a handful of element-wise tensor ops.
///
/// The alpha phase is identical to `ctc_loss_kernel` except it additionally
/// publishes each row to global memory. The beta phase mirrors it in reverse:
/// initialize at `t = input_len - 1` from `log_probs[t, l'[s]]` at the two
/// boundary `s` positions, then step backward reading `beta[t+1, s]`,
/// `beta[t+1, s+1]`, and (when the skip transition is allowed) `beta[t+1, s+2]`.
#[cube(launch)]
fn ctc_alpha_beta_kernel<F: Float, I: Numeric>(
    log_probs: &Tensor<F>,      // [T, N, C]
    targets: &Tensor<I>,        // [N, S_max]
    input_lengths: &Tensor<I>,  // [N]
    target_lengths: &Tensor<I>, // [N]
    alpha_out: &mut Tensor<F>,  // [T, N, 2S+1]
    beta_out: &mut Tensor<F>,   // [T, N, 2S+1]
    nll_out: &mut Tensor<F>,    // [N]
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
    let ao_t = alpha_out.stride(0);
    let ao_n = alpha_out.stride(1);
    let ao_s = alpha_out.stride(2);
    let bo_t = beta_out.stride(0);
    let bo_n = beta_out.stride(1);
    let bo_s = beta_out.stride(2);

    // Shared memory layout: [0..alpha_cap] is the active row; [alpha_cap..2*alpha_cap]
    // is scratch for the next row. Same layout is reused for alpha and beta. Beta
    // reads are guarded by `s + 1 < l_prime_len` / `s + 2 < l_prime_len`, so the
    // residual alpha values sitting in the active row between phases are never
    // observed by beta (its boundary init overwrites every slot it reads).
    let mut state = SharedMemory::<F>::new(2 * alpha_cap);
    let neg_inf = F::new(f32::NEG_INFINITY);
    let one = F::new(1.0);

    // ---------- Alpha phase (forward) ----------
    //
    // Initialize alpha at t = 0. We write s=0 and s=1 from log_probs; the rest
    // is -inf. Every thread also publishes its slot to alpha_out[0, n, s].
    let mut s = UNIT_POS_X as usize;
    while s < alpha_cap {
        let mut init = neg_inf;
        if s < l_prime_len {
            if s == 0 {
                init = log_probs[n * lp_n + blank_u * lp_c];
            } else if s == 1 {
                let l1 = u32::cast_from(targets[n * tgt_n]) as usize;
                init = log_probs[n * lp_n + l1 * lp_c];
            }
        }
        state[s] = init;
        if s < l_prime_len {
            alpha_out[n * ao_n + s * ao_s] = init;
        }
        s += cube_dim;
    }
    sync_cube();

    for t in 1..input_len {
        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            let l_class = if s % 2 == 1 {
                u32::cast_from(targets[n * tgt_n + ((s - 1) / 2) * tgt_s]) as usize
            } else {
                blank_u
            };
            let log_p = log_probs[t * lp_t + n * lp_n + l_class * lp_c];

            let mut l_class_m2 = blank_u;
            if s >= 2 && (s - 2) % 2 == 1 {
                l_class_m2 =
                    u32::cast_from(targets[n * tgt_n + ((s - 2 - 1) / 2) * tgt_s]) as usize;
            }
            let skip_allowed = s >= 2 && l_class != blank_u && l_class != l_class_m2;

            let a_s = state[s];
            let mut a_s_m1 = neg_inf;
            if s >= 1 {
                a_s_m1 = state[s - 1];
            }
            let mut a_s_m2 = neg_inf;
            if s >= 2 {
                a_s_m2 = state[s - 2];
            }

            let mut mx01 = a_s;
            let mut mn01 = a_s_m1;
            if a_s_m1 > a_s {
                mx01 = a_s_m1;
                mn01 = a_s;
            }
            let lse_01 = if mx01 == neg_inf {
                mx01
            } else {
                mx01 + (one + (mn01 - mx01).exp()).ln()
            };

            let mut combined = lse_01;
            if skip_allowed {
                let mut mx2 = lse_01;
                let mut mn2 = a_s_m2;
                if a_s_m2 > lse_01 {
                    mx2 = a_s_m2;
                    mn2 = lse_01;
                }
                combined = if mx2 == neg_inf {
                    mx2
                } else {
                    mx2 + (one + (mn2 - mx2).exp()).ln()
                };
            }
            state[alpha_cap + s] = log_p + combined;
            s += cube_dim;
        }
        sync_cube();

        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            state[s] = state[alpha_cap + s];
            alpha_out[t * ao_t + n * ao_n + s * ao_s] = state[s];
            s += cube_dim;
        }
        sync_cube();
    }

    if UNIT_POS_X == 0 {
        let last_blank = state[2 * target_len];
        // See ctc_loss_kernel: -inf sentinel keeps log_sum_exp correct for target_len = 0.
        let mut last_label = neg_inf;
        if target_len > 0 {
            last_label = state[2 * target_len - 1];
        }
        let mut mx = last_blank;
        let mut mn = last_label;
        if last_label > last_blank {
            mx = last_label;
            mn = last_blank;
        }
        let log_lik = if mx == neg_inf {
            mx
        } else {
            mx + (one + (mn - mx).exp()).ln()
        };
        nll_out[n] = F::new(0.0) - log_lik;
    }

    // Fence thread 0's read of state[2*target_len] / state[2*target_len - 1]
    // against the beta boundary init, which writes those same positions.
    sync_cube();

    // ---------- Beta phase (reverse) ----------

    if input_len > 0 {
        // Boundary initialization at t = input_len - 1: set beta[s] = log_probs[t, l'[s]]
        // at s = 2*target_len, and when target_len > 0 also at s = 2*target_len - 1.
        // All other s positions in range get -inf.
        let t_last = input_len - 1;
        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            let is_last_blank = s == 2 * target_len;
            let is_last_label = target_len > 0 && s == 2 * target_len - 1;
            let mut init = neg_inf;
            if is_last_blank || is_last_label {
                let l_class = if s % 2 == 1 {
                    u32::cast_from(targets[n * tgt_n + ((s - 1) / 2) * tgt_s]) as usize
                } else {
                    blank_u
                };
                init = log_probs[t_last * lp_t + n * lp_n + l_class * lp_c];
            }
            state[s] = init;
            beta_out[t_last * bo_t + n * bo_n + s * bo_s] = init;
            s += cube_dim;
        }
        sync_cube();

        // Step back from t = input_len - 2 down to t = 0.
        for t_rev in 1..input_len {
            let t = input_len - 1 - t_rev;

            let mut s = UNIT_POS_X as usize;
            while s < l_prime_len {
                let l_class = if s % 2 == 1 {
                    u32::cast_from(targets[n * tgt_n + ((s - 1) / 2) * tgt_s]) as usize
                } else {
                    blank_u
                };
                let log_p = log_probs[t * lp_t + n * lp_n + l_class * lp_c];

                let mut l_class_p2 = blank_u;
                if s + 2 < l_prime_len && (s + 2) % 2 == 1 {
                    l_class_p2 =
                        u32::cast_from(targets[n * tgt_n + s.div_ceil(2) * tgt_s]) as usize;
                }
                let skip_allowed =
                    s + 2 < l_prime_len && l_class != blank_u && l_class != l_class_p2;

                let b_s = state[s];
                let mut b_s_p1 = neg_inf;
                if s + 1 < l_prime_len {
                    b_s_p1 = state[s + 1];
                }
                let mut b_s_p2 = neg_inf;
                if s + 2 < l_prime_len {
                    b_s_p2 = state[s + 2];
                }

                let mut mx01 = b_s;
                let mut mn01 = b_s_p1;
                if b_s_p1 > b_s {
                    mx01 = b_s_p1;
                    mn01 = b_s;
                }
                let lse_01 = if mx01 == neg_inf {
                    mx01
                } else {
                    mx01 + (one + (mn01 - mx01).exp()).ln()
                };

                let mut combined = lse_01;
                if skip_allowed {
                    let mut mx2 = lse_01;
                    let mut mn2 = b_s_p2;
                    if b_s_p2 > lse_01 {
                        mx2 = b_s_p2;
                        mn2 = lse_01;
                    }
                    combined = if mx2 == neg_inf {
                        mx2
                    } else {
                        mx2 + (one + (mn2 - mx2).exp()).ln()
                    };
                }
                state[alpha_cap + s] = log_p + combined;
                s += cube_dim;
            }
            sync_cube();

            let mut s = UNIT_POS_X as usize;
            while s < l_prime_len {
                state[s] = state[alpha_cap + s];
                beta_out[t * bo_t + n * bo_n + s * bo_s] = state[s];
                s += cube_dim;
            }
            sync_cube();
        }
    }
}

/// Host entry point for the fused alpha + beta + nll kernel.
///
/// Returns `(log_alpha_full, log_beta_full, nll)` with shapes
/// `([T, N, 2S+1], [T, N, 2S+1], [N])`. Positions outside the valid
/// `(t < input_length, s < 2*target_length+1)` rectangle hold the
/// pre-fill value `-inf`, matching the default backend's convention.
///
/// Panics if `2 * max_target_len + 1` exceeds `SHARED_ALPHA_CAPACITY`.
pub fn ctc_alpha_beta<R: CubeRuntime>(
    log_probs: CubeTensor<R>,
    targets: CubeTensor<R>,
    input_lengths: CubeTensor<R>,
    target_lengths: CubeTensor<R>,
    blank: usize,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let log_probs_shape = log_probs.shape();
    let [max_input_length, batch_size, _c] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime = 2 * max_target_len + 1;

    assert!(
        max_l_prime as u32 <= SHARED_ALPHA_CAPACITY,
        "ctc_loss_backward: 2 * max_target_len + 1 = {} exceeds the kernel's shared-memory \
         alpha capacity ({}). Reduce target length or raise SHARED_ALPHA_CAPACITY.",
        max_l_prime,
        SHARED_ALPHA_CAPACITY,
    );

    let hw_max = log_probs.client.properties().hardware.max_cube_dim.0;
    let cube_dim_x = (max_l_prime as u32).min(hw_max).min(256);

    let client = log_probs.client.clone();
    let device = log_probs.device.clone();
    let f_dtype = log_probs.dtype;
    let i_dtype = targets.dtype;

    // Pre-fill alpha/beta with -inf so positions the kernel doesn't touch
    // (s >= 2U+1, or t outside the valid range for an individual batch
    // element) are not read as stale zeros by the gradient composition.
    let shape_abt = Shape::new([max_input_length, batch_size, max_l_prime]);
    let neg_inf = InputScalar::new(f32::NEG_INFINITY, f_dtype);
    let alpha_out = crate::ops::numeric::full_device_dtype::<R>(
        client.clone(),
        shape_abt.clone(),
        device.clone(),
        neg_inf,
        f_dtype,
    );
    let beta_out = crate::ops::numeric::full_device_dtype::<R>(
        client.clone(),
        shape_abt,
        device.clone(),
        neg_inf,
        f_dtype,
    );
    let nll_out =
        empty_device_dtype::<R>(client.clone(), device, Shape::new([batch_size]), f_dtype);

    let cube_count = CubeCount::Static(batch_size as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(cube_dim_x);

    ctc_alpha_beta_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        log_probs.into_tensor_arg(),
        targets.into_tensor_arg(),
        input_lengths.into_tensor_arg(),
        target_lengths.into_tensor_arg(),
        alpha_out.clone().into_tensor_arg(),
        beta_out.clone().into_tensor_arg(),
        nll_out.clone().into_tensor_arg(),
        blank as u32,
        max_l_prime as u32,
        [f_dtype.into(), i_dtype.into()],
    );

    (alpha_out, beta_out, nll_out)
}
