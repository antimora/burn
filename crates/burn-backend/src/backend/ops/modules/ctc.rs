use burn_std::{Shape, Slice};

use crate::{
    Backend, TensorMetadata, get_device_settings,
    tensor::{FloatTensor, IntTensor},
};

/// Default CTC loss implementation using the forward (alpha) algorithm.
///
/// Computes the Connectionist Temporal Classification loss by summing over
/// all valid alignments between the input and target sequences.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities of shape `[T, N, C]`
/// * `targets` - Target indices of shape `[N, S]`
/// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
/// * `target_lengths` - Actual target lengths per batch element `[N]`
/// * `blank` - Index of the blank label
///
/// # Returns
///
/// Per-sample loss of shape `[N]`
pub fn ctc_loss_default<B: Backend>(
    log_probs: FloatTensor<B>,
    targets: IntTensor<B>,
    input_lengths: IntTensor<B>,
    target_lengths: IntTensor<B>,
    blank: usize,
) -> FloatTensor<B> {
    let alpha = AlphaCtx::<B>::compute(
        log_probs,
        &targets,
        input_lengths,
        target_lengths.clone(),
        blank,
    );
    extract_loss::<B>(&alpha, target_lengths)
}

/// Default CTC loss backward: returns the gradient w.r.t. `log_probs`.
///
/// Implements the standard CTC gradient using the forward (alpha) and backward
/// (beta) recursions:
///
/// `grad[t, n, k] = grad_loss[n] * (exp(log_probs[t, n, k]) -
///                                  sum_{s: l'[n, s] == k} exp(log_alpha[t, n, s] +
///                                                             log_beta[t, n, s] - nll[n]))`
///
/// where `nll[n] = -log_likelihood = forward loss for sample n`.
///
/// Positions where `t >= input_lengths[n]` are masked to zero so they do not
/// contaminate the gradient.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities of shape `[T, N, C]`
/// * `targets` - Target indices of shape `[N, S]`
/// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
/// * `target_lengths` - Actual target lengths per batch element `[N]`
/// * `grad_loss` - Upstream gradient w.r.t. the per-sample loss, shape `[N]`
/// * `blank` - Index of the blank label
///
/// # Returns
///
/// Gradient w.r.t. `log_probs` of shape `[T, N, C]`
pub fn ctc_loss_backward_default<B: Backend>(
    log_probs: FloatTensor<B>,
    targets: IntTensor<B>,
    input_lengths: IntTensor<B>,
    target_lengths: IntTensor<B>,
    grad_loss: FloatTensor<B>,
    blank: usize,
) -> FloatTensor<B> {
    let alpha = AlphaCtx::<B>::compute(
        log_probs.clone(),
        &targets,
        input_lengths.clone(),
        target_lengths.clone(),
        blank,
    );
    let log_beta_full = compute_log_beta_full::<B>(
        log_probs.clone(),
        &alpha,
        input_lengths.clone(),
        target_lengths.clone(),
    );
    let nll = extract_loss::<B>(&alpha, target_lengths);

    ctc_grad_from_alpha_beta_default::<B>(
        log_probs,
        targets,
        input_lengths,
        grad_loss,
        alpha.full,
        log_beta_full,
        nll,
        blank,
    )
}

/// Compose the CTC gradient w.r.t. `log_probs` from pre-computed alpha, beta, and nll.
///
/// The T-iteration alpha and beta recursions are the dominant cost of the backward
/// pass. Backends that fuse those recursions into a single kernel launch (such as
/// burn-cubecl) can call this helper to reuse the gradient composition.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities `[T, N, C]`
/// * `targets` - Target label indices `[N, S]`
/// * `input_lengths` - Actual input sequence lengths per batch element `[N]`
/// * `grad_loss` - Upstream gradient w.r.t. the per-sample loss `[N]`
/// * `log_alpha_full` - Alpha recursion output `[T, N, 2S+1]`
/// * `log_beta_full` - Beta recursion output `[T, N, 2S+1]`
/// * `nll` - Per-sample negative log-likelihood (forward loss) `[N]`
/// * `blank` - Index of the blank label
#[allow(clippy::too_many_arguments)]
pub fn ctc_grad_from_alpha_beta_default<B: Backend>(
    log_probs: FloatTensor<B>,
    targets: IntTensor<B>,
    input_lengths: IntTensor<B>,
    grad_loss: FloatTensor<B>,
    log_alpha_full: FloatTensor<B>,
    log_beta_full: FloatTensor<B>,
    nll: FloatTensor<B>,
    blank: usize,
) -> FloatTensor<B> {
    let log_probs_shape = log_probs.shape();
    let [max_input_length, batch_size, num_classes] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime_len = 2 * max_target_len + 1;
    let device = B::float_device(&log_probs);
    let settings = get_device_settings::<B>(&device);

    let blank_inserted_targets = insert_blanks::<B>(
        &targets,
        batch_size,
        max_target_len,
        max_l_prime_len,
        blank,
        &device,
        settings.int_dtype,
    );

    // Both log_alpha[t, n, s] and log_beta[t, n, s] include a factor of
    // log_probs[t, n, l'[s]] (added on every recursion step). The CTC paper's
    // alpha_hat * beta_hat product divides one of those factors out, so we
    // subtract log_probs[t, n, l'[s]] when forming log_post.
    //
    // We then divide by total_prob = exp(-nll) to obtain the alignment
    // posterior, which in log space means *adding* nll (since nll = -log P,
    // dividing by P is adding nll). Per PyTorch's CTC backward kernel:
    //   log_post[t, n, s] = log_alpha + log_beta - log_probs[t, n, l'[s]] - log P
    //                     = log_alpha + log_beta - log_probs[t, n, l'[s]] + nll
    let indices_3d = B::int_reshape(
        blank_inserted_targets,
        Shape::new([1, batch_size, max_l_prime_len]),
    );
    let indices_3d = B::int_expand(
        indices_3d,
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let log_probs_at_l = B::float_gather(2, log_probs.clone(), indices_3d.clone());

    // Samples with an unreachable target yield nll = +inf. For those, log_alpha
    // stays at -inf at many (t, s) while log_beta is finite at the boundary, so
    // log_post = (-inf) + finite - finite + (+inf) = NaN and -exp(NaN) = NaN
    // contaminates the gradient. `NaN * 0 = NaN` under IEEE 754, so zero_infinity
    // masking on the outer grad_loss can't clear it. Capture the mask now and
    // zero the gradient for those samples at the end.
    let nll_is_inf = B::float_is_inf(nll.clone(), settings.bool_dtype);

    let nll_b = B::float_reshape(nll, Shape::new([1, batch_size, 1]));
    let nll_b = B::float_expand(
        nll_b,
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let log_post = B::float_add(
        B::float_sub(B::float_add(log_alpha_full, log_beta_full), log_probs_at_l),
        nll_b,
    );

    // grad starts as exp(log_probs) * grad_loss[None, :, None].
    // Build both the [T, N, C] and [T, N, 2S+1] broadcasts from the same
    // [1, N, 1] source instead of slicing C down to 1 from grad_loss_b.
    let grad_loss_3d = B::float_reshape(grad_loss, Shape::new([1, batch_size, 1]));
    let grad_loss_b = B::float_expand(
        grad_loss_3d.clone(),
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let mut grad = B::float_mul(B::float_exp(log_probs), grad_loss_b);

    // Subtract sum over s of grad_loss[n] * exp(log_post[t, n, s]) at index l'[n, s].
    let grad_loss_post = B::float_expand(
        grad_loss_3d,
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let scatter_value = B::float_neg(B::float_mul(B::float_exp(log_post), grad_loss_post));

    grad = B::float_scatter_add(2, grad, indices_3d, scatter_value);

    // Mask out timesteps where t >= input_lengths[n].
    let t_indices = B::int_arange(0..max_input_length as i64, &device, settings.int_dtype);
    let t_indices = B::int_reshape(t_indices, Shape::new([max_input_length, 1, 1]));
    let t_indices = B::int_expand(
        t_indices,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let il_b = B::int_reshape(input_lengths, Shape::new([1, batch_size, 1]));
    let il_b = B::int_expand(
        il_b,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let oob_mask = B::int_greater_equal(t_indices, il_b, settings.bool_dtype);

    // Broadcast the nll-is-inf mask across [T, N, C] and OR with oob_mask so a
    // single mask_fill zeros both unreachable samples and out-of-bound timesteps.
    let nll_inf_b = B::bool_reshape(nll_is_inf, Shape::new([1, batch_size, 1]));
    let nll_inf_b = B::bool_expand(
        nll_inf_b,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let mask = B::bool_or(oob_mask, nll_inf_b);
    B::float_mask_fill(grad, mask, 0.0.into())
}

/// Cached state from the alpha recursion, shared between `ctc_loss_default` and
/// `ctc_loss_backward_default`.
struct AlphaCtx<B: Backend> {
    /// `log_alpha[T, N, 2S+1]` (full history; needed for backward).
    full: FloatTensor<B>,
    /// `log_alpha[T-1, :, :]` (last timestep; used to read out the loss).
    last: FloatTensor<B>,
    /// `l'` after blank insertion `[N, 2S+1]`.
    blank_inserted_targets: IntTensor<B>,
    /// `log_probs[t, n, l'[n, s]]` pre-gathered as `[T, N, 2S+1]`. The
    /// alpha recursion gathers this once, then `compute_log_beta_full`
    /// borrows it via `AlphaCtx` so beta does not re-gather.
    log_probs_at_l_full: FloatTensor<B>,
    max_l_prime_len: usize,
}

impl<B: Backend> AlphaCtx<B> {
    fn compute(
        log_probs: FloatTensor<B>,
        targets: &IntTensor<B>,
        input_lengths: IntTensor<B>,
        target_lengths: IntTensor<B>,
        blank: usize,
    ) -> Self {
        let log_probs_shape = log_probs.shape();
        let [max_input_length, batch_size, num_classes] = log_probs_shape.dims::<3>();
        let target_shape = targets.shape();
        let max_target_len = target_shape.dims::<2>()[1];
        let device = B::float_device(&log_probs);
        let float_dtype: burn_std::FloatDType = log_probs.dtype().into();
        let settings = get_device_settings::<B>(&device);

        let max_l_prime_len = 2 * max_target_len + 1;
        let blank_inserted_targets = insert_blanks::<B>(
            targets,
            batch_size,
            max_target_len,
            max_l_prime_len,
            blank,
            &device,
            settings.int_dtype,
        );

        // Pre-allocate the full alpha tensor [T, N, 2S+1] filled with -inf.
        let mut alpha_full = B::float_full(
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );

        // Initialize alpha[0, :, 0] = log_probs[0, :, blank]
        // and alpha[0, :, 1] = log_probs[0, :, l'[1]].
        let log_probs_t0 = B::float_slice(
            log_probs.clone(),
            &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
        );
        let log_probs_t0 = B::float_reshape(log_probs_t0, Shape::new([batch_size, num_classes]));

        let first_blank = B::int_slice(
            blank_inserted_targets.clone(),
            &[Slice::full(), Slice::new(0, Some(1), 1)],
        );
        let log_prob_blank = B::float_gather(1, log_probs_t0.clone(), first_blank);
        // Broadcast to [1, N, 1] for slice_assign into alpha_full.
        let log_prob_blank_3d = B::float_reshape(log_prob_blank, Shape::new([1, batch_size, 1]));
        alpha_full = B::float_slice_assign(
            alpha_full,
            &[
                Slice::new(0, Some(1), 1),
                Slice::full(),
                Slice::new(0, Some(1), 1),
            ],
            log_prob_blank_3d,
        );

        if max_l_prime_len > 1 {
            let first_label = B::int_slice(
                blank_inserted_targets.clone(),
                &[Slice::full(), Slice::new(1, Some(2), 1)],
            );
            let log_prob_first = B::float_gather(1, log_probs_t0, first_label);
            let log_prob_first_3d =
                B::float_reshape(log_prob_first, Shape::new([1, batch_size, 1]));
            alpha_full = B::float_slice_assign(
                alpha_full,
                &[
                    Slice::new(0, Some(1), 1),
                    Slice::full(),
                    Slice::new(1, Some(2), 1),
                ],
                log_prob_first_3d,
            );
        }

        // Track the latest row separately for the recursion (cheaper than
        // re-slicing alpha_full each iteration).
        let mut log_alpha = B::float_slice(
            alpha_full.clone(),
            &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
        );
        log_alpha = B::float_reshape(log_alpha, Shape::new([batch_size, max_l_prime_len]));

        let l_prime_mask = create_l_prime_mask::<B>(
            &blank_inserted_targets,
            batch_size,
            max_l_prime_len,
            blank,
            &device,
            settings.int_dtype,
            settings.bool_dtype,
        );
        let s_mask = create_s_mask::<B>(
            &target_lengths,
            batch_size,
            max_l_prime_len,
            &device,
            settings.int_dtype,
            settings.bool_dtype,
        );

        // Hoist out of the T-loop: padding tensors for right_shift (same
        // value/shape at every iteration) and the full `[T, N, 2S+1]`
        // gather of log_probs at l' (one T-sized gather replaces T small
        // gathers).
        let pad_1 = B::float_full(
            Shape::new([batch_size, 1]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );
        let pad_2 = B::float_full(
            Shape::new([batch_size, 2]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );
        let indices_3d = B::int_expand(
            B::int_reshape(
                blank_inserted_targets.clone(),
                Shape::new([1, batch_size, max_l_prime_len]),
            ),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let log_probs_at_l_full = B::float_gather(2, log_probs.clone(), indices_3d);

        // Precompute `combined_mask_all[t, n, s] = (input_lengths[n] > t) AND
        // s_mask[n, s]` for every t in one shot. The T-loop reads its row via
        // a metadata-only slice instead of recomputing the `int_greater_elem`
        // + bool_and per iteration.
        let t_indices_2d = B::int_expand(
            B::int_reshape(
                B::int_arange(0..max_input_length as i64, &device, settings.int_dtype),
                Shape::new([max_input_length, 1]),
            ),
            Shape::new([max_input_length, batch_size]),
        );
        let il_tn = B::int_expand(
            B::int_reshape(input_lengths.clone(), Shape::new([1, batch_size])),
            Shape::new([max_input_length, batch_size]),
        );
        let t_mask_all = B::bool_expand(
            B::bool_reshape(
                B::int_greater(il_tn, t_indices_2d, settings.bool_dtype),
                Shape::new([max_input_length, batch_size, 1]),
            ),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let s_mask_bcast = B::bool_expand(
            B::bool_reshape(s_mask.clone(), Shape::new([1, batch_size, max_l_prime_len])),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let combined_mask_all = B::bool_and(t_mask_all, s_mask_bcast);

        for t in 1..max_input_length {
            let combined_mask = B::bool_reshape(
                B::bool_slice(
                    combined_mask_all.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, max_l_prime_len]),
            );

            let log_alpha_s = log_alpha.clone();
            let log_alpha_s_m1 = right_shift::<B>(&log_alpha, &pad_1, max_l_prime_len, 1);
            let log_alpha_s_m2 = right_shift::<B>(&log_alpha, &pad_2, max_l_prime_len, 2);

            let bar = log_sum_exp::<B>(log_alpha_s, log_alpha_s_m1, settings.bool_dtype);
            let bar_with_skip = log_sum_exp::<B>(bar.clone(), log_alpha_s_m2, settings.bool_dtype);
            let log_alpha_combined = B::float_mask_where(bar, l_prime_mask.clone(), bar_with_skip);

            // Slice row t from the pre-gathered `[T, N, 2S+1]` tensor.
            let log_probs_at_l = B::float_reshape(
                B::float_slice(
                    log_probs_at_l_full.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, max_l_prime_len]),
            );
            let new_alpha = B::float_add(log_alpha_combined, log_probs_at_l);
            log_alpha = B::float_mask_where(log_alpha, combined_mask, new_alpha);

            let log_alpha_3d = B::float_reshape(
                log_alpha.clone(),
                Shape::new([1, batch_size, max_l_prime_len]),
            );
            alpha_full = B::float_slice_assign(
                alpha_full,
                &[
                    Slice::new(t as isize, Some(t as isize + 1), 1),
                    Slice::full(),
                    Slice::full(),
                ],
                log_alpha_3d,
            );
        }

        Self {
            full: alpha_full,
            last: log_alpha,
            blank_inserted_targets,
            log_probs_at_l_full,
            max_l_prime_len,
        }
    }
}

/// Extract the per-sample loss from the last alpha row.
fn extract_loss<B: Backend>(alpha: &AlphaCtx<B>, target_lengths: IntTensor<B>) -> FloatTensor<B> {
    let log_alpha_shape = alpha.last.shape();
    let [batch_size, _] = log_alpha_shape.dims::<2>();
    let device = B::float_device(&alpha.last);
    let settings = get_device_settings::<B>(&device);

    let last_blank_idx = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let last_blank_idx = B::int_reshape(last_blank_idx, Shape::new([batch_size, 1]));
    let last_label_idx = B::int_clamp_min(
        B::int_sub_scalar(last_blank_idx.clone(), 1.into()),
        0.into(),
    );

    let log_alpha_last_blank = B::float_gather(1, alpha.last.clone(), last_blank_idx);
    let log_alpha_last_blank = B::float_reshape(log_alpha_last_blank, Shape::new([batch_size]));

    let log_alpha_last_label = B::float_gather(1, alpha.last.clone(), last_label_idx);
    let log_alpha_last_label = B::float_reshape(log_alpha_last_label, Shape::new([batch_size]));

    // For target_lengths == 0, last_label is meaningless: substitute -inf.
    let target_len_zero = B::int_equal_elem(target_lengths, 0.into(), settings.bool_dtype);
    let log_alpha_last_label = B::float_mask_fill(
        log_alpha_last_label,
        target_len_zero,
        f32::NEG_INFINITY.into(),
    );

    let log_likelihood = log_sum_exp::<B>(
        log_alpha_last_blank,
        log_alpha_last_label,
        settings.bool_dtype,
    );
    B::float_neg(log_likelihood)
}

/// Compute the full beta tensor `[T, N, 2S+1]` via reverse recursion.
fn compute_log_beta_full<B: Backend>(
    log_probs: FloatTensor<B>,
    alpha: &AlphaCtx<B>,
    input_lengths: IntTensor<B>,
    target_lengths: IntTensor<B>,
) -> FloatTensor<B> {
    let log_probs_shape = log_probs.shape();
    let [max_input_length, batch_size, _num_classes] = log_probs_shape.dims::<3>();
    let max_l_prime_len = alpha.max_l_prime_len;
    let device = B::float_device(&log_probs);
    let float_dtype: burn_std::FloatDType = log_probs.dtype().into();
    let settings = get_device_settings::<B>(&device);
    let blank_inserted_targets = &alpha.blank_inserted_targets;

    // Pre-allocate beta [T, N, 2S+1] filled with -inf.
    let mut beta_full = B::float_full(
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
        f32::NEG_INFINITY.into(),
        &device,
        float_dtype,
    );

    // Initialize beta at t = T-1 - 1 (the last valid timestep per batch). Because
    // input_lengths can vary per batch, beta is initialized at index
    // input_lengths[n] - 1 for each n. We achieve this by running a forward sweep
    // that only updates beta at the boundary t == input_length - 1.
    //
    // For the single-length case (all input_lengths equal), this just sets row
    // T-1 directly. For variable lengths, beta at indices >= input_length stays
    // at -inf, and beta at indices < input_length is filled by the recursion
    // started at input_length - 1.
    //
    // Implementation: at each timestep t, if input_length[n] - 1 == t, initialize
    // beta[t, n, 2U] and beta[t, n, 2U-1] from log_probs[t, n, ...]. Otherwise
    // use the recursion from beta[t+1].

    // Build a per-batch "boundary mask": true at the s positions {2U, 2U-1}.
    let last_blank_idx = B::int_mul_scalar(target_lengths.clone(), 2.into()); // [N]
    let last_label_idx = B::int_clamp_min(
        B::int_sub_scalar(last_blank_idx.clone(), 1.into()),
        0.into(),
    );
    let last_blank_idx_2d = B::int_reshape(last_blank_idx.clone(), Shape::new([batch_size, 1]));
    let last_blank_idx_b =
        B::int_expand(last_blank_idx_2d, Shape::new([batch_size, max_l_prime_len]));
    let last_label_idx_2d = B::int_reshape(last_label_idx.clone(), Shape::new([batch_size, 1]));
    let last_label_idx_b =
        B::int_expand(last_label_idx_2d, Shape::new([batch_size, max_l_prime_len]));

    let col_indices = B::int_arange(0..max_l_prime_len as i64, &device, settings.int_dtype);
    let col_indices_2d = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices_b = B::int_expand(col_indices_2d, Shape::new([batch_size, max_l_prime_len]));

    let is_last_blank = B::int_equal(col_indices_b.clone(), last_blank_idx_b, settings.bool_dtype);
    let is_last_label = B::int_equal(col_indices_b, last_label_idx_b, settings.bool_dtype);
    let is_boundary = B::bool_or(is_last_blank, is_last_label.clone());
    // For target_lengths == 0, last_label_idx was clamped to 0, which would
    // double-mark s=0 as boundary; suppress the last_label boundary in that case.
    let target_len_zero = B::int_equal_elem(target_lengths.clone(), 0.into(), settings.bool_dtype);
    let target_len_zero_b = B::bool_expand(
        B::bool_reshape(target_len_zero, Shape::new([batch_size, 1])),
        Shape::new([batch_size, max_l_prime_len]),
    );
    let is_last_label_real = B::bool_and(is_last_label, B::bool_not(target_len_zero_b));
    let _ = is_last_label_real; // already in is_boundary via union

    // Reverse loop: t = T-1, T-2, ..., 0.
    let mut log_beta = B::float_full(
        Shape::new([batch_size, max_l_prime_len]),
        f32::NEG_INFINITY.into(),
        &device,
        float_dtype,
    );

    let l_prime_mask_skip = create_l_prime_skip_forward_mask::<B>(
        blank_inserted_targets,
        batch_size,
        max_l_prime_len,
        &device,
        settings.int_dtype,
        settings.bool_dtype,
    );
    let s_mask = create_s_mask::<B>(
        &target_lengths,
        batch_size,
        max_l_prime_len,
        &device,
        settings.int_dtype,
        settings.bool_dtype,
    );

    // Hoist out of the T-loop: padding tensors for left_shift, the full
    // `[T, N, 2S+1]` gather of log_probs at l' (one big gather replaces T
    // small ones), and per-t boundary/validity masks built once as
    // `[T, N]` bool tensors and sliced per iteration.
    let pad_1 = B::float_full(
        Shape::new([batch_size, 1]),
        f32::NEG_INFINITY.into(),
        &device,
        float_dtype,
    );
    let pad_2 = B::float_full(
        Shape::new([batch_size, 2]),
        f32::NEG_INFINITY.into(),
        &device,
        float_dtype,
    );
    // Reuse the [T, N, 2S+1] gather computed once in AlphaCtx::compute.
    let log_probs_at_l_full = alpha.log_probs_at_l_full.clone();

    // `t_indices`: [T, 1]. `input_lengths`: [1, N]. Broadcast to [T, N].
    let t_indices_all = B::int_reshape(
        B::int_arange(0..max_input_length as i64, &device, settings.int_dtype),
        Shape::new([max_input_length, 1]),
    );
    let t_indices_all = B::int_expand(t_indices_all, Shape::new([max_input_length, batch_size]));
    let il_2d = B::int_expand(
        B::int_reshape(input_lengths.clone(), Shape::new([1, batch_size])),
        Shape::new([max_input_length, batch_size]),
    );
    // boundary_t_all[t, n] = (t == input_lengths[n] - 1), via t + 1 == il.
    let t_plus_one = B::int_add_scalar(t_indices_all.clone(), 1.into());
    let boundary_t_all = B::int_equal(t_plus_one, il_2d.clone(), settings.bool_dtype);
    let t_valid_all = B::int_greater(il_2d, t_indices_all, settings.bool_dtype);

    for t_rev in 0..max_input_length {
        let t = max_input_length - 1 - t_rev;

        // Slice log_probs[t, n, l'[s]] = [N, 2S+1] from the pre-gathered tensor.
        let log_probs_at_l = B::float_reshape(
            B::float_slice(
                log_probs_at_l_full.clone(),
                &[
                    Slice::new(t as isize, Some(t as isize + 1), 1),
                    Slice::full(),
                    Slice::full(),
                ],
            ),
            Shape::new([batch_size, max_l_prime_len]),
        );

        // Slice the hoisted [T, N] bool mask for this timestep and broadcast
        // straight to [N, 2S+1]. The intermediate [1, N] slice is reshaped
        // directly to [N, 1] (no separate [N] step).
        let boundary_t = B::bool_expand(
            B::bool_reshape(
                B::bool_slice(
                    boundary_t_all.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, 1]),
            ),
            Shape::new([batch_size, max_l_prime_len]),
        );

        // Boundary initialization: where t == input_length - 1 AND s in {2U, 2U-1},
        // set beta[t, n, s] = log_probs[t, n, l'[s]]. Elsewhere zero.
        let boundary_init_mask = B::bool_and(boundary_t, is_boundary.clone());
        let boundary_init = B::float_mask_fill(
            log_probs_at_l.clone(),
            B::bool_not(boundary_init_mask.clone()),
            f32::NEG_INFINITY.into(),
        );

        // Recursion contribution: from beta_{t+1}.
        // Only valid where t < input_length - 1 (beta from the future).
        // Compute log_sum_exp(beta[t+1, s], beta[t+1, s+1], beta[t+1, s+2] if skip).
        let beta_s = log_beta.clone();
        let beta_s_p1 = left_shift::<B>(&log_beta, &pad_1, max_l_prime_len, 1);
        let beta_s_p2 = left_shift::<B>(&log_beta, &pad_2, max_l_prime_len, 2);
        let bar = log_sum_exp::<B>(beta_s, beta_s_p1, settings.bool_dtype);
        let bar_with_skip = log_sum_exp::<B>(bar.clone(), beta_s_p2, settings.bool_dtype);
        let bar_combined = B::float_mask_where(bar, l_prime_mask_skip.clone(), bar_with_skip);
        let recursion_val = B::float_add(bar_combined, log_probs_at_l);

        // Combine: if boundary, use boundary_init; else use recursion_val.
        // Then apply s_mask (suppress invalid s) and t-validity (t < input_length).
        let chosen = B::float_mask_where(recursion_val, boundary_init_mask, boundary_init);

        // Mask out s positions beyond 2*target_len + 1.
        let chosen = B::float_mask_fill(
            chosen,
            B::bool_not(s_mask.clone()),
            f32::NEG_INFINITY.into(),
        );
        // Mask out t positions beyond input_length (hoisted slice).
        let t_valid = B::bool_expand(
            B::bool_reshape(
                B::bool_slice(
                    t_valid_all.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, 1]),
            ),
            Shape::new([batch_size, max_l_prime_len]),
        );
        log_beta = B::float_mask_fill(chosen, B::bool_not(t_valid), f32::NEG_INFINITY.into());

        let log_beta_3d = B::float_reshape(
            log_beta.clone(),
            Shape::new([1, batch_size, max_l_prime_len]),
        );
        beta_full = B::float_slice_assign(
            beta_full,
            &[
                Slice::new(t as isize, Some(t as isize + 1), 1),
                Slice::full(),
                Slice::full(),
            ],
            log_beta_3d,
        );
    }

    beta_full
}

/// Insert blank labels between each target label: [b, l1, b, l2, ..., b]
fn insert_blanks<B: Backend>(
    targets: &IntTensor<B>,
    batch_size: usize,
    max_target_len: usize,
    max_l_prime_len: usize,
    blank: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
) -> IntTensor<B> {
    let result = B::int_full(
        Shape::new([batch_size, max_l_prime_len]),
        (blank as i64).into(),
        device,
        int_dtype,
    );

    if max_target_len == 0 {
        return result;
    }

    // Place every target label at odd columns {1, 3, 5, ...} in one
    // strided slice_assign, equivalent to `result[:, 1::2] = targets`.
    B::int_slice_assign(
        result,
        &[Slice::full(), Slice::new(1, None, 2)],
        targets.clone(),
    )
}

/// Right-shift a 2D float tensor by `shift` positions, prepending the
/// pre-allocated `padding` tensor (shape `[batch_size, shift]`, value
/// `-inf`) instead of materializing it each call.
///
/// Called inside the T-loop of the alpha recursion; hoisting the padding
/// out of the loop eliminates `O(T)` `float_full` allocations.
fn right_shift<B: Backend>(
    tensor: &FloatTensor<B>,
    padding: &FloatTensor<B>,
    cols: usize,
    shift: usize,
) -> FloatTensor<B> {
    // Shifting by more than the column count pushes every data slot off
    // the right. Avoid the `cols - shift` usize underflow when
    // `max_target_len == 0` (so `max_l_prime_len == 1`) by narrowing the
    // all-`-inf` padding down to `cols`.
    if cols < shift {
        return B::float_slice(
            padding.clone(),
            &[Slice::full(), Slice::new(0, Some(cols as isize), 1)],
        );
    }
    let shortened = B::float_slice(
        tensor.clone(),
        &[
            Slice::full(),
            Slice::new(0, Some((cols - shift) as isize), 1),
        ],
    );
    B::float_cat(alloc::vec![padding.clone(), shortened], 1)
}

/// Left-shift a 2D float tensor by `shift` positions, appending the
/// pre-allocated `padding` tensor (shape `[batch_size, shift]`, value
/// `-inf`). Mirror of `right_shift` for the beta recursion.
fn left_shift<B: Backend>(
    tensor: &FloatTensor<B>,
    padding: &FloatTensor<B>,
    cols: usize,
    shift: usize,
) -> FloatTensor<B> {
    // Same degenerate-case guard as right_shift.
    if cols < shift {
        return B::float_slice(
            padding.clone(),
            &[Slice::full(), Slice::new(0, Some(cols as isize), 1)],
        );
    }
    let shortened = B::float_slice(
        tensor.clone(),
        &[Slice::full(), Slice::new(shift as isize, None, 1)],
    );
    B::float_cat(alloc::vec![shortened, padding.clone()], 1)
}

/// Compute log(exp(a) + exp(b)) in a numerically stable way.
///
/// `log_sum_exp(a, b) = max(a, b) + log1p(exp(-|a - b|))`. The only edge
/// case is `a = b = -inf`, where `-|(-inf) - (-inf)| = NaN`. We detect
/// `max == -inf` and substitute `0` for the diff there; the final sum
/// stays `-inf` because `-inf + log(2) = -inf`.
///
/// Handles -inf correctly and is safe for gradient computation:
/// - Both -inf: returns -inf
/// - One -inf, one finite: returns the finite value
/// - Both finite: standard log-sum-exp
///
/// Assumes inputs are `<= 0` (log-probabilities). `+inf` inputs produce
/// NaN (the `+inf + -inf` intermediate in the diff), but log-probs never
/// exceed `0` in a well-formed CTC input.
fn log_sum_exp<B: Backend>(
    a: FloatTensor<B>,
    b: FloatTensor<B>,
    bool_dtype: burn_std::BoolDType,
) -> FloatTensor<B> {
    // `-inf` values in `a` or `b` would make `a - b` evaluate to `NaN`
    // (when both are `-inf`) and the backward pass through that `NaN`
    // intermediate propagates `NaN` into the gradient even when the
    // forward mask discards it (`0 * NaN = NaN` in IEEE). Clamp `-inf`
    // to `0` on safe copies used only for the diff computation; compute
    // `max` on the original values so its output is correct in the
    // `-inf` cases.
    let a_is_neg_inf = B::float_equal_elem(a.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let b_is_neg_inf = B::float_equal_elem(b.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let either_neg_inf = B::bool_or(a_is_neg_inf.clone(), b_is_neg_inf.clone());

    let a_safe = B::float_mask_fill(a.clone(), a_is_neg_inf, 0.0.into());
    let b_safe = B::float_mask_fill(b.clone(), b_is_neg_inf, 0.0.into());

    let lt_mask = B::float_lower(a.clone(), b.clone(), bool_dtype);
    let mx = B::float_mask_where(a, lt_mask, b);

    // diff_safe = -|a_safe - b_safe|. Finite by construction. When either
    // input was `-inf`, force it to `-inf` so `exp(diff) == 0` and the
    // `log1p` term contributes nothing (`result = mx`). When both were
    // `-inf`, `mx = -inf` so `result = -inf + 0 = -inf`.
    let diff_safe = B::float_neg(B::float_abs(B::float_sub(a_safe, b_safe)));
    let diff_final = B::float_mask_fill(diff_safe, either_neg_inf, f32::NEG_INFINITY.into());

    B::float_add(mx, B::float_log1p(B::float_exp(diff_final)))
}

/// Mask for the alpha skip transition: `l'[s] != blank AND l'[s] != l'[s-2] AND s >= 2`.
fn create_l_prime_mask<B: Backend>(
    blank_inserted_targets: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    blank: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
    bool_dtype: burn_std::BoolDType,
) -> <B as Backend>::BoolTensorPrimitive {
    // The mask requires `s >= 2`, which is unsatisfiable when max_l_prime_len < 2
    // (i.e. targets have shape [N, 0]). Bail out before the `max_l_prime_len - 2`
    // usize subtraction underflows.
    if max_l_prime_len < 2 {
        return B::bool_zeros(
            Shape::new([batch_size, max_l_prime_len]),
            device,
            bool_dtype,
        );
    }
    let l_prime = blank_inserted_targets.clone();

    let not_blank = B::int_not_equal_elem(l_prime.clone(), (blank as i64).into(), bool_dtype);

    let l_prime_shifted = {
        let padding = B::int_full(
            Shape::new([batch_size, 2]),
            (blank as i64).into(),
            device,
            int_dtype,
        );
        let shortened = B::int_slice(
            l_prime.clone(),
            &[
                Slice::full(),
                Slice::new(0, Some((max_l_prime_len - 2) as isize), 1),
            ],
        );
        B::int_cat(alloc::vec![padding, shortened], 1)
    };
    let not_equal_s_m2 = B::int_not_equal(l_prime, l_prime_shifted, bool_dtype);

    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));
    let s_ge_2 = B::int_greater_equal_elem(col_indices, 2.into(), bool_dtype);

    B::bool_and(B::bool_and(not_blank, not_equal_s_m2), s_ge_2)
}

/// Mask for the beta skip transition: `l'[s] != blank AND l'[s] != l'[s+2] AND s+2 < L'`.
/// This is the symmetric counterpart of `create_l_prime_mask` for the beta recursion.
fn create_l_prime_skip_forward_mask<B: Backend>(
    blank_inserted_targets: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
    bool_dtype: burn_std::BoolDType,
) -> <B as Backend>::BoolTensorPrimitive {
    // No `s + 2 < L'` position exists when L' < 2, so the skip mask is
    // unconditionally false. Also avoids underflow in the `L' - 2` slice below.
    if max_l_prime_len < 2 {
        return B::bool_zeros(
            Shape::new([batch_size, max_l_prime_len]),
            device,
            bool_dtype,
        );
    }
    let l_prime = blank_inserted_targets.clone();

    // l'[s+2]: left-shift l' by 2 positions, padding the last two with `l'[s]` so
    // the equality check at out-of-range positions degenerates to `l'[s] == l'[s]`
    // (skip not allowed - which is what we want at s+2 >= L').
    let l_prime_left = {
        let padding_src = B::int_slice(
            l_prime.clone(),
            &[
                Slice::full(),
                Slice::new(
                    (max_l_prime_len - 2) as isize,
                    Some(max_l_prime_len as isize),
                    1,
                ),
            ],
        );
        let shortened = B::int_slice(l_prime.clone(), &[Slice::full(), Slice::new(2, None, 1)]);
        B::int_cat(alloc::vec![shortened, padding_src], 1)
    };
    let not_equal_s_p2 = B::int_not_equal(l_prime.clone(), l_prime_left, bool_dtype);

    // s + 2 < L'
    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));
    let s_lt_l_prime_minus_2 =
        B::int_lower_elem(col_indices, (max_l_prime_len as i64 - 2).into(), bool_dtype);

    // l'[s] != blank: needed - the skip transition is allowed only when the
    // current position is a label, not blank.
    let blank_int = blank_inserted_targets.clone();
    // Use 0 below as a placeholder; cubecl backends will pick this up via `_blank`.
    // Actually we need the blank value here. The caller has it. Re-derive from l'.
    let _ = blank_int;
    // Workaround: reuse the alpha mask logic by checking against the first blank
    // position, which is l'[0] by construction.
    let blank_class_per_row = B::int_slice(
        blank_inserted_targets.clone(),
        &[Slice::full(), Slice::new(0, Some(1), 1)],
    );
    let blank_class_per_row = B::int_expand(
        blank_class_per_row,
        Shape::new([batch_size, max_l_prime_len]),
    );
    let not_blank = B::int_not_equal(l_prime, blank_class_per_row, bool_dtype);

    B::bool_and(B::bool_and(not_blank, not_equal_s_p2), s_lt_l_prime_minus_2)
}

/// Create a mask for valid s positions: s < 2 * target_length + 1
fn create_s_mask<B: Backend>(
    target_lengths: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    device: &B::Device,
    int_dtype: burn_std::IntDType,
    bool_dtype: burn_std::BoolDType,
) -> <B as Backend>::BoolTensorPrimitive {
    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));

    let lengths = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let lengths = B::int_add_scalar(lengths, 1.into());
    let lengths = B::int_reshape(lengths, Shape::new([batch_size, 1]));
    let lengths = B::int_expand(lengths, Shape::new([batch_size, max_l_prime_len]));

    B::int_lower(col_indices, lengths, bool_dtype)
}
