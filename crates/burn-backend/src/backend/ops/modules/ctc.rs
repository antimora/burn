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
    let log_probs_shape = log_probs.shape();
    let [max_input_length, batch_size, num_classes] = log_probs_shape.dims::<3>();
    let device = B::float_device(&log_probs);
    let settings = get_device_settings::<B>(&device);

    // Compute alpha (full history) and beta in log space.
    let alpha = AlphaCtx::<B>::compute(
        log_probs.clone(),
        &targets,
        input_lengths.clone(),
        target_lengths.clone(),
        blank,
    );
    let log_alpha_full = alpha.full.clone();
    let max_l_prime_len = alpha.max_l_prime_len;
    let blank_inserted_targets = alpha.blank_inserted_targets.clone();
    let log_beta_full = compute_log_beta_full::<B>(
        log_probs.clone(),
        &alpha,
        input_lengths.clone(),
        target_lengths.clone(),
    );

    // Per-sample negative log-likelihood (== forward loss).
    let nll = extract_loss::<B>(&alpha, target_lengths.clone());

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
    let grad_loss_b = B::float_reshape(grad_loss, Shape::new([1, batch_size, 1]));
    let grad_loss_b = B::float_expand(
        grad_loss_b,
        Shape::new([max_input_length, batch_size, num_classes]),
    );
    let mut grad = B::float_mul(B::float_exp(log_probs), grad_loss_b.clone());

    // Subtract sum over s of grad_loss[n] * exp(log_post[t, n, s]) at index l'[n, s].
    let grad_loss_post = B::float_reshape(
        B::float_slice(
            grad_loss_b.clone(),
            &[Slice::full(), Slice::full(), Slice::new(0, Some(1), 1)],
        ),
        Shape::new([max_input_length, batch_size, 1]),
    );
    let grad_loss_post = B::float_expand(
        grad_loss_post,
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
    B::float_mask_fill(grad, oob_mask, 0.0.into())
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

        for t in 1..max_input_length {
            let t_mask_1d = B::int_greater_elem(
                input_lengths.clone(),
                (t as i64).into(),
                settings.bool_dtype,
            );
            let t_mask = B::bool_expand(
                B::bool_reshape(t_mask_1d, Shape::new([batch_size, 1])),
                Shape::new([batch_size, max_l_prime_len]),
            );
            let combined_mask = B::bool_and(t_mask, s_mask.clone());

            let log_alpha_s = log_alpha.clone();
            let log_alpha_s_m1 = right_shift::<B>(
                &log_alpha,
                batch_size,
                max_l_prime_len,
                1,
                &device,
                float_dtype,
            );
            let log_alpha_s_m2 = right_shift::<B>(
                &log_alpha,
                batch_size,
                max_l_prime_len,
                2,
                &device,
                float_dtype,
            );

            let bar = log_sum_exp::<B>(log_alpha_s, log_alpha_s_m1, settings.bool_dtype);
            let bar_with_skip = log_sum_exp::<B>(bar.clone(), log_alpha_s_m2, settings.bool_dtype);
            let log_alpha_combined = B::float_mask_where(bar, l_prime_mask.clone(), bar_with_skip);

            let log_probs_t = B::float_slice(
                log_probs.clone(),
                &[
                    Slice::new(t as isize, Some(t as isize + 1), 1),
                    Slice::full(),
                    Slice::full(),
                ],
            );
            let log_probs_t = B::float_reshape(log_probs_t, Shape::new([batch_size, num_classes]));
            let log_probs_at_l = B::float_gather(1, log_probs_t, blank_inserted_targets.clone());
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
    let [max_input_length, batch_size, num_classes] = log_probs_shape.dims::<3>();
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

    for t_rev in 0..max_input_length {
        let t = max_input_length - 1 - t_rev;

        // log_probs[t, :, :] -> [N, C]
        let log_probs_t = B::float_slice(
            log_probs.clone(),
            &[
                Slice::new(t as isize, Some(t as isize + 1), 1),
                Slice::full(),
                Slice::full(),
            ],
        );
        let log_probs_t = B::float_reshape(log_probs_t, Shape::new([batch_size, num_classes]));
        let log_probs_at_l = B::float_gather(1, log_probs_t, blank_inserted_targets.clone()); // [N, 2S+1]

        // Detect boundary: input_length[n] - 1 == t (per-batch).
        let boundary_t_1d = B::int_equal_elem(
            B::int_sub_scalar(input_lengths.clone(), 1.into()),
            (t as i64).into(),
            settings.bool_dtype,
        );
        let boundary_t = B::bool_expand(
            B::bool_reshape(boundary_t_1d.clone(), Shape::new([batch_size, 1])),
            Shape::new([batch_size, max_l_prime_len]),
        );

        // Boundary initialization: where t == input_length - 1 AND s in {2U, 2U-1},
        // set beta[t, n, s] = log_probs[t, n, l'[s]]. Elsewhere zero.
        let boundary_init_mask = B::bool_and(boundary_t.clone(), is_boundary.clone());
        let boundary_init = B::float_mask_fill(
            log_probs_at_l.clone(),
            B::bool_not(boundary_init_mask.clone()),
            f32::NEG_INFINITY.into(),
        );

        // Recursion contribution: from beta_{t+1}.
        // Only valid where t < input_length - 1 (beta from the future).
        // Compute log_sum_exp(beta[t+1, s], beta[t+1, s+1], beta[t+1, s+2] if skip).
        let beta_s = log_beta.clone();
        let beta_s_p1 = left_shift::<B>(
            &log_beta,
            batch_size,
            max_l_prime_len,
            1,
            &device,
            float_dtype,
        );
        let beta_s_p2 = left_shift::<B>(
            &log_beta,
            batch_size,
            max_l_prime_len,
            2,
            &device,
            float_dtype,
        );
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
        // Mask out t positions beyond input_length.
        let t_valid_1d = B::int_greater_elem(
            input_lengths.clone(),
            (t as i64).into(),
            settings.bool_dtype,
        );
        let t_valid = B::bool_expand(
            B::bool_reshape(t_valid_1d, Shape::new([batch_size, 1])),
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
    let mut result = B::int_full(
        Shape::new([batch_size, max_l_prime_len]),
        (blank as i64).into(),
        device,
        int_dtype,
    );

    // Place labels at odd indices: 1, 3, 5, ...
    for s in 0..max_target_len {
        let label = B::int_slice(
            targets.clone(),
            &[
                Slice::full(),
                Slice::new(s as isize, Some(s as isize + 1), 1),
            ],
        );
        let dest_col = 2 * s + 1;
        result = B::int_slice_assign(
            result,
            &[
                Slice::full(),
                Slice::new(dest_col as isize, Some(dest_col as isize + 1), 1),
            ],
            label,
        );
    }

    result
}

/// Right-shift a 2D float tensor by `shift` positions, filling the leading
/// columns with `f32::NEG_INFINITY` cast to the requested float dtype.
fn right_shift<B: Backend>(
    tensor: &FloatTensor<B>,
    batch_size: usize,
    cols: usize,
    shift: usize,
    device: &B::Device,
    float_dtype: burn_std::FloatDType,
) -> FloatTensor<B> {
    let padding = B::float_full(
        Shape::new([batch_size, shift]),
        f32::NEG_INFINITY.into(),
        device,
        float_dtype,
    );
    let shortened = B::float_slice(
        tensor.clone(),
        &[
            Slice::full(),
            Slice::new(0, Some((cols - shift) as isize), 1),
        ],
    );
    B::float_cat(alloc::vec![padding, shortened], 1)
}

/// Left-shift a 2D float tensor by `shift` positions, filling the trailing
/// columns with `f32::NEG_INFINITY` cast to the requested float dtype.
/// Used by the beta recursion to read `beta[t+1, s+1]` and `beta[t+1, s+2]`.
fn left_shift<B: Backend>(
    tensor: &FloatTensor<B>,
    batch_size: usize,
    _cols: usize,
    shift: usize,
    device: &B::Device,
    float_dtype: burn_std::FloatDType,
) -> FloatTensor<B> {
    let padding = B::float_full(
        Shape::new([batch_size, shift]),
        f32::NEG_INFINITY.into(),
        device,
        float_dtype,
    );
    let shortened = B::float_slice(
        tensor.clone(),
        &[Slice::full(), Slice::new(shift as isize, None, 1)],
    );
    B::float_cat(alloc::vec![shortened, padding], 1)
}

/// Compute log(exp(a) + exp(b)) in a numerically stable way.
///
/// Handles -inf correctly and is safe for gradient computation:
/// - Both -inf: returns -inf
/// - One -inf, one finite: returns the finite value
/// - Both finite: standard log-sum-exp
fn log_sum_exp<B: Backend>(
    a: FloatTensor<B>,
    b: FloatTensor<B>,
    bool_dtype: burn_std::BoolDType,
) -> FloatTensor<B> {
    let a_is_neg_inf = B::float_equal_elem(a.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let b_is_neg_inf = B::float_equal_elem(b.clone(), f32::NEG_INFINITY.into(), bool_dtype);

    let ones = B::float_ones(a.shape(), &B::float_device(&a), a.dtype().into());
    let fallback = B::float_mask_where(ones.clone(), a_is_neg_inf.clone(), b.clone());
    let fallback = B::float_mask_where(fallback, b_is_neg_inf.clone(), a.clone());

    let a_safe = B::float_mask_fill(a, a_is_neg_inf.clone(), 0.0.into());
    let b_safe = B::float_mask_fill(b, b_is_neg_inf.clone(), 0.0.into());

    let lt_mask = B::float_lower(a_safe.clone(), b_safe.clone(), bool_dtype);
    let max_val = B::float_mask_where(a_safe.clone(), lt_mask, b_safe.clone());
    let diff = B::float_sub(a_safe, b_safe);
    let exp_neg_abs_diff = B::float_exp(B::float_neg(B::float_abs(diff)));
    let lse = B::float_add(max_val, B::float_log(B::float_add(ones, exp_neg_abs_diff)));

    let either_inf = B::bool_or(a_is_neg_inf, b_is_neg_inf);
    let neither_inf = B::bool_not(either_inf);
    B::float_mask_where(fallback, neither_inf, lse)
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
