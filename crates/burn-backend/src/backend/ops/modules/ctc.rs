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
    let shape = log_probs.shape();
    let [max_input_length, batch_size, num_classes] = shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let device = B::float_device(&log_probs);
    let float_dtype: burn_std::FloatDType = log_probs.dtype().into();
    let settings = get_device_settings::<B>(&device);

    // Build modified label sequence l' by inserting blanks: [b, l1, b, l2, b, ...]
    let max_l_prime_len = 2 * max_target_len + 1;
    let blank_inserted_targets = insert_blanks::<B>(
        &targets,
        batch_size,
        max_target_len,
        max_l_prime_len,
        blank,
        &device,
        settings.int_dtype,
    );

    // Initialize log_alpha[0, s] = -inf for all s, then set s=0 and s=1
    let mut log_alpha = B::float_full(
        Shape::new([batch_size, max_l_prime_len]),
        f32::NEG_INFINITY.into(),
        &device,
        float_dtype,
    );

    // log_probs[0] -> [N, C]
    let log_probs_t0 = B::float_slice(
        log_probs.clone(),
        &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
    );
    let log_probs_t0 = B::float_reshape(log_probs_t0, Shape::new([batch_size, num_classes]));

    // log_alpha[:, 0] = log_probs[0, :, blank]
    let first_blank = B::int_slice(
        blank_inserted_targets.clone(),
        &[Slice::full(), Slice::new(0, Some(1), 1)],
    ); // [N, 1]
    let log_prob_blank = B::float_gather(1, log_probs_t0.clone(), first_blank); // [N, 1]
    log_alpha = B::float_slice_assign(
        log_alpha,
        &[Slice::full(), Slice::new(0, Some(1), 1)],
        log_prob_blank,
    );

    // log_alpha[:, 1] = log_probs[0, :, l'[1]]
    if max_l_prime_len > 1 {
        let first_label = B::int_slice(
            blank_inserted_targets.clone(),
            &[Slice::full(), Slice::new(1, Some(2), 1)],
        ); // [N, 1]
        let log_prob_first = B::float_gather(1, log_probs_t0, first_label); // [N, 1]
        log_alpha = B::float_slice_assign(
            log_alpha,
            &[Slice::full(), Slice::new(1, Some(2), 1)],
            log_prob_first,
        );
    }

    // Precompute the combined mask for l'[s] != blank AND l'[s] != l'[s-2] AND s >= 2
    let l_prime_mask = create_l_prime_mask::<B>(
        &blank_inserted_targets,
        batch_size,
        max_l_prime_len,
        blank,
        &device,
        settings.int_dtype,
        settings.bool_dtype,
    );

    // Precompute s_mask: which s positions are valid given target_lengths
    let s_mask = create_s_mask::<B>(
        &target_lengths,
        batch_size,
        max_l_prime_len,
        &device,
        settings.int_dtype,
        settings.bool_dtype,
    );

    // Main loop over time steps
    for t in 1..max_input_length {
        // t_mask: which batch elements have input_length > t
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

        // alpha_{t-1}(s): no move
        let log_alpha_s = log_alpha.clone();

        // alpha_{t-1}(s-1): single move (right-shift by 1)
        let log_alpha_s_m1 = right_shift::<B>(
            &log_alpha,
            batch_size,
            max_l_prime_len,
            1,
            &device,
            float_dtype,
        );

        // alpha_{t-1}(s-2): skip move (right-shift by 2)
        let log_alpha_s_m2 = right_shift::<B>(
            &log_alpha,
            batch_size,
            max_l_prime_len,
            2,
            &device,
            float_dtype,
        );

        // log_sum_exp(alpha_s, alpha_s_m1)
        let bar = log_sum_exp::<B>(log_alpha_s, log_alpha_s_m1, settings.bool_dtype);

        // Conditionally add alpha_s_m2 where l_prime_mask is true
        let bar_with_skip = log_sum_exp::<B>(bar.clone(), log_alpha_s_m2, settings.bool_dtype);
        let log_alpha_combined = B::float_mask_where(bar, l_prime_mask.clone(), bar_with_skip);

        // Gather log_probs[t] at l' positions
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

        // new_alpha = log_alpha_combined + log_probs_at_l
        let new_alpha = B::float_add(log_alpha_combined, log_probs_at_l);

        // Apply mask: keep old log_alpha where mask is false
        log_alpha = B::float_mask_where(log_alpha, combined_mask, new_alpha);
    }

    // Extract final log-likelihood: log_sum_exp(alpha[2*U], alpha[2*U-1])
    let last_blank_idx = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let last_blank_idx = B::int_reshape(last_blank_idx, Shape::new([batch_size, 1]));
    // Guard target_lengths = 0: clamp last_label_idx to 0 to avoid an invalid
    // negative gather index, then mask the corresponding alpha value to -inf
    // so log_sum_exp falls back to last_blank for that batch element.
    let last_label_idx = B::int_clamp_min(
        B::int_sub_scalar(last_blank_idx.clone(), 1.into()),
        0.into(),
    );

    let log_alpha_last_blank = B::float_gather(1, log_alpha.clone(), last_blank_idx);
    let log_alpha_last_blank = B::float_reshape(log_alpha_last_blank, Shape::new([batch_size]));

    let log_alpha_last_label = B::float_gather(1, log_alpha, last_label_idx);
    let log_alpha_last_label = B::float_reshape(log_alpha_last_label, Shape::new([batch_size]));

    // Where target_lengths == 0, last_label is meaningless: substitute -inf so
    // log_sum_exp(last_blank, -inf) = last_blank.
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
        ); // [N, 1]
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

    // For positions where one is -inf, select the other.
    // For positions where both are -inf, this gives -inf (correct).
    let ones = B::float_ones(a.shape(), &B::float_device(&a), a.dtype().into());
    // Start with b for positions where a is -inf
    let fallback = B::float_mask_where(ones.clone(), a_is_neg_inf.clone(), b.clone());
    // Override with a for positions where b is -inf
    let fallback = B::float_mask_where(fallback, b_is_neg_inf.clone(), a.clone());

    // Sanitize -inf values to 0 so math ops don't produce NaN
    let a_safe = B::float_mask_fill(a, a_is_neg_inf.clone(), 0.0.into());
    let b_safe = B::float_mask_fill(b, b_is_neg_inf.clone(), 0.0.into());

    // Standard log-sum-exp on sanitized values
    let lt_mask = B::float_lower(a_safe.clone(), b_safe.clone(), bool_dtype);
    let max_val = B::float_mask_where(a_safe.clone(), lt_mask, b_safe.clone());
    let diff = B::float_sub(a_safe, b_safe);
    let exp_neg_abs_diff = B::float_exp(B::float_neg(B::float_abs(diff)));
    let lse = B::float_add(max_val, B::float_log(B::float_add(ones, exp_neg_abs_diff)));

    // Use lse only where neither input was -inf; otherwise use the fallback
    let either_inf = B::bool_or(a_is_neg_inf, b_is_neg_inf);
    let neither_inf = B::bool_not(either_inf);
    B::float_mask_where(fallback, neither_inf, lse)
}

/// Create a mask for positions where the skip transition (s-2) is valid:
/// l'[s] != blank AND l'[s] != l'[s-2] AND s >= 2
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

    // l'[s] != blank
    let not_blank = B::int_not_equal_elem(l_prime.clone(), (blank as i64).into(), bool_dtype);

    // l'[s] != l'[s-2]: right-shift l' by 2 and compare
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

    // s >= 2
    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));
    let s_ge_2 = B::int_greater_equal_elem(col_indices, 2.into(), bool_dtype);

    B::bool_and(B::bool_and(not_blank, not_equal_s_m2), s_ge_2)
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
