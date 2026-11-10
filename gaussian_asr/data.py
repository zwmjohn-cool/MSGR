"""Acquired-only ASR problems and native-slice support handling."""
import numpy as np


def sparse_problem(case, observed_indices, *, with_mri=False):
    """Return a solver-only dictionary; hidden images/evaluation masks are absent."""
    indices = np.asarray(observed_indices, dtype=np.int64)
    if (indices.ndim != 1 or len(indices) != len(set(indices.tolist()))
            or not len(indices) or indices.min() < 0 or indices.max() >= len(case["bvals"])):
        raise ValueError("Acquired encoding indices must be unique and in range")
    problem = dict(
        id=case["id"], observed=case["measured"][indices].copy(),
        observed_indices=indices, directions=case["directions"].copy(),
        bvals=case["bvals"].copy(), b0=case["b0"].copy(),
        support=case["support"].copy(),
    )
    if with_mri:
        problem["fm_condition"] = case["fm_condition"].copy()
    return problem



def b0_supported_indices(b0_images):
    """Select optimizable slices using permitted B0 images only.

    Empty planes remain part of the native volume. Neither positive-b signals
    nor evaluator masks enter this decision.
    """
    return [index for index, image in enumerate(b0_images)
            if np.any(np.isfinite(np.asarray(image)) & (np.asarray(image) > 0))]



def reconstruct_b0_supported_batch(problems, predict_valid, *, query_count=270):
    """Predict supported slices and scatter into the unchanged native batch.

    ``predict_valid(problems, local_indices)`` receives only the nonempty-B0
    subbatch. It is never called for an all-empty batch. Empty-B0 slices get
    zero missing-direction predictions; callers subsequently restore their
    actually acquired measurements, just as for every other slice.
    """
    if not problems or not isinstance(query_count, int) or query_count < 1:
        raise ValueError("A native joint batch and positive query count are required")
    native_shape = np.asarray(problems[0]['b0']).shape
    if len(native_shape) != 2 or min(native_shape) < 1 or any(
            np.asarray(problem['b0']).shape != native_shape for problem in problems):
        raise ValueError("Joint-batch B0 images must have one matching native H,W shape")
    valid = b0_supported_indices([problem['b0'] for problem in problems])
    valid_set = set(valid)
    empty = [index for index in range(len(problems)) if index not in valid_set]
    expected_shape = (len(valid), query_count, *native_shape)
    if valid:
        predicted_valid = np.asarray(predict_valid([problems[index] for index in valid], valid), dtype=np.float32)
        if predicted_valid.shape != expected_shape or not np.isfinite(predicted_valid).all():
            raise ValueError("Malformed supported joint reconstruction")
        if not empty:
            # Preserve the exact all-supported path, including batch order.
            prediction = predicted_valid
        else:
            prediction = np.zeros((len(problems), query_count, *native_shape), np.float32)
            prediction[valid] = predicted_valid
    else:
        prediction = np.zeros((len(problems), query_count, *native_shape), np.float32)
    partition = dict(valid_indices=valid, empty_indices=empty,
                     empty_case_ids=[problems[index]['id'] for index in empty])
    return prediction, partition
