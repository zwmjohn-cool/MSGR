"""Validation of the frozen teacher-fitting controls used by the shared field."""
import math
from numbers import Integral
import numpy as np
import torch

DIMENSION = 84
DEFAULT_CONFIG = dict(width=2., smooth=.0005, mp_blend=.23,
    group_size=144, rank=14, raw_refit_blend=0.,
    patch_size=9, noise_factor=1., patch_offsets=4,
    denoise_backend="auto", denoise_batch_size=256,
    parameter_ratio_limit=.5, output_dtype="float64")


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)



def configuration(config=None):
    """Validate fit controls while preserving caller-owned keys such as sigma."""
    source = dict(config or {})
    for old, new in (("radial_width", "width"), ("denoise_patch_size", "patch_size"),
                     ("denoise_noise_factor", "noise_factor"), ("denoise_offsets", "patch_offsets")):
        if old in source:
            if new in source and source[new] != source[old]:
                raise ValueError(f"Conflicting {old} and {new} configuration")
            source[new] = source.pop(old)
    cfg = dict(DEFAULT_CONFIG, **source)
    cfg["group_size"] = _positive_integer(cfg["group_size"], "group_size")
    cfg["rank"] = _positive_integer(cfg["rank"], "rank")
    if cfg["rank"] > DIMENSION:
        raise ValueError("rank cannot exceed the 84-dimensional analytic dictionary")
    for key in ("width", "smooth", "mp_blend", "raw_refit_blend", "parameter_ratio_limit"):
        value = cfg[key]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a finite numeric value")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        cfg[key] = value
    if cfg["smooth"] < 0:
        raise ValueError("smooth must be nonnegative")
    if cfg["width"] <= 0:
        raise ValueError("width must be positive")
    if not 0 <= cfg["mp_blend"] <= 1 or not 0 <= cfg["raw_refit_blend"] <= 1:
        raise ValueError("mp_blend and raw_refit_blend must lie in [0,1]")
    if not 0 < cfg["parameter_ratio_limit"] <= .5:
        raise ValueError("parameter_ratio_limit must lie in (0,0.5]")
    if cfg["denoise_backend"] not in ("auto", "cpu", "torch"):
        raise ValueError("denoise_backend must be auto, cpu, or torch")
    if cfg["output_dtype"] not in ("float32", "float64", torch.float32, torch.float64):
        raise ValueError("output_dtype must be float32 or float64")
    # Validate even when MP is disabled, so unused invalid controls do not
    # quietly become accepted configuration values.
    cfg["patch_size"] = _positive_integer(cfg["patch_size"], "patch_size")
    cfg["denoise_batch_size"] = _positive_integer(cfg["denoise_batch_size"], "denoise_batch_size")
    cfg["patch_offsets"] = _positive_integer(cfg["patch_offsets"], "patch_offsets")
    if cfg["patch_offsets"] not in (1, 4):
        raise ValueError("patch_offsets must be 1 or 4")
    factor = cfg["noise_factor"]
    if isinstance(factor, (bool, np.bool_)) or not math.isfinite(float(factor)) or float(factor) < 0:
        raise ValueError("noise_factor must be finite and nonnegative")
    cfg["noise_factor"] = float(factor)
    return cfg
