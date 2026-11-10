"""Fixed-center, full-tail Gaussian spatial operators for the shared R22 field."""
from __future__ import annotations
from functools import lru_cache
import math
import torch

DEFAULT_SIGMA = .7

def configuration(config=None):
    sigma = float(dict(config or {}).get("sigma", DEFAULT_SIGMA))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    return {"sigma": sigma}


def _shape(shape):
    if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
        raise ValueError("Expected a positive (height,width) spatial shape")
    return tuple(shape)



@lru_cache(maxsize=16)
def _grid_matrices(shape, sigma, device, dtype):
    # These are fixed geometry-only constants, not subject fields or signals.
    matrices = []
    for length in shape:
        axis = torch.arange(length, device=device, dtype=dtype)
        difference = axis[:, None] - axis[None, :]
        matrices.append(torch.exp(-.5 * (difference / sigma).square()))
    return tuple(matrices)



def grid_kernel_matrices(shape, sigma=DEFAULT_SIGMA, *, device="cpu", dtype=torch.float32):
    """Return row and column Gaussian matrices; treat cached constants as read-only."""
    shape = _shape(shape)
    sigma = configuration({"sigma": sigma})["sigma"]
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Gaussian matrices require float32 or float64")
    return _grid_matrices(shape, sigma, str(torch.device(device)), dtype)



def _values(values):
    values = torch.as_tensor(values)
    if not values.is_floating_point() or values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    if values.ndim < 3:
        raise ValueError("Expected [...,height,width,features]")
    return values



def _occupancy(occupancy, values, *, validate=False):
    occupancy = torch.as_tensor(occupancy, device=values.device, dtype=values.dtype)
    if occupancy.shape != values.shape[:-1]:
        raise ValueError("Occupancy must have shape [...,height,width] matching coefficients")
    if validate and (not torch.isfinite(occupancy).all() or torch.any(occupancy < 0)
                     or torch.any(occupancy > 1)):
        raise ValueError("Occupancy must be finite in [0,1]")
    return occupancy



def gaussian_convolve(values, sigma=DEFAULT_SIGMA, *, transpose=False):
    """Apply separable full-tail Gaussian matrices to [...,H,W,D].

    ``transpose=True`` explicitly applies the adjoint. The present fixed,
    equal input/output grids make both matrices symmetric, but retaining the
    transpose documents the linear-operator contract for fitting routines.
    """
    values = _values(values)
    row, column = grid_kernel_matrices(tuple(values.shape[-3:-1]), sigma,
                                       device=values.device, dtype=values.dtype)
    if transpose:
        row, column = row.T, column.T
    with torch.autocast(device_type=values.device.type, enabled=False):
        feature_first = values.movedim(-1, -3)
        result = row @ feature_first @ column.T
        return result.movedim(-3, -1)



def coefficient_field(coefficients, occupancy, sigma=DEFAULT_SIGMA):
    """Gaussian coefficients -> full integer-grid coefficient field, differentiably."""
    coefficients = _values(coefficients)
    if coefficients.shape[-1] < 1:
        raise ValueError("Expected at least one coefficient per Gaussian")
    occupancy = _occupancy(occupancy, coefficients)
    return gaussian_convolve(coefficients * occupancy[..., None], sigma)



def coefficient_adjoint(field, occupancy, sigma=DEFAULT_SIGMA):
    """Exact Euclidean adjoint of coefficient_field for fixed occupancy."""
    field = _values(field)
    occupancy = _occupancy(occupancy, field)
    return gaussian_convolve(field, sigma, transpose=True) * occupancy[..., None]



def field_points(coefficients, occupancy, xy, sigma=DEFAULT_SIGMA, *, chunk_size=128):
    """Evaluate the same full-tail Gaussian field at continuous [N,2] row/column.

    Leading coefficient batch dimensions are supported; xy is shared across
    those batches. The operations remain differentiable in coefficients,
    occupancy and continuous spatial coordinates.
    """
    coefficients = _values(coefficients)
    if coefficients.shape[-1] < 1:
        raise ValueError("Expected at least one coefficient per Gaussian")
    occupancy = _occupancy(occupancy, coefficients)
    sigma = configuration({"sigma": sigma})["sigma"]
    xy = torch.as_tensor(xy, device=coefficients.device, dtype=coefficients.dtype)
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError("Expected continuous spatial points [N,2], ordered row,column")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    height, width = coefficients.shape[-3:-1]
    rows, columns = torch.meshgrid(torch.arange(height, device=xy.device, dtype=xy.dtype),
                                   torch.arange(width, device=xy.device, dtype=xy.dtype), indexing="ij")
    centers = torch.stack((rows, columns), -1).reshape(-1, 2)
    weighted = (coefficients * occupancy[..., None]).reshape(
        *coefficients.shape[:-3], -1, coefficients.shape[-1])
    result = []
    with torch.autocast(device_type=coefficients.device.type, enabled=False):
        for points in xy.split(chunk_size):
            difference = points[:, None] - centers[None]
            weights = torch.exp(-.5 * (difference / sigma).square().sum(-1))
            result.append(weights @ weighted)
    return torch.cat(result, dim=-2) if result else weighted[..., :0, :]
