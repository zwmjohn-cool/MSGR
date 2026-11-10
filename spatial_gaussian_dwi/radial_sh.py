"""Continuous cardinal radial × real SH6 dictionary, independent of DIPY.

The 84 columns are radial-major: three Gaussian-cardinal radial functions,
each followed by the 28 symmetric Descoteaux07 (legacy=False) harmonics.
Cartesian harmonic polynomials keep direction derivatives finite at the poles.
This dictionary describes positive-b measurements; b=0 is deliberately not
supported and no positivity or monotonicity constraint is implied.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch


ANGULAR_DIMENSION = 28
DIMENSION = 84
RADIAL_CENTERS = (1., 2., 3.)
DEFAULT_WIDTH = 2.


def angular_columns():
    """DIPY symmetric SH6 order: ell=0,2,4,6, then m=-ell,...,+ell."""
    return [(ell, m) for ell in range(0, 7, 2) for m in range(-ell, ell + 1)]


def columns():
    """Stored column triples: (zero-based radial index, ell, m)."""
    return [(radial, ell, m) for radial in range(3) for ell, m in angular_columns()]


def configuration(config=None):
    width = float(dict(config or {}).get("radial_width", DEFAULT_WIDTH))
    if not math.isfinite(width) or width <= 0:
        raise ValueError("radial_width must be finite and positive")
    return {"radial_width": width}


@lru_cache(maxsize=1)
def _angular_numpy_constants():
    from numpy.polynomial import Legendre, Polynomial

    polynomial = np.zeros((7, ANGULAR_DIMENSION), dtype=np.float64)
    for index, (ell, m) in enumerate(angular_columns()):
        order = abs(m)
        derivative = Legendre.basis(ell).convert(kind=Polynomial).deriv(order).coef
        scale = math.sqrt((2 * ell + 1) / (4 * math.pi)
                          * math.factorial(ell - order) / math.factorial(ell + order))
        # P_l^k includes the Condon--Shortley phase. For m<0, legacy=False
        # additionally uses Y_l^{-k}=(-1)^k conjugate(Y_l^k), unlike legacy=True.
        phase = (-1) ** order * ((-1) ** order if m < 0 else 1)
        polynomial[:len(derivative), index] = derivative * scale * phase
    return polynomial, np.asarray([m for _, m in angular_columns()], np.int64)


@lru_cache(maxsize=8)
def _angular_constants(device, dtype):
    polynomial, orders = _angular_numpy_constants()
    return (torch.as_tensor(polynomial, device=device, dtype=dtype),
            torch.as_tensor(orders, device=device))


@lru_cache(maxsize=16)
def _radial_numpy_inverse(width):
    centers = np.asarray(RADIAL_CENTERS, dtype=np.float64)
    kernel = np.exp(-.5 * ((centers[:, None] - centers[None, :]) / width) ** 2)
    try:
        inverse = np.linalg.solve(kernel, np.eye(3, dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise ValueError("radial_width produces a singular cardinal kernel") from error
    if not np.isfinite(inverse).all():
        raise ValueError("radial_width produces nonfinite cardinal coefficients")
    return inverse


@lru_cache(maxsize=32)
def _radial_constants(width, device, dtype):
    # The inversion is always FP64, even for a FP32 render.
    return (torch.tensor(RADIAL_CENTERS, device=device, dtype=dtype),
            torch.as_tensor(_radial_numpy_inverse(width), device=device, dtype=dtype))


def _directions(directions, validate):
    directions = torch.as_tensor(directions)
    if directions.is_complex():
        raise ValueError("Directions must be real values")
    if not directions.is_floating_point() or directions.dtype in (torch.float16, torch.bfloat16):
        directions = directions.float()
    if directions.dtype not in (torch.float32, torch.float64):
        raise ValueError("Directions must be real float32 or float64 values")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("Expected directions [Q,3]")
    squared_norm = directions.square().sum(-1, keepdim=True)
    if validate and (not torch.isfinite(directions).all() or torch.any(squared_norm <= 0)
                     or not torch.isfinite(squared_norm).all()):
        raise ValueError("Directions must be finite and nonzero")
    return directions, squared_norm


def _angular(directions, squared_norm):
    x, y, z = (directions * torch.rsqrt(squared_norm)).unbind(-1)
    polynomial, orders = _angular_constants(str(directions.device), directions.dtype)
    real, imaginary = [torch.ones_like(x)], [torch.zeros_like(x)]
    for _ in range(6):
        real.append(real[-1] * x - imaginary[-1] * y)
        imaginary.append(real[-2] * y + imaginary[-1] * x)
    azimuthal = torch.where(orders[None] > 0,
                           torch.stack(imaginary, -1)[:, orders.abs()],
                           torch.stack(real, -1)[:, orders.abs()])
    angular = torch.zeros((len(directions), ANGULAR_DIMENSION),
                          device=directions.device, dtype=directions.dtype)
    for degree in range(6, -1, -1):
        angular = angular * z[:, None] + polynomial[degree]
    scale = torch.full((ANGULAR_DIMENSION,), math.sqrt(2.),
                       device=directions.device, dtype=directions.dtype)
    scale = torch.where(orders == 0, torch.ones_like(scale), scale)
    return angular * azimuthal * scale[None]


def angular_basis(directions, *, validate=True):
    """Return [Q,28] SH6 harmonics, differentiable in nonzero directions."""
    directions, squared_norm = _directions(directions, validate)
    with torch.autocast(device_type=directions.device.type, enabled=False):
        return _angular(directions, squared_norm)


def cardinal_basis(directions, bvals, width=DEFAULT_WIDTH, *, validate=True):
    """Return [Q,84] continuous functions at positive b (in s/mm²).

    R(b) = k(b) K^{-1}, k_j(b)=exp(-0.5*((b/1000-c_j)/width)^2),
    K_ij=k_j(1000*c_i). Thus each radial function equals its cardinal
    Kronecker value at b=1000,2000,3000. Query interpolation is continuous;
    measurements are never rounded to the nearest shell.

    ``validate=False`` is only for acquisition tables checked beforehand.
    Float32/float64 direction precision is preserved, including under AMP.
    """
    width = configuration({"radial_width": width})["radial_width"]
    directions, squared_norm = _directions(directions, validate)
    bvals = torch.as_tensor(bvals, device=directions.device, dtype=directions.dtype)
    if bvals.shape != directions.shape[:1]:
        raise ValueError("Expected b values [Q] matching directions [Q,3]")
    if validate and (not torch.isfinite(bvals).all() or torch.any(bvals <= 0)):
        raise ValueError("This positive-b dictionary requires finite b>0; b=0 is not supported")
    with torch.autocast(device_type=directions.device.type, enabled=False):
        angular = _angular(directions, squared_norm)
        centers, inverse = _radial_constants(width, str(directions.device), directions.dtype)
        raw = torch.exp(-.5 * ((bvals[:, None] / 1000. - centers[None]) / width).square())
        radial = raw @ inverse
        return (radial[:, :, None] * angular[:, None, :]).reshape(len(directions), DIMENSION)


def basis(directions, bvals, config=None, *, validate=True):
    return cardinal_basis(directions, bvals, configuration(config)["radial_width"],
                          validate=validate)


def penalty_diagonal(config=None, *, smooth=None, device=None, dtype=torch.float64):
    """Angular Laplace--Beltrami penalty, repeated over the three radial blocks."""
    config = dict(config or {})
    configuration(config)
    smooth = float(config.get("smooth", .0005) if smooth is None else smooth)
    if not math.isfinite(smooth) or smooth < 0:
        raise ValueError("smooth must be finite and nonnegative")
    return torch.tensor([smooth * (ell * (ell + 1)) ** 2 for _, ell, _ in columns()],
                        device=device, dtype=dtype)


def definition(config=None):
    return dict(**configuration(config), dimension=DIMENSION, angular_dimension=ANGULAR_DIMENSION,
                angular_order=6, radial_centers_b=[1000., 2000., 3000.], columns=columns(),
                angular_convention="DIPY descoteaux07 legacy=False",
                column_order="radial-major, angular ell ascending then m ascending",
                b_domain="strictly positive; studied on b=1000..3000 s/mm2",
                continuous_direction_and_b=True, antipodally_symmetric=True,
                nonnegative_signal_guaranteed=False, monotonic_b_guaranteed=False,
                b0_supported=False)
