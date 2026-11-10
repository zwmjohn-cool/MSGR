"""Fused full-tail rendering of fixed-grid shared radial--SH Gaussian fields.

Only geometry/dictionary constants are retained. Spatial fields, query bases,
and predictions are recomputed from the current inputs on every call. CUDA
compilation fuses the small Cartesian-SH operations; dense spatial matrices
retain every Gaussian tail, including the numerical underflow convention of
the reference implementation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import grid_kernel_matrices
from .radial_sh import _angular_numpy_constants, _radial_numpy_inverse
from .shared_model import SharedBasis


class FastSharedGaussianRenderer(nn.Module):
    """Native ``[H,W,R]`` amplitudes to ``[H,W,Q]`` raw DWI.

    ``basis_matrix`` remains connected to the supplied basis tensor, so image
    gradients can reach both amplitudes and B. Fixed mask/geometry constants
    move with this module through ``.to``. No subject amplitudes are stored.

    The default forward assumes finite, nonzero directions and finite positive
    b-values previously checked with :meth:`validate_queries`. Passing
    ``validate=True`` checks them on each call; on CUDA those numerical checks
    synchronize the device. Both paths rebuild the query basis every time.
    """

    def __init__(self, basis, mask):
        super().__init__()
        if not isinstance(basis, SharedBasis):
            raise TypeError("Expected an authenticated SharedBasis")
        support = np.asarray(mask)
        if (support.ndim != 2 or not support.any()
                or not np.all((support == 0) | (support == 1))):
            raise ValueError("Require a nonempty binary native mask")
        self.spatial_shape = tuple(support.shape)
        self.rank = basis.rank
        self.sigma = basis.config["sigma"]
        self.width = basis.config["width"]
        self.basis_content_id = basis.content_id
        matrix = basis.matrix
        self.register_buffer("basis_matrix", matrix)
        self.register_buffer("mask", torch.as_tensor(support.astype(bool), device=matrix.device))
        row, column = grid_kernel_matrices(self.spatial_shape, self.sigma,
                                           device=matrix.device, dtype=matrix.dtype)
        self.register_buffer("row_kernel", row)
        self.register_buffer("column_kernel", column)
        polynomial, orders = _angular_numpy_constants()
        self.register_buffer("angular_polynomial", torch.as_tensor(
            polynomial, device=matrix.device, dtype=matrix.dtype))
        self.register_buffer("angular_orders", torch.as_tensor(orders, device=matrix.device))
        self.register_buffer("angular_scale", torch.as_tensor(
            np.where(orders == 0, 1., np.sqrt(2.)), device=matrix.device, dtype=matrix.dtype))
        self.register_buffer("radial_centers", torch.tensor(
            [1., 2., 3.], device=matrix.device, dtype=matrix.dtype))
        self.register_buffer("radial_inverse", torch.as_tensor(
            _radial_numpy_inverse(self.width), device=matrix.device, dtype=matrix.dtype))
        self._compiled_render = None
        self._compiled_validator = None

    def _coerce_queries(self, directions, bvals):
        directions = torch.as_tensor(directions, device=self.basis_matrix.device,
                                     dtype=self.basis_matrix.dtype)
        bvals = torch.as_tensor(bvals, device=self.basis_matrix.device,
                                dtype=self.basis_matrix.dtype)
        if (directions.ndim != 2 or directions.shape[-1] != 3
                or bvals.shape != directions.shape[:1]):
            raise ValueError("Expected directions[Q,3] and bvals[Q]")
        return directions, bvals

    @staticmethod
    def _valid_query_values(directions, bvals):
        """One scalar device predicate, including overflow of the norm square."""
        squared = directions.square().sum(-1)
        return (torch.isfinite(directions).all() & torch.isfinite(squared).all()
                & (squared > 0).all() & torch.isfinite(bvals).all() & (bvals > 0).all())

    def validate_queries(self, directions, bvals):
        """Check current query values with one host/device synchronization.

        Checking remains strict on every call. Compilation only fuses the
        predicate's reductions; it does not retain a query-validity result.
        """
        directions, bvals = self._coerce_queries(directions, bvals)
        predicate = self._valid_query_values if self._compiled_validator is None else self._compiled_validator
        if not bool(predicate(directions, bvals)):
            raise ValueError("Require finite nonzero directions and finite positive b-values")
        return directions, bvals

    def _query_matrix(self, directions, bvals):
        x, y, z = (directions * torch.rsqrt(directions.square().sum(-1, keepdim=True))).unbind(-1)
        real, imaginary = [torch.ones_like(x)], [torch.zeros_like(x)]
        for _ in range(6):
            previous_real, previous_imaginary = real[-1], imaginary[-1]
            real.append(previous_real * x - previous_imaginary * y)
            imaginary.append(previous_real * y + previous_imaginary * x)
        orders = self.angular_orders
        azimuth = torch.where(orders[None] > 0,
                              torch.stack(imaginary, -1)[:, orders.abs()],
                              torch.stack(real, -1)[:, orders.abs()])
        angular = torch.zeros((len(directions), 28), device=directions.device,
                              dtype=directions.dtype)
        for degree in range(6, -1, -1):
            angular = angular * z[:, None] + self.angular_polynomial[degree]
        angular = angular * azimuth * self.angular_scale[None]
        raw_radial = torch.exp(-.5 * ((bvals[:, None] / 1000. - self.radial_centers[None])
                                      / self.width).square())
        radial = raw_radial @ self.radial_inverse
        phi = (radial[:, :, None] * angular[:, None]).reshape(len(directions), 84)
        return phi @ self.basis_matrix

    def _render(self, latent, directions, bvals):
        h, w = self.spatial_shape
        amplitudes = torch.where(self.mask[..., None], latent, 0.)
        # Larger, regular GEMMs avoid 22 strided/broadcast per-channel products.
        intermediate = (self.row_kernel @ amplitudes.reshape(h, w * self.rank)).reshape(h, w, self.rank)
        horizontal = intermediate.permute(0, 2, 1).contiguous().reshape(h * self.rank, w)
        field = (horizontal @ self.column_kernel.T).reshape(h, self.rank, w).permute(0, 2, 1)
        return field @ self._query_matrix(directions, bvals).T

    def enable_compilation(self, *, mode="default"):
        """Compile the current-input render; first-call compilation is setup.

        The compiled graph still receives the current amplitudes, g, b and B.
        Inputs can change between calls; changing shapes/dtypes may recompile.
        This optional method requires a working PyTorch Inductor backend.
        """
        self._compiled_render = torch.compile(self._render, fullgraph=True, dynamic=False, mode=mode)
        self._compiled_validator = torch.compile(self._valid_query_values, fullgraph=True,
                                                 dynamic=False, mode=mode)
        return self

    def disable_compilation(self):
        self._compiled_render = None
        self._compiled_validator = None
        return self

    def forward(self, latent, directions, bvals, *, validate=False):
        latent = torch.as_tensor(latent, device=self.basis_matrix.device, dtype=self.basis_matrix.dtype)
        if latent.shape != (*self.spatial_shape, self.rank):
            raise ValueError("Expected amplitudes[H,W,R] matching the fixed mask and public basis")
        if validate:
            directions, bvals = self.validate_queries(directions, bvals)
        else:
            directions, bvals = self._coerce_queries(directions, bvals)
        with torch.autocast(device_type=latent.device.type, enabled=False):
            function = self._render if self._compiled_render is None else self._compiled_render
            return function(latent, directions, bvals)
