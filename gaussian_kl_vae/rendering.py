"""Differentiable batched fixed-native-center Gaussian rendering for VAE loss."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spatial_gaussian_dwi.model import gaussian_convolve
from spatial_gaussian_dwi.radial_sh import cardinal_basis
from spatial_gaussian_dwi.shared_model import SharedBasis

from .data import ChannelNormalization


class SharedGaussianRenderer(nn.Module):
    """Normalized BCHW -> native raw-intensity BQHW, with full Gaussian tails.

    A supplied primitive mask is external side information and must be used
    identically for target and prediction. It does not zero rendered background.
    Inference without a known mask is possible with ``mask=None``, but has a
    different all-native-grid support and is not claimed equivalent to this pilot.
    No B0 division or multiplication is performed.
    """

    def __init__(self, basis, normalization=None, *, native_shape=(145, 174)):
        super().__init__()
        if isinstance(basis, (str, Path)):
            basis = SharedBasis.load(basis)
        if not isinstance(basis, SharedBasis):
            raise TypeError("Expected an authenticated SharedBasis or its file path")
        if normalization is not None and normalization.basis_content_id != basis.content_id:
            raise ValueError("Renderer normalization/public basis hash mismatch")
        self.register_buffer("basis_matrix", basis.matrix.detach().clone())
        self.basis_content_id = basis.content_id
        self.sigma, self.width = basis.config["sigma"], basis.config["width"]
        self.normalization = normalization
        self.native_shape = tuple(int(value) for value in native_shape)
        if len(self.native_shape) != 2 or min(self.native_shape) < 1:
            raise ValueError("Native shape must be positive H,W")

    def query_matrix(self, directions, bvals):
        """Return [Q,R] or [B,Q,R] for continuous positive-b queries."""
        directions = torch.as_tensor(directions, device=self.basis_matrix.device, dtype=self.basis_matrix.dtype)
        bvals = torch.as_tensor(bvals, device=self.basis_matrix.device, dtype=self.basis_matrix.dtype)
        if directions.ndim not in (2, 3) or directions.shape[-1] != 3 or bvals.shape != directions.shape[:-1]:
            raise ValueError("Queries must be [Q,3]/[Q] or [B,Q,3]/[B,Q]")
        phi = cardinal_basis(directions.reshape(-1, 3), bvals.reshape(-1), self.width)
        with torch.autocast(device_type=directions.device.type, enabled=False):
            result = phi @ self.basis_matrix
        return result.reshape(*directions.shape[:-1], self.basis_matrix.shape[1])

    def inverse_coefficients(self, coefficients, mask=None, *, normalized=True):
        """Crop native coordinates before inverse; padding never becomes centers."""
        h, w = self.native_shape
        if coefficients.ndim != 4 or coefficients.shape[1] != self.basis_matrix.shape[1]:
            raise ValueError("Expected [B,R,H,W] coefficient tensor")
        if coefficients.shape[-2] < h or coefficients.shape[-1] < w:
            raise ValueError("Coefficient tensor cannot be smaller than native shape")
        values = coefficients[..., :h, :w].to(dtype=self.basis_matrix.dtype)
        if mask is not None:
            if mask.ndim != 4 or mask.shape[:2] != (len(values), 1) or mask.shape[-2:] != coefficients.shape[-2:]:
                raise ValueError("Primitive mask must match coefficient [B,1,H,W]")
            mask = mask[..., :h, :w].to(device=values.device, dtype=torch.bool)
        if normalized:
            if self.normalization is None:
                raise ValueError("Normalized inputs require frozen channel normalization")
            values = self.normalization.inverse(values, mask)
        elif mask is not None:
            values = torch.where(mask, values, torch.zeros((), device=values.device, dtype=values.dtype))
        return values

    def forward(self, coefficients, mask=None, directions=None, bvals=None, *,
                query_matrix=None, normalized=True, padded=False):
        """Render a shared or per-sample Q matrix; optional bottom/right output pad.

        Q can vary between calls. Within a batch use the same Q length, or call
        individual batches; zero-b padding queries are not accepted.
        """
        raw = self.inverse_coefficients(coefficients, mask, normalized=normalized)
        if query_matrix is None:
            if directions is None or bvals is None:
                raise ValueError("Provide directions/bvals or a precomputed query_matrix")
            query_matrix = self.query_matrix(directions, bvals)
        query_matrix = query_matrix.to(device=raw.device, dtype=raw.dtype)
        if query_matrix.ndim == 2:
            query_matrix = query_matrix.unsqueeze(0).expand(len(raw), -1, -1)
        if query_matrix.ndim != 3 or query_matrix.shape[0] != len(raw) or query_matrix.shape[-1] != raw.shape[1]:
            raise ValueError("Query matrix must have shape [Q,R] or [B,Q,R]")
        with torch.autocast(device_type=raw.device.type, enabled=False):
            field = gaussian_convolve(raw.permute(0, 2, 3, 1), self.sigma)
            output = torch.bmm(field.flatten(1, 2), query_matrix.transpose(1, 2))
            output = output.reshape(len(raw), *self.native_shape, query_matrix.shape[1]).permute(0, 3, 1, 2)
        if padded:
            h, w = self.native_shape
            output = F.pad(output, (0, coefficients.shape[-1] - w, 0, coefficients.shape[-2] - h))
        return output

    def forward_from_query_basis(self, coefficients, mask, query_basis, *, normalized=True, padded=False):
        """Cached acquisition basis fast path; gradients still reach coefficients."""
        return self(coefficients, mask, query_matrix=query_basis, normalized=normalized, padded=padded)
