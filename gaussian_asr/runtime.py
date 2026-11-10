"""Frozen Gaussian decoder/renderer adapter with no access to test targets.

Construct this from an already authenticated synthesis adapter and an explicit
B0/query table. This class never calls ``adapter.targets`` or a dataset reader:
it cannot load clean latents, fitted coefficients, or missing test DWIs.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FrozenGaussianRuntime:
    """Invert standardized latents using the exact frozen synthesis decoder.

    ``b0`` supplies only primitive support (strictly positive native B0); the
    evaluation mask and positive-b observations do not determine that support.
    A full query basis is acquisition metadata, not an acquired signal. All
    observed shells are selected together and reconstructed in one solve.
    """

    def __init__(self, *, model, codec, latent_mean, latent_std, b0,
                 query_basis, observed_indices, mixed_precision=True,
                 checkpoint_decoder=True, metadata=None):
        self.model, self.codec = model, codec
        self.device = next(codec.parameters()).device
        self.mixed_precision = bool(mixed_precision)
        self.checkpoint_decoder = bool(checkpoint_decoder)
        for network in (model, codec):
            if network is not None:
                network.eval().requires_grad_(False)
                for parameter in network.parameters():
                    parameter.grad = None
        self.latent_mean = torch.as_tensor(latent_mean, device=self.device, dtype=torch.float32).detach().reshape(1, -1, 1, 1)
        self.latent_std = torch.as_tensor(latent_std, device=self.device, dtype=torch.float32).detach().reshape(1, -1, 1, 1)
        if (self.latent_mean.shape != self.latent_std.shape
                or not torch.isfinite(self.latent_mean).all()
                or not torch.isfinite(self.latent_std).all() or (self.latent_std <= 0).any()):
            raise ValueError("Invalid frozen standardized-latent normalization")
        h, w = codec.renderer.native_shape
        b0 = torch.as_tensor(b0, device=self.device).detach()
        self.unbatched_output = b0.ndim == 2
        if b0.ndim == 2:
            b0 = b0[None]
        if b0.ndim != 3 or b0.shape[-2:] != (h, w) or not torch.isfinite(b0).all():
            raise ValueError("Supply finite native B0 [H,W] or [B,H,W], not a DWI evaluation mask")
        self.batch_size = len(b0)
        support = b0 > 0
        if not support.flatten(1).any(1).all():
            raise ValueError("At least one case has empty B0-derived support")
        padded = tuple(codec.model.config.padded_shape)
        if padded[0] < h or padded[1] < w:
            raise ValueError("Decoder padding cannot be smaller than the native image")
        self.support = F.pad(support[:, None], (0, padded[1] - w, 0, padded[0] - h))
        basis = torch.as_tensor(query_basis, device=self.device, dtype=torch.float32).detach().clone()
        if basis.ndim == 2:
            basis = basis[None]
        if basis.ndim == 3 and len(basis) == 1:
            basis = basis.expand(self.batch_size, -1, -1)
        if (basis.ndim != 3 or len(basis) != self.batch_size or min(basis.shape) < 1
                or not torch.isfinite(basis).all()):
            raise ValueError("Query basis must be finite shared [Q,R] or case-wise [B,Q,R]")
        indices = torch.as_tensor(observed_indices, device=self.device, dtype=torch.long).detach().clone()
        if indices.ndim == 1:
            indices = indices[None].expand(self.batch_size, -1)
        if (indices.ndim != 2 or len(indices) != self.batch_size or indices.shape[1] < 1
                or any(len(torch.unique(case)) != len(case) for case in indices)
                or (indices < 0).any() or (indices >= basis.shape[1]).any()):
            raise ValueError("Observed indices must be nonempty, unique acquisition indices")
        self.query_basis = basis
        self.observed_indices = indices
        self.observed_query_basis = basis.gather(1, indices[..., None].expand(-1, -1, basis.shape[-1]))
        self.latent_shape = (self.batch_size, model.config.latent_channels, *model.config.latent_shape)
        if self.latent_shape[1] != self.latent_mean.shape[1]:
            raise ValueError("Frozen flow and latent-normalization channels differ")
        self.metadata = dict(metadata or {})
        self.metadata.update(
            primitive_support="native B0 > 0; bottom/right zero padding only",
            batch_size=self.batch_size, observed_directions=indices.shape[1], output_directions=basis.shape[1],
            reconstruction="one joint-shell solve",
            latent_denormalization="z * frozen latent_std + frozen latent_mean",
            raw_signal_output=True, checkpoint_decoder=self.checkpoint_decoder)

    @classmethod
    def from_adapter(cls, adapter, *, b0, query_basis, observed_indices,
                     checkpoint_decoder=True):
        """Reuse already authenticated frozen assets; never read adapter targets."""
        return cls(model=adapter.model, codec=adapter.codec,
                   latent_mean=adapter.latent_mean, latent_std=adapter.latent_std,
                   b0=b0, query_basis=query_basis, observed_indices=observed_indices,
                   mixed_precision=adapter.mixed_precision,
                   checkpoint_decoder=checkpoint_decoder,
                   metadata=getattr(adapter, "metadata", None))

    def autocast(self):
        if self.mixed_precision and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _decode(self, latent):
        with self.autocast():
            return self.codec.model.decode(latent.float() * self.latent_std + self.latent_mean)

    def _render(self, latent, query_basis):
        if tuple(latent.shape) != self.latent_shape:
            raise ValueError("Expected standardized latent batch with the frozen flow shape")
        if self.checkpoint_decoder and torch.is_grad_enabled() and latent.requires_grad:
            coefficients = checkpoint(self._decode, latent, use_reentrant=False)
        else:
            coefficients = self._decode(latent)
        images = self.codec.renderer.forward_from_query_basis(
            coefficients.float(), self.support, query_basis).float()
        return images[0] if self.unbatched_output else images

    def render_observed(self, latent):
        return self._render(latent, self.observed_query_basis)

    def render_all(self, latent):
        return self._render(latent, self.query_basis)
