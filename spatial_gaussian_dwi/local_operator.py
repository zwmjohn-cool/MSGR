"""Local decoder Gaussian linear operator and its true Euclidean adjoint.

A latent row belongs to a Gaussian at its native pixel center. Its local
decoder acts *before* spatial Gaussian mixing: Tz = G(B_center z) Phi.T.
Local decoders cannot be pulled outside G. A global query design Phi may be
applied before G after each center has been decoded, saving channels when Q<D.
The geometry is fixed, full-tail and isotropic; this module learns neither
the decoder nor any grouping. Forward/adjoint retain tensor derivatives.
"""
from __future__ import annotations

import math
from time import perf_counter

import torch

from .model import gaussian_convolve


class LocalGaussianOperator:
    """Prepared T:[H,W,R]->[H,W,Q], or [H,W,D] without a query design.

    Args:
        decoders: [P,D,R], containing all physical local decoder values.
        group_index: integer [H,W], -1 outside the primitive support and a
            decoder index in [0,P) at each native Gaussian center.
        sigma: fixed isotropic standard deviation in native pixel units.
        query_design: optional [Q,D] analytic basis evaluated at allowed queries.

    Group packing is geometry-only. A prepared instance can be reused inside
    a solver. For learning the decoders, create a fresh instance for each new
    backward graph; no decoder-dependent tensor is detached or cached globally.
    Output images are not multiplied by a mask.
    """

    def __init__(self, decoders, group_index, sigma, query_design=None):
        decoders = torch.as_tensor(decoders)
        if (decoders.ndim != 3 or min(decoders.shape) < 1
                or decoders.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(decoders).all()):
            raise ValueError('decoders must be finite float32/float64 [P,D,R]')
        groups = torch.as_tensor(group_index, device=decoders.device)
        if (groups.ndim != 2 or min(groups.shape) < 1
                or groups.dtype == torch.bool or groups.is_floating_point()
                or groups.is_complex()):
            raise ValueError('group_index must be a nonempty integer [H,W] array')
        groups = groups.long()
        if torch.any(groups < -1) or torch.any(groups >= decoders.shape[0]):
            raise ValueError('group_index requires -1 or a valid decoder index')
        if isinstance(sigma, bool) or not math.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError('sigma must be finite and positive')
        query = None if query_design is None else torch.as_tensor(
            query_design, device=decoders.device, dtype=decoders.dtype)
        if query is not None and (query.ndim != 2 or query.shape[0] < 1
                or query.shape[1] != decoders.shape[1] or not torch.isfinite(query).all()):
            raise ValueError('query_design must be finite [Q,D]')
        self.decoders, self.group_index = decoders, groups
        self.sigma, self.query_design = float(sigma), query
        self.spatial_shape = tuple(groups.shape)
        self.rank, self.dimension = decoders.shape[2], decoders.shape[1]
        self.channels = self.dimension if query is None else len(query)
        self.active = groups >= 0
        self.valid_flat = torch.nonzero(self.active.reshape(-1), as_tuple=False).flatten()
        group = groups.reshape(-1)[self.valid_flat]
        order = torch.argsort(group, stable=True)
        self.sorted_flat = self.valid_flat[order]
        self.sorted_groups = group[order]
        counts = torch.bincount(group, minlength=len(decoders))
        starts = counts.cumsum(0) - counts
        self.local_positions = torch.arange(len(group), device=groups.device) - torch.repeat_interleave(starts, counts)
        self.maximum_group_size = int(counts.max())
        self.group_slots = self.sorted_groups * self.maximum_group_size + self.local_positions
        # Decode-to-query is legal here: the local decoder still acts at each
        # center, before spatial mixing. Never convolve latent channels first.
        self.project_first = query is not None and self.channels < self.dimension
        self.effective_decoders = (torch.einsum('qd,pdr->pqr', query, decoders)
                                   if self.project_first else decoders)

    def _tensor(self, value, channels, name):
        value = torch.as_tensor(value, device=self.decoders.device, dtype=self.decoders.dtype)
        if value.shape != (*self.spatial_shape, channels):
            raise ValueError(f'{name} must have shape {(*self.spatial_shape, channels)}')
        return value

    def _local_product(self, values, decoder, transpose=False):
        """Block-diagonal local map, with no expanded [H,W,D,R] allocation."""
        incoming = values.shape[-1]
        outgoing = decoder.shape[-1] if transpose else decoder.shape[-2]
        packed = values.new_zeros((len(decoder) * self.maximum_group_size, incoming))
        packed = packed.index_copy(0, self.group_slots, values.reshape(-1, incoming)[self.sorted_flat])
        packed = packed.reshape(len(decoder), self.maximum_group_size, incoming)
        product = torch.bmm(packed, decoder if transpose else decoder.transpose(1, 2))
        selected = product.reshape(-1, outgoing)[self.group_slots]
        output = values.new_zeros((self.group_index.numel(), outgoing))
        return output.index_copy(0, self.sorted_flat, selected).reshape(*self.spatial_shape, outgoing)

    def center_coefficients(self, latent):
        """Decode [H,W,R] to per-center physical [H,W,D] amplitudes, before G."""
        latent = self._tensor(latent, self.rank, 'latent')
        return self._local_product(latent, self.decoders)

    def forward(self, latent):
        """Render full image, including Gaussian tails outside primitive support."""
        latent = self._tensor(latent, self.rank, 'latent')
        decoded = self._local_product(latent, self.effective_decoders)
        field = gaussian_convolve(decoded, self.sigma)
        if self.query_design is not None and not self.project_first:
            field = field @ self.query_design.T
        return field

    __call__ = forward

    def adjoint(self, residual):
        """Exact Euclidean T*: image residual to latent-center residual."""
        residual = self._tensor(residual, self.channels, 'residual')
        if self.query_design is not None and not self.project_first:
            residual = residual @ self.query_design
        field = gaussian_convolve(residual, self.sigma, transpose=True)
        return self._local_product(field, self.effective_decoders, transpose=True)

    def _mask(self, mask):
        mask = self.active if mask is None else torch.as_tensor(mask, device=self.decoders.device)
        if mask.shape != self.spatial_shape or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('training mask must be binary [H,W]')
        return mask.bool()

    def normal_diagonal(self, mask=None, ridge=0.):
        """Exact diagonal of T* M T + ridge I on active latent coordinates.

        G(x,p)^2 has sigma/sqrt(2), so its weighted column norms require only
        one scalar-field convolution. The local query sensitivity is exact.
        """
        ridge = _ridge(ridge)
        mask = self._mask(mask)
        mass = gaussian_convolve(mask.to(self.decoders.dtype)[..., None],
                                 self.sigma / math.sqrt(2))[..., 0]
        decoder = (self.decoders if self.query_design is None else
                   torch.einsum('qd,pdr->pqr', self.query_design, self.decoders))
        sensitivity = decoder.square().sum(1)
        values = sensitivity[self.group_index.clamp_min(0)] * mass[..., None] + ridge
        return values * self.active[..., None]


def _ridge(ridge):
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError('ridge must be finite and nonnegative')
    return ridge


def forward(latent, decoders, group_index, sigma, query_design=None):
    return LocalGaussianOperator(decoders, group_index, sigma, query_design).forward(latent)


def adjoint(residual, decoders, group_index, sigma, query_design=None):
    return LocalGaussianOperator(decoders, group_index, sigma, query_design).adjoint(residual)


@torch.no_grad()
def fit_latent(target, decoders, group_index, sigma, query_design=None, *,
               target_space='signal', mask=None, initial=None, ridge=0.,
               tolerance=1e-6, max_iterations=500, require_convergence=True):
    """Minimize ||M(Tz-y)||_F^2 + ridge*||z||_F^2 by true-adjoint PCG.

    ``target_space='coefficients'`` accepts a physical [H,W,D] pixel field;
    it is evaluated using query_design before the *signal-metric* fit.
    ``target_space='signal'`` accepts [H,W,Q] observations. Query rows must
    already be restricted to the caller's permitted training acquisitions.
    The solver never estimates/changes a decoder, grouping, or query design.

    ``mask`` selects training image equations; default is group_index>=0.
    Nonfinite target values outside that mask are ignored. Background latent
    values are exactly zero. Ridge is absolute, with no hidden rescaling.
    The fit is offline; use forward for differentiable generated tensors.
    """
    began = perf_counter()
    ridge = _ridge(ridge)
    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if isinstance(max_iterations, bool) or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    operator = LocalGaussianOperator(decoders, group_index, sigma, query_design)
    selected = operator._mask(mask)
    if target_space not in ('signal', 'coefficients'):
        raise ValueError("target_space must be 'signal' or 'coefficients'")
    channels = operator.dimension if target_space == 'coefficients' else operator.channels
    target = operator._tensor(target, channels, 'target')
    if not torch.isfinite(target[selected]).all():
        raise ValueError('Selected training targets must be finite')
    target = torch.where(selected[..., None], target, 0.)
    if target_space == 'coefficients' and operator.query_design is not None:
        target = target @ operator.query_design.T
    active = operator.active[..., None]
    latent = (operator.decoders.new_zeros((*operator.spatial_shape, operator.rank)) if initial is None
              else operator._tensor(initial, operator.rank, 'initial').detach().clone())
    latent = torch.where(active, latent, 0.)
    if not torch.isfinite(latent).all():
        raise ValueError('Initial active latent values must be finite')

    def normal(value):
        prediction = operator.forward(value)
        return operator.adjoint(prediction * selected[..., None]) + ridge * value

    rhs = operator.adjoint(target)
    rhs_norm = rhs.norm()
    if rhs_norm == 0:
        # The zero vector is the minimum-norm solution even if a provided
        # warm start lies in a null space; avoid a zero-relative-scale target.
        latent.zero_()
    diagonal = operator.normal_diagonal(selected, ridge)
    scale = diagonal.max().clamp_min(torch.finfo(diagonal.dtype).tiny)
    diagonal = diagonal.clamp_min(scale * torch.finfo(diagonal.dtype).eps)
    residual = rhs - normal(latent)
    reference = rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
    threshold = float(tolerance) * reference
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum()
    iterations = 0
    # The operator couples latent channels via local B and q-space design;
    # independent per-channel CG steps would be mathematically incorrect.
    for step in range(max_iterations):
        if residual.norm() <= threshold:
            break
        action = normal(direction)
        denominator = (direction * action).sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError('Local Gaussian PCG lost positive definiteness')
        alpha = rz / denominator
        latent = latent + alpha * direction
        residual = residual - alpha * action
        iterations = step + 1
        # Recompute from the actual operator periodically to avoid accepting a
        # small recursive residual after accumulated floating-point error.
        if iterations % 25 == 0 or residual.norm() <= threshold:
            residual = rhs - normal(latent)
        if residual.norm() <= threshold:
            break
        preconditioned = residual / diagonal
        next_rz = (residual * preconditioned).sum()
        direction = preconditioned + (next_rz / rz) * direction
        rz = next_rz
    latent = torch.where(active, latent, 0.)
    true_residual = rhs - normal(latent)
    relative_residual = float(true_residual.norm() / reference)
    converged = relative_residual <= float(tolerance) * 1.05
    prediction = operator.forward(latent)
    error = (prediction - target) * selected[..., None]
    signal_energy = target.square().sum()
    relative_error = float(error.norm() / signal_energy.sqrt().clamp_min(torch.finfo(target.dtype).tiny))
    if not torch.isfinite(latent).all():
        raise RuntimeError('Nonfinite local Gaussian latent solution')
    details = dict(iterations=iterations, maximum_iterations=max_iterations,
        relative_normal_residual=relative_residual, converged=converged,
        relative_training_error=relative_error, ridge=ridge,
        objective=float(error.square().sum() + ridge * latent.square().sum()),
        target_space=target_space, training_pixels=int(selected.sum()),
        primitive_count=int(operator.active.sum()), signal_queries=operator.channels,
        latent_rank=operator.rank, physical_dimension=operator.dimension,
        fit_seconds=perf_counter()-began,
        algorithm='true-adjoint globally coupled PCG; exact Jacobi diagonal',
        decoder_updated=False, gaussian_sigma=float(sigma), full_tails=True)
    if require_convergence and not converged:
        raise RuntimeError(f'Local Gaussian solve did not converge: relative normal residual {relative_residual}')
    return latent, details
