"""Training-only teachers and a common explicit q-space subspace.

No subject in the evaluation-only group is accepted implicitly: the driver
passes exactly the selected basis-training sufficient statistics. Whitening
is a numerical coordinate change; the returned basis is physical phi84 B.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import torch

from .denoise import patch_mp_denoise_cpu
from .local_fitting import configuration
from .radial_sh import cardinal_basis, penalty_diagonal


def prepare_teacher(data, directions, bvals, mask, indices, config=None):
    """Return complete physical [H,W,84] teacher and training-only statistics."""
    start = perf_counter()
    cfg = configuration(config)
    data, g, b, mask, ids = map(np.asarray, (data, directions, bvals, mask, indices))
    if data.ndim != 3 or data.dtype.kind != 'f':
        raise ValueError('data must be floating H,W,Q')
    if mask.shape != data.shape[:2] or not np.all((mask == 0) | (mask == 1)):
        raise ValueError('mask must be matching binary H,W')
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError('Shared experiment requires a nonempty brain mask')
    if g.shape != (data.shape[-1], 3) or b.shape != (data.shape[-1],):
        raise ValueError('Acquisition dimensions disagree')
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
            or np.any(ids >= len(b))):
        raise ValueError('Training indices must be unique and in range')
    # Selection precedes all statistics and validation of signal/query values.
    signals = np.asarray(data[..., ids], dtype=np.float64)[mask]
    if not np.isfinite(signals).all():
        raise ValueError('Training brain signals must be finite')
    phi = cardinal_basis(torch.from_numpy(g[ids].astype(np.float64)),
                         torch.from_numpy(b[ids].astype(np.float64)),
                         width=cfg['width']).numpy()
    canvas = np.zeros((*mask.shape, len(ids)), np.float64)
    canvas[mask] = signals
    if cfg['mp_blend'] and cfg['noise_factor']:
        clean = patch_mp_denoise_cpu(canvas, mask, size=cfg['patch_size'],
                    noise_factor=cfg['noise_factor'], offsets=cfg['patch_offsets'])[mask]
        target = (1-cfg['mp_blend'])*signals + cfg['mp_blend']*clean
    else:
        target = signals
    penalty = penalty_diagonal(cfg, dtype=torch.float64, device='cpu').numpy()
    coefficients = np.linalg.solve(phi.T @ phi + np.diag(penalty),
                                   (target @ phi).T).T
    field = np.zeros((*mask.shape, 84), np.float64)
    field[mask] = coefficients
    rms2 = float(np.mean(signals**2))
    scale = rms2 if rms2 > 0 else 1.
    covariance = coefficients.T @ coefficients / (mask.sum()*scale)
    metric = phi.T @ phi / len(ids)
    return field, dict(covariance=covariance, query_metric=metric), dict(
        training_indices=ids.tolist(), selected_only=True,
        training_signal_rms=float(np.sqrt(rms2)), brain_pixels=int(mask.sum()),
        covariance_weight='per-subject mean voxel coefficient moment / raw training signal RMS squared',
        raw_intensity=True, b0_division=False, config=cfg,
        total_seconds=perf_counter()-start)


def _projection_loss(matrix, covariance, metric):
    """Squared relative training-signal projection error, averaged by subject."""
    hb = metric @ matrix
    gram = matrix.T @ hb
    captured = hb.transpose(-1, -2) @ covariance @ hb
    total = torch.einsum('sij,sji->s', covariance, metric)
    retained = torch.linalg.solve(gram, captured).diagonal(dim1=-2, dim2=-1).sum(-1)
    return (total - retained).mean()


def learn_common_basis(statistics, rank, *, steps=300, learning_rate=.01, device='cpu'):
    """Common-metric PCA followed by exact varying-query subspace refinement.

    Sufficient statistics must come only from basis-training subjects and
    training directions. Every subject has equal weight in squared relative
    training-signal error. No per-subject/voxel decoder is returned or saved.
    """
    if not statistics or type(rank) is not int or not 1 <= rank <= 84:
        raise ValueError('Require training statistics and rank in [1,84]')
    if type(steps) is not int or steps < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid optimization controls')
    dtype = torch.float64
    covariance = torch.as_tensor(np.stack([s['covariance'] for s in statistics]), device=device, dtype=dtype)
    metric = torch.as_tensor(np.stack([s['query_metric'] for s in statistics]), device=device, dtype=dtype)
    if (covariance.shape != metric.shape or covariance.shape[1:] != (84,84)
            or not torch.isfinite(covariance).all() or not torch.isfinite(metric).all()):
        raise ValueError('Require finite 84x84 moments and query metrics')
    with torch.no_grad():
        common = metric.mean(0)
        chol = torch.linalg.cholesky(common)
        inverse_transpose = torch.linalg.solve_triangular(chol.T, torch.eye(84, device=device, dtype=dtype), upper=True)
        whitened = chol.T @ covariance.mean(0) @ chol
        eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
        initial = eigenvectors[:, -rank:].flip(1).contiguous()
    modes = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([modes], lr=learning_rate)
    trace = []
    best_modes = initial.clone()
    best_loss = float('inf')
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        orthogonal = torch.linalg.qr(modes, mode='reduced').Q
        matrix = inverse_transpose @ orthogonal
        loss = _projection_loss(matrix, covariance, metric)
        value = float(loss.detach())
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite common basis objective')
        if value < best_loss:
            best_loss = value
            best_modes = orthogonal.detach().clone()
        if step == 0 or step % 25 == 0 or step == steps:
            trace.append(dict(step=step, training_relative_mse=value))
        if step < steps:
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        result = inverse_transpose @ best_modes
        # Canonical signs, with common-metric unit norms; no hidden scales.
        columns = torch.arange(rank, device=result.device)
        signs = torch.sign(result[result.abs().argmax(0), columns])
        result = result * torch.where(signs == 0, torch.ones_like(signs), signs)
    return result.detach().cpu().numpy().astype(np.float32), dict(
        rank=rank, basis_training_subjects=len(statistics), steps=steps,
        learning_rate=learning_rate, objective='equal-subject squared relative training-signal projection error',
        method='common-query-metric PCA + exact per-subject-query-metric Adam refinement',
        training_only=True, validation_used=False, trace=trace,
        best_training_relative_mse=best_loss, public_parameters=84*rank,
        leading_common_metric_eigenvalues=eigenvalues[-rank:].flip(0).cpu().tolist())
