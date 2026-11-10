"""Training-only patch spectral denoising on the selected measured queries."""
from functools import lru_cache
import math
import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import brentq


def _as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)



def _patch_controls(size, noise_factor, offsets):
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Patch size must be a positive integer")
    if isinstance(offsets, (bool, np.bool_)) or not isinstance(offsets, (int, np.integer)) or offsets not in (1, 4):
        raise ValueError("Patch offsets must be 1 or 4")
    if (isinstance(noise_factor, (bool, np.bool_))
            or not isinstance(noise_factor, (int, float, np.integer, np.floating))
            or not math.isfinite(float(noise_factor)) or noise_factor < 0):
        raise ValueError("Patch noise factor must be finite and nonnegative")
    return int(size), float(noise_factor), int(offsets)



def _mask_array(mask, shape):
    mask = _as_numpy(mask)
    if mask.shape != shape or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Require a matching 2D binary brain mask")
    return mask.astype(bool, copy=False)



@lru_cache(maxsize=512)
def marchenko_pastur_median(rows, columns):
    """Median of the nonzero MP spectrum for the effective centered dimensions."""
    if rows < 1 or columns < 1:
        raise ValueError("MP dimensions must be positive")
    beta = min(rows, columns) / max(rows, columns)
    low, high = (1 - math.sqrt(beta)) ** 2, (1 + math.sqrt(beta)) ** 2

    def density(value):
        return math.sqrt(max((high - value) * (value - low), 0.)) / (2 * math.pi * beta * value)

    return brentq(lambda value: quad(density, low, value)[0] - .5,
                  low + 1e-12, high - 1e-12)



def _patch_indices(mask, size, offsets):
    height, width = mask.shape
    shifts = (0,) if offsets == 1 else (0, size // 2)
    indices = np.arange(height * width).reshape(height, width)
    patches = []
    for row_shift in shifts:
        for column_shift in shifts:
            for row in range(-row_shift, height, size):
                for column in range(-column_shift, width, size):
                    region = (slice(max(0, row), min(height, row + size)),
                              slice(max(0, column), min(width, column + size)))
                    selected = indices[region][mask[region]]
                    if selected.size:
                        patches.append(selected)
    return patches



def patch_mp_denoise_cpu(data, mask, size=9, noise_factor=1., offsets=4):
    """NumPy SVD oracle, including the general centered rows >= queries case.

    A patch with P brain pixels has P-1 effective rows after subtracting its
    mean. When P <= Q its final singular value is a structural zero and is
    excluded from noise estimation. When P > Q all Q singular values enter.
    Patches with fewer than four brain pixels are passed through unchanged.
    """
    size, noise_factor, offsets = _patch_controls(size, noise_factor, offsets)
    data = np.asarray(data)
    if data.ndim != 3 or min(data.shape) < 1 or data.dtype.kind != "f":
        raise ValueError("Require nonempty floating point data [H,W,Q]")
    mask = _mask_array(mask, data.shape[:2])
    if not np.isfinite(data[mask]).all():
        raise ValueError("Selected brain signals must be finite")
    source = np.where(mask[..., None], data, 0.).reshape(-1, data.shape[-1])
    result = np.zeros_like(source)
    counts = np.zeros((len(source), 1), dtype=source.dtype)
    for indices in _patch_indices(mask, size, offsets):
        block = source[indices]
        if len(block) < 4 or noise_factor == 0:
            clean = block
        else:
            mean = block.mean(0)
            left, singular, right = np.linalg.svd(block - mean, full_matrices=False)
            rows, queries = len(block) - 1, data.shape[-1]
            rank = min(rows, queries)
            normalization = math.sqrt(max(rows, queries))
            sigma = (np.median(singular[:rank])
                     / math.sqrt(max(rows, queries) * marchenko_pastur_median(rows, queries))
                     * noise_factor)
            scaled = singular / (normalization * sigma + 1e-15)
            beta = min(rows, queries) / max(rows, queries)
            discriminant = (scaled * scaled - beta - 1) ** 2 - 4 * beta
            shrunk = np.where(scaled > 1 + math.sqrt(beta),
                             np.sqrt(np.maximum(discriminant, 0.)) / (scaled + 1e-15), 0.)
            shrunk *= normalization * sigma
            if rank < len(shrunk):
                shrunk[rank:] = 0.
            clean = (left * shrunk) @ right + mean
        result[indices] += clean
        counts[indices] += 1
    return (result / np.maximum(counts, 1)).reshape(data.shape)



@torch.no_grad()
def patch_mp_denoise_torch(data, mask, size=9, noise_factor=1., offsets=4,
                          batch_size=256):
    """Batched eigendecomposition with exact brain-only patch membership.

    Patches are grouped by their number of brain pixels. The smaller of the
    row and query covariance matrices is diagonalized; no padded background
    contributes to a mean, covariance, noise estimate, or output.
    """
    size, noise_factor, offsets = _patch_controls(size, noise_factor, offsets)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Patch batch size must be a positive integer")
    if data.ndim != 3 or min(data.shape) < 1 or data.dtype not in (torch.float32, torch.float64):
        raise ValueError("Require nonempty float32/float64 data [H,W,Q]")
    brain = _mask_array(mask, tuple(data.shape[:2]))
    brain_device = torch.as_tensor(brain, device=data.device)
    if not bool(torch.isfinite(data[brain_device]).all()):
        raise ValueError("Selected brain signals must be finite")
    source = torch.where(brain_device[..., None], data, 0.).reshape(-1, data.shape[-1])
    output = torch.zeros_like(source)
    counts = torch.zeros((len(source), 1), dtype=source.dtype, device=source.device)
    grouped = {}
    for indices in _patch_indices(brain, size, offsets):
        grouped.setdefault(len(indices), []).append(indices)
    queries = data.shape[-1]
    for points, patches in grouped.items():
        rows = points - 1
        for start in range(0, len(patches), int(batch_size)):
            indices = torch.as_tensor(np.stack(patches[start:start + batch_size]), device=data.device)
            block = source[indices]
            if points < 4 or noise_factor == 0:
                clean = block
            else:
                mean = block.mean(1, keepdim=True)
                centered = block - mean
                left_covariance = points <= queries
                covariance = (centered @ centered.transpose(-2, -1) if left_covariance
                              else centered.transpose(-2, -1) @ centered)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                singular = eigenvalues.clamp_min(0).sqrt()
                rank = min(rows, queries)
                effective = singular[:, -rank:]
                # torch.median chooses the lower middle value, whereas the SVD
                # oracle uses the arithmetic mean of the two middle values.
                median = (effective[:, (rank - 1) // 2] + effective[:, rank // 2]) * .5
                normalization = math.sqrt(max(rows, queries))
                sigma = median / math.sqrt(max(rows, queries) * marchenko_pastur_median(rows, queries)) * noise_factor
                scaled = singular / (normalization * sigma[:, None] + 1e-15)
                beta = min(rows, queries) / max(rows, queries)
                discriminant = (scaled.square() - beta - 1).square() - 4 * beta
                shrunk = torch.where(scaled > 1 + math.sqrt(beta),
                    discriminant.clamp_min(0).sqrt() / (scaled + 1e-15), 0.)
                shrunk *= normalization * sigma[:, None]
                ratio = shrunk / singular.clamp_min(torch.finfo(singular.dtype).tiny)
                if rank < ratio.shape[-1]:
                    ratio[:, :-rank] = 0.
                if left_covariance:
                    clean = (eigenvectors * ratio[:, None]) @ (eigenvectors.transpose(-2, -1) @ centered)
                else:
                    clean = ((centered @ eigenvectors) * ratio[:, None]) @ eigenvectors.transpose(-2, -1)
                clean += mean
            output.index_add_(0, indices.reshape(-1), clean.reshape(-1, queries))
            counts.index_add_(0, indices.reshape(-1), torch.ones((indices.numel(), 1), device=data.device, dtype=data.dtype))
    return (output / counts.clamp_min(1)).reshape(data.shape)
