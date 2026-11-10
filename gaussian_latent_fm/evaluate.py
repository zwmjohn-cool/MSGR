"""Deterministic monitor seed and image similarity metric."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

def masked_ssim(prediction, target, mask, *, window=7):
    """Uniform-window SSIM, only windows wholly inside brain, one GT range.

    Raw intensity has no universal MRI range.  This report fixes the range to
    the target's brain 99.5th percentile for this subject across all queries;
    Online/EMA/NFE predictions never set the range.  Empty eroded masks return
    None instead of introducing background into the score.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('SSIM requires matching QHW arrays')
    active = mask.bool()
    if not active.any() or min(active.shape) < window:
        return None
    inside = F.avg_pool2d(active[None, None].float(), window, stride=1)[0, 0] >= 1.
    if not inside.any():
        return None
    scale = torch.quantile(target[:, active].float().reshape(-1), .995).clamp_min(1e-6)
    c1, c2 = .01**2, .03**2
    values = []
    for p, t in zip(prediction.split(32), target.split(32)):
        p, t = p[:, None].float() / scale, t[:, None].float() / scale
        pool = lambda x: F.avg_pool2d(x, window, stride=1)
        mp, mt = pool(p), pool(t)
        vp, vt = (pool(p * p) - mp * mp).clamp_min(0), (pool(t * t) - mt * mt).clamp_min(0)
        covariance = pool(p * t) - mp * mt
        score = ((2 * mp * mt + c1) * (2 * covariance + c2)
                 / ((mp * mp + mt * mt + c1) * (vp + vt + c2)))
        values.append(score[:, 0, inside].mean(dim=1))
    return float(torch.cat(values).mean())

def _seed(seed, identity):
    digest = hashlib.sha256(str(identity).encode()).digest()
    return (int(seed) + int.from_bytes(digest[:8], 'little')) % (2**63 - 1)
