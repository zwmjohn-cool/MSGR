"""The two requested L1 losses; no velocity, GAN, KL, or measured-DWI loss."""
from __future__ import annotations

import torch


def rendered_l1(prediction, target, mask, *, background_weight=0.1):
    """Brain L1 plus background L1 in detached target-RMS units.

    Outputs and targets remain raw MRI intensities. The common scale only
    conditions the image loss, making its unit weight comparable across scans
    and with the standardized latent loss. Empty support has zero image loss.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("DWI predictions and Gaussian GT must both be B,Q,H,W")
    b, q, h, w = target.shape
    if mask.shape[:2] != (b, 1) or min(mask.shape[-2:]) < min(h, w):
        raise ValueError("Expected a matching B,1,H,W native or padded support")
    mask = mask[..., :h, :w].to(device=prediction.device, dtype=torch.float32)
    if mask.shape != (b, 1, h, w):
        raise ValueError("Support cannot crop native anatomy")
    real = target.detach().float()
    active_count = mask.sum((1, 2, 3)) * q
    valid = active_count > 0
    rms = ((real.square() * mask).sum((1, 2, 3)) / active_count.clamp_min(1)).sqrt().clamp_min(1).detach()
    error = (prediction.float() - real).abs() / rms[:, None, None, None]
    brain = (error * mask).sum((1, 2, 3)) / active_count.clamp_min(1)
    background = (error * (1-mask)).sum((1, 2, 3)) / ((1-mask).sum((1, 2, 3))*q).clamp_min(1)
    count = valid.sum().clamp_min(1)
    brain_mean = (brain * valid).sum() / count
    background_mean = (background * valid).sum() / count
    total = brain_mean + float(background_weight) * background_mean
    return total, dict(brain_l1=brain_mean.detach(), background_l1=background_mean.detach(),
                      target_rms=(rms * valid).sum().detach()/count,
                      nonempty_cases=valid.sum().detach())


def objective(predicted_z, clean_z, prediction, target, mask, *, x_weight=1., image_weight=1., background_weight=.1):
    if predicted_z.shape != clean_z.shape:
        raise ValueError("Predicted and clean latent shapes differ")
    x_loss = (predicted_z.float() - clean_z.detach().float()).abs().mean()
    image_loss, logs = rendered_l1(prediction, target, mask, background_weight=background_weight)
    total = x_weight * x_loss + image_weight * image_loss
    return total, dict(x_loss=x_loss.detach(), img_loss=image_loss.detach(),
                      total_loss=total.detach(), **logs)
