"""KL regularization and reconstruction/adversarial losses in rendered DWI.

Both image targets and adversarial real examples are renders of fitted Gaussian
GT, never the noisy measured DWI.  This module accepts rendered N,Q,H,W tensors;
it does not render or reconstruct Gaussian parameters itself.  Each sample's
target brain RMS is detached and applied unchanged to target and reconstruction.
This keeps raw-intensity cases comparable without dividing DWI by B0.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RenderedLossConfig:
    reconstruction_weight: float = 1.0
    background_weight: float = 0.1
    kl_weight: float = 1e-4
    adversarial_weight: float = 0.01
    discriminator_start: int = 500
    discriminator_ramp: int = 1000
    discriminator_channels: int = 32
    discriminator_layers: int = 3
    minimum_intensity_scale: float = 1.0
    bvalue_scale: float = 3000.0

    def __post_init__(self):
        for name in ("reconstruction_weight", "background_weight", "kl_weight", "adversarial_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.discriminator_start < 0 or self.discriminator_ramp < 0:
            raise ValueError("Discriminator schedule must be nonnegative")
        if self.minimum_intensity_scale <= 0 or self.bvalue_scale <= 0:
            raise ValueError("Intensity and b-value scales must be positive")


class QueryPatchDiscriminator(nn.Module):
    """PatchGAN on one DWI direction at a time, conditioned on that query.

    Queries are [...,4] = (gx,gy,gz,b) with b in s/mm2.  The six independent
    components of gg^T make conditioning antipodally symmetric; b/3000 adds
    shell information.  No structural or measured image is exposed to D.
    GroupNorm avoids mutable running statistics during the generator update.
    """

    def __init__(self, channels=32, layers=3, bvalue_scale=3000.0):
        super().__init__()
        if channels <= 0 or layers < 1:
            raise ValueError("Discriminator channels/layers must be positive")
        self.bvalue_scale = float(bvalue_scale)
        sequence = []
        previous = 8  # One rendered image plus seven query features.
        for level in range(layers):
            width = channels * min(2**level, 8)
            sequence.append(nn.Conv2d(previous, width, kernel_size=4, stride=2, padding=1))
            if level:
                sequence.append(nn.GroupNorm(gcd(32, width), width))
            sequence.append(nn.LeakyReLU(0.2, inplace=False))
            previous = width
        sequence.append(nn.Conv2d(previous, 1, kernel_size=3, padding=1))
        self.network = nn.Sequential(*sequence)

    def query_features(self, queries):
        direction = F.normalize(queries[..., :3], dim=-1, eps=1e-12)
        x, y, z = direction.unbind(dim=-1)
        return torch.stack((x*x, y*y, z*z, x*y, x*z, y*z, queries[..., 3] / self.bvalue_scale), -1)

    def forward(self, images, queries):
        if images.ndim != 4 or images.shape[1] != 1 or queries.shape != (images.shape[0], 4):
            raise ValueError("Discriminator expects images[N,1,H,W] and queries[N,4]")
        features = self.query_features(queries.to(device=images.device, dtype=images.dtype))
        features = features[..., None, None].expand(-1, -1, *images.shape[-2:])
        return self.network(torch.cat((images, features), dim=1))


@contextmanager
def _frozen_parameters(module):
    flags = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class RenderedDWILoss(nn.Module):
    """Separate generator/discriminator objectives for two optimizers.

    Generator: brain MSE + background_weight * background MSE + beta * KL
    + warmed-up adversarial_weight * (-D(fake)).  MSE is averaged per sample,
    query, and valid pixel; KL is averaged per latent scalar (not summed).
    There is no Gaussian coefficient loss and no RGB LPIPS loss.

    Do not mask fake images before this loss: both the background reconstruction
    term and the discriminator must see unwanted outside-brain signal.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = RenderedLossConfig(**config) if isinstance(config, dict) else config or RenderedLossConfig()
        cfg = self.config
        self.discriminator = QueryPatchDiscriminator(
            cfg.discriminator_channels, cfg.discriminator_layers, cfg.bvalue_scale
        )

    def gan_factor(self, step):
        elapsed = int(step) - self.config.discriminator_start
        if elapsed < 0:
            return 0.0
        ramp = self.config.discriminator_ramp
        return 1.0 if ramp == 0 else min(1.0, (elapsed + 1) / ramp)

    def prepare(self, reconstructed, target, brain_mask, queries):
        if reconstructed.ndim != 4 or target.shape != reconstructed.shape:
            raise ValueError("Rendered target and reconstruction must both have shape N,Q,H,W")
        n, q, h, w = target.shape
        if brain_mask.ndim == 3:
            brain_mask = brain_mask[:, None]
        if brain_mask.shape != (n, 1, h, w):
            raise ValueError("brain_mask must be N,H,W or N,1,H,W")
        if queries.shape == (q, 4):
            queries = queries[None].expand(n, -1, -1)
        if queries.shape != (n, q, 4):
            raise ValueError("queries must be Q,4 or N,Q,4 with raw b-values")
        target = target.detach().float()
        reconstructed = reconstructed.float()
        mask = brain_mask.detach().to(device=target.device, dtype=torch.float32)
        denominator = (mask.sum((1, 2, 3)) * q).clamp_min(1)
        scale = ((target.square() * mask).sum((1, 2, 3)) / denominator).sqrt()
        scale = scale.clamp_min(self.config.minimum_intensity_scale).detach()[:, None, None, None]
        return reconstructed / scale, target / scale, mask, queries, scale

    @staticmethod
    def _masked_mean(values, mask):
        denominator = (mask.sum((1, 2, 3)) * values.shape[1]).clamp_min(1)
        per_sample = (values * mask).sum((1, 2, 3)) / denominator
        return per_sample.mean()

    def generator_loss(self, reconstructed, target, brain_mask, queries, posterior, step):
        fake, real, mask, queries, scale = self.prepare(reconstructed, target, brain_mask, queries)
        squared_error = (fake - real).square()
        brain_mse = self._masked_mean(squared_error, mask)
        background_mse = self._masked_mean(squared_error, 1 - mask)
        reconstruction = brain_mse + self.config.background_weight * background_mse
        if posterior is None:
            raise ValueError("A KL posterior is required for the KL autoencoder objective")
        kl = posterior.kl().mean() / posterior.mean[0].numel()
        factor = self.gan_factor(step)
        adversarial = fake.sum() * 0.0
        if factor and self.config.adversarial_weight:
            with _frozen_parameters(self.discriminator):
                adversarial = -self.discriminator(fake.flatten(0, 1)[:, None], queries.reshape(-1, 4)).mean()
        total = (
            self.config.reconstruction_weight * reconstruction
            + self.config.kl_weight * kl
            + self.config.adversarial_weight * factor * adversarial
        )
        logs = dict(
            generator_loss=total.detach(), reconstruction_loss=reconstruction.detach(),
            brain_mse=brain_mse.detach(), background_mse=background_mse.detach(),
            kl_per_scalar=kl.detach(), adversarial_generator=adversarial.detach(),
            gan_factor=factor, target_rms=scale.mean().detach(),
        )
        return total, logs

    def discriminator_loss(self, reconstructed, target, brain_mask, queries, step):
        factor = self.gan_factor(step)
        if not factor or not self.config.adversarial_weight:
            zero = next(self.discriminator.parameters()).sum() * 0.0
            return zero, dict(discriminator_loss=zero.detach(), gan_factor=factor)
        fake, real, _, queries, _ = self.prepare(reconstructed.detach(), target, brain_mask, queries)
        query = queries.reshape(-1, 4)
        logits_real = self.discriminator(real.flatten(0, 1)[:, None], query)
        logits_fake = self.discriminator(fake.flatten(0, 1)[:, None], query)
        loss = factor * 0.5 * (F.relu(1 - logits_real).mean() + F.relu(1 + logits_fake).mean())
        return loss, dict(discriminator_loss=loss.detach(), logits_real=logits_real.mean().detach(),
                         logits_fake=logits_fake.mean().detach(), gan_factor=factor)
