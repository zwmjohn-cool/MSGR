"""Gaussian reconstruction plus rendered-image reconstruction/GAN for the corpus.

Gaussian coefficients use the frozen, invertible training normalization. Their
MSE is computed only at existing native Gaussian centers, equally weighting
nonempty cases. Rendered-image losses retain the original target-brain RMS
scaling and background term. The discriminator sees only rendered DWI images.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .losses import RenderedDWILoss, RenderedLossConfig


@dataclass(frozen=True)
class GaussianLossConfig(RenderedLossConfig):
    """reconstruction_weight weights Gaussian MSE; image_weight weights DWI MSE.

    All inherited KL, GAN warmup, intensity scaling and background options keep
    their existing definitions. In particular KL is averaged per latent scalar.
    """
    reconstruction_weight: float = 1.0
    image_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.image_weight < 0:
            raise ValueError("image_weight must be nonnegative")


def gaussian_reconstruction_loss(reconstructed, target, brain_mask, native_shape):
    """FP32 standardized-coefficient MSE on native active centers only.

    Average the R coefficient channels and active centers within each sample,
    then average nonempty samples. Padding and inactive centers do not enter
    either numerator or denominator. The target and mask are detached.
    """
    if reconstructed.ndim != 4 or target.shape != reconstructed.shape:
        raise ValueError("Gaussian reconstruction and GT must both have shape N,R,H,W")
    h, w = map(int, native_shape)
    n, channels, padded_h, padded_w = reconstructed.shape
    if h <= 0 or w <= 0 or h > padded_h or w > padded_w:
        raise ValueError("Native shape must fit within the Gaussian coefficient canvas")
    if brain_mask.ndim == 3:
        brain_mask = brain_mask[:, None]
    if (brain_mask.shape[:2] != (n, 1) or brain_mask.ndim != 4
            or tuple(brain_mask.shape[-2:]) not in ((h, w), (padded_h, padded_w))):
        raise ValueError("Gaussian support mask must be N,1,H,W on the native or padded canvas")
    mask = brain_mask[..., :h, :w].detach().to(device=reconstructed.device, dtype=torch.bool)
    counts = mask.sum((1, 2, 3))
    valid = counts > 0
    if not valid.any():
        return reconstructed.new_zeros((), dtype=torch.float32), valid
    predicted = reconstructed[..., :h, :w].float()
    reference = target[..., :h, :w].detach().to(device=predicted.device, dtype=torch.float32)
    difference = torch.where(mask, predicted - reference, torch.zeros((), device=predicted.device))
    per_case = difference.square().sum((1, 2, 3)) / (counts * channels).clamp_min(1)
    return per_case[valid].mean(), valid


class GaussianReconstructionLoss(nn.Module):
    """Two-optimizer objective with separate Gaussian, image, GAN and KL terms.

    ``generator_loss`` accepts standardized N,R,H,W Gaussian maps, raw N,Q,H,W
    rendered images, and native/padded support masks. Targets for both domains
    are fitted Gaussian GT; measured DWI is never an optimization input.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = GaussianLossConfig(**config) if isinstance(config, dict) else config or GaussianLossConfig()
        rendered_config = asdict(self.config)
        rendered_config["reconstruction_weight"] = rendered_config.pop("image_weight")
        self.rendered = RenderedDWILoss(RenderedLossConfig(**rendered_config))

    @property
    def discriminator(self):
        return self.rendered.discriminator

    def gan_factor(self, step):
        return self.rendered.gan_factor(step)

    def generator_loss(self, reconstructed_coefficients, target_coefficients, reconstructed_images,
                       target_images, brain_mask, queries, posterior, step):
        if reconstructed_images.ndim != 4 or target_images.shape != reconstructed_images.shape:
            raise ValueError("Rendered reconstruction and GT must both have shape N,Q,H,W")
        if len(reconstructed_images) != len(reconstructed_coefficients):
            raise ValueError("Coefficient and rendered-image batch sizes differ")
        if posterior is None or posterior.mean.shape[0] != len(reconstructed_coefficients):
            raise ValueError("KL posterior must cover every Gaussian sample, including empty cases")
        native_shape = reconstructed_images.shape[-2:]
        coefficient_mse, valid = gaussian_reconstruction_loss(
            reconstructed_coefficients, target_coefficients, brain_mask, native_shape)
        h, w = native_shape
        if brain_mask.ndim == 3:
            brain_mask = brain_mask[:, None]
        mask = brain_mask[..., :h, :w]
        if valid.any():
            query = queries[valid] if queries.ndim == 3 else queries
            rendered_loss, logs = self.rendered.generator_loss(
                reconstructed_images[valid], target_images[valid], mask[valid], query, posterior, step)
            image_mse = logs.pop("reconstruction_loss")
        else:
            kl = posterior.kl().mean() / posterior.mean[0].numel()
            rendered_loss = self.config.kl_weight * kl
            zero = rendered_loss.detach() * 0
            image_mse = zero
            logs = dict(brain_mse=zero, background_mse=zero, kl_per_scalar=kl.detach(),
                        adversarial_generator=zero, gan_factor=self.gan_factor(step), target_rms=zero)
        total = self.config.reconstruction_weight * coefficient_mse + rendered_loss
        logs.update(generator_loss=total.detach(), gaussian_reconstruction_loss=coefficient_mse.detach(),
                    reconstruction_loss=coefficient_mse.detach(), recon_loss=coefficient_mse.detach(),
                    image_loss=image_mse, img_loss=image_mse,
                    weighted_reconstruction_loss=self.config.reconstruction_weight * coefficient_mse.detach(),
                    weighted_image_loss=self.config.image_weight * image_mse,
                    weighted_kl_loss=self.config.kl_weight * logs["kl_per_scalar"],
                    weighted_generator_gan_loss=self.config.adversarial_weight * logs["gan_factor"] * logs["adversarial_generator"],
                    nonempty_cases=int(valid.sum()), empty_cases=int((~valid).sum()))
        return total, logs, valid

    def discriminator_loss(self, reconstructed_images, target_images, brain_mask, queries, step):
        """Discriminator receives only rendered DWI; fake images are detached."""
        return self.rendered.discriminator_loss(reconstructed_images, target_images, brain_mask, queries, step)
