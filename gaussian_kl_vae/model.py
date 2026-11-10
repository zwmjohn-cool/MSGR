"""Stable Diffusion's KL autoencoder applied to spatial Gaussian coefficients.

The local ``stable-diffusion`` checkout supplies the actual Encoder, Decoder,
and posterior distribution.  We do not import its Lightning training wrapper,
RGB perceptual loss, pretrained weights, or diffusion model.  Tensor layout is
NCHW throughout; padding is bottom/right only and never rescales coordinates.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib
import sys

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAEConfig:
    input_channels: int = 22
    latent_channels: int = 8
    downsample_factor: int = 4
    base_channels: int = 64
    num_res_blocks: int = 2
    padded_shape: tuple[int, int] = (160, 176)
    dropout: float = 0.0
    attention_type: str = "vanilla"
    sdpa_attention: bool = True

    def __post_init__(self):
        if self.downsample_factor not in (4, 8):
            raise ValueError("Only spatial downsampling factors 4 and 8 are supported")
        if self.input_channels <= 0 or self.latent_channels <= 0:
            raise ValueError("Channel counts must be positive")
        if self.base_channels < 32 or self.base_channels % 32:
            raise ValueError("SD GroupNorm requires base_channels to be a multiple of 32")
        if self.num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        if len(self.padded_shape) != 2 or any(
            n <= 0 or n % self.downsample_factor for n in self.padded_shape
        ):
            raise ValueError("padded_shape must be positive and divisible by downsample_factor")

    @property
    def latent_shape(self):
        return (
            self.latent_channels,
            self.padded_shape[0] // self.downsample_factor,
            self.padded_shape[1] // self.downsample_factor,
        )


def _sd_components(checkout=None):
    checkout = Path(checkout or Path(__file__).resolve().parents[1] / "stable-diffusion").resolve()
    if not (checkout / "ldm/modules/diffusionmodules/model.py").is_file():
        raise FileNotFoundError(f"Stable Diffusion architecture checkout is missing: {checkout}")
    existing = sys.modules.get("ldm")
    if existing is not None:
        # This checkout uses a PEP 420 namespace package, hence __file__ is None.
        locations = [Path(path).resolve() for path in getattr(existing, "__path__", ())]
        if not any(checkout in location.parents for location in locations):
            raise RuntimeError(f"Another ldm package is already imported from {locations}")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    architecture = importlib.import_module("ldm.modules.diffusionmodules.model")
    distributions = importlib.import_module("ldm.modules.distributions.distributions")
    for component in (architecture, distributions):
        if checkout not in Path(component.__file__).resolve().parents:
            raise RuntimeError(f"Loaded unexpected Stable Diffusion component: {component.__file__}")
    return architecture.Encoder, architecture.Decoder, distributions.DiagonalGaussianDistribution


class SDPAIntegration(nn.Module):
    """Same single-head attention and parameter names as SD's AttnBlock.

    PyTorch dispatches SDPA to a memory-efficient kernel when supported; this
    changes attention execution, not its mathematical definition or weights.
    """

    def __init__(self, original):
        super().__init__()
        self.in_channels = original.in_channels
        for name in ("norm", "q", "k", "v", "proj_out"):
            setattr(self, name, getattr(original, name))

    def forward(self, x):
        normalized = self.norm(x)
        b, c, h, w = x.shape
        q, k, v = (projection(normalized).flatten(2).transpose(1, 2)[:, None]
                   for projection in (self.q, self.k, self.v))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = attended[:, 0].transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj_out(attended)


def _install_sdpa(module):
    for name, child in module.named_children():
        if child.__class__.__name__ == "AttnBlock" and child.__class__.__module__ == "ldm.modules.diffusionmodules.model":
            setattr(module, name, SDPAIntegration(child))
        else:
            _install_sdpa(child)


class GaussianAutoencoderKL(nn.Module):
    """Trainable KL VAE; outputs Gaussian coefficient maps, not DWI images.

    Normalize coefficient channels using frozen training statistics before
    ``forward``; inverse normalization and Gaussian rendering belong to the
    training pipeline.  The decoder output is unrestricted (signed coefficients).
    ``decode`` without ``output_shape`` returns the full padded coefficient map.
    """

    def __init__(self, config=None, *, sd_checkout=None):
        super().__init__()
        self.config = VAEConfig(**config) if isinstance(config, dict) else config or VAEConfig()
        Encoder, Decoder, self.posterior_type = _sd_components(sd_checkout)
        cfg = self.config
        architecture = dict(
            double_z=True,
            z_channels=cfg.latent_channels,
            resolution=cfg.padded_shape[0],
            in_channels=cfg.input_channels,
            out_ch=cfg.input_channels,
            ch=cfg.base_channels,
            ch_mult=(1, 2, 4) if cfg.downsample_factor == 4 else (1, 2, 4, 4),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=(),
            dropout=cfg.dropout,
            attn_type=cfg.attention_type,
        )
        self.encoder = Encoder(**architecture)
        self.decoder = Decoder(**architecture)
        if cfg.sdpa_attention:
            _install_sdpa(self.encoder)
            _install_sdpa(self.decoder)
        # Identical moment/posterior projections to ldm.models.AutoencoderKL.
        self.quant_conv = nn.Conv2d(2 * cfg.latent_channels, 2 * cfg.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)

    def configuration(self):
        return asdict(self.config)

    def pad(self, coefficients):
        if coefficients.ndim != 4 or coefficients.shape[1] != self.config.input_channels:
            raise ValueError(f"Expected N,{self.config.input_channels},H,W coefficients")
        h, w = coefficients.shape[-2:]
        target_h, target_w = self.config.padded_shape
        if h > target_h or w > target_w:
            raise ValueError(f"Input {(h, w)} exceeds configured padded shape {self.config.padded_shape}")
        return F.pad(coefficients, (0, target_w - w, 0, target_h - h))

    def encode(self, coefficients):
        moments = self.quant_conv(self.encoder(self.pad(coefficients)))
        # Keep KL and reparameterization arithmetic FP32 under autocast, avoiding
        # exp(logvar) overflow in half precision.  The native distribution clamps
        # log variance to [-30,20], exactly as in Stable Diffusion.
        return self.posterior_type(moments.float())

    def decode(self, latent, output_shape=None):
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.config.latent_shape:
            raise ValueError(f"Expected N,{self.config.latent_shape} latent tensor")
        reconstructed = self.decoder(self.post_quant_conv(latent))
        if output_shape is not None:
            h, w = map(int, output_shape)
            if h <= 0 or w <= 0 or h > reconstructed.shape[-2] or w > reconstructed.shape[-1]:
                raise ValueError("output_shape must be a positive crop within the padded canvas")
            reconstructed = reconstructed[..., :h, :w]
        return reconstructed

    def forward(self, coefficients, sample_posterior=True):
        posterior = self.encode(coefficients)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latent, coefficients.shape[-2:]), posterior

    def get_last_layer(self):
        return self.decoder.conv_out.weight
