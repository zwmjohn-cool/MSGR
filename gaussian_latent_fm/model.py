"""Conditional latent x-prediction with the local Stable Diffusion U-Net.

Backbone: actual ``stable-diffusion/ldm/.../openaimodel.py::UNetModel`` and its
``SpatialTransformer`` cross attention. Only attention execution is replaced
with equivalent PyTorch SDPA; parameter names and attention mathematics match.

Condition encoder: adaptation of Q-MMTN/models/networks.py AdaEncoder and
MultiModalAttentionLayer: shared single-modality CNN, parallel dense/residual
local branches, global ViT branch, and channel-wise modality attention with
0.9/0.1 mixing. Differences are explicit: rectangular inputs, compact ViT,
GroupNorm instead of direction-conditioned AdaIN, deterministic 2-D positions,
and pooled spatial tokens. No Q-MMTN decoder, q-space conditioning, pretrained
weights, noisy-input concatenation, or ControlNet residual injections are used.
The three conditions have the fixed channel order T1, T2, B0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "sd_unet_qmmtn_hybrid_crossattention_latent_v1"


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 8
    latent_shape: tuple[int, int] = (40, 44)
    condition_shape: tuple[int, int] = (160, 176)
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    context_dim: int = 256
    context_pool: tuple[int, int] = (20, 22)
    local_channels: tuple[int, int] = (32, 64)
    hybrid_channels: int = 128
    transformer_width: int = 128
    transformer_depth: int = 4
    transformer_heads: int = 4
    patch_size: int = 16
    modal_attention_pool: tuple[int, int] = (10, 11)
    attention_head_dim: int = 32
    dropout: float = 0.0
    time_scale: float = 1000.0
    sdpa: bool = True

    def __post_init__(self):
        for name in ("latent_shape", "condition_shape", "channel_mult", "context_pool",
                     "local_channels", "modal_attention_pool"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.latent_shape) != 2 or len(self.condition_shape) != 2:
            raise ValueError("Spatial shapes must be (height, width)")
        factor = 2 ** (len(self.channel_mult) - 1)
        if any(n <= 0 or n % factor for n in self.latent_shape):
            raise ValueError("Latent H,W must divide the U-Net downsample factor exactly")
        if self.patch_size <= 0 or any(n <= 0 or n % self.patch_size or n % 4
                                       for n in self.condition_shape):
            raise ValueError("Condition H,W must divide patch_size and four exactly")
        if self.base_channels < 32 or self.base_channels % 32:
            raise ValueError("Stable Diffusion GroupNorm needs base_channels divisible by 32")
        if not self.channel_mult or any(n <= 0 for n in self.channel_mult):
            raise ValueError("channel_mult must contain positive integers")
        if self.attention_head_dim <= 0 or any(self.base_channels * n % self.attention_head_dim
                                             for n in self.channel_mult):
            raise ValueError("Each U-Net width must divide attention_head_dim")
        if self.context_dim <= 0 or self.context_dim % 4:
            raise ValueError("context_dim must be positive and divisible by four")
        if (self.transformer_width <= 0 or self.transformer_width % 4 or
                self.transformer_heads <= 0 or self.transformer_width % self.transformer_heads):
            raise ValueError("Transformer width must divide four and head count")
        if self.num_res_blocks < 1 or self.transformer_depth < 1 or self.latent_channels < 1:
            raise ValueError("Channel and layer counts must be positive")
        if len(self.local_channels) != 2 or min(self.local_channels) <= 0:
            raise ValueError("local_channels requires two positive widths")
        if self.hybrid_channels < 1 or not 0 <= self.dropout < 1 or self.time_scale <= 0:
            raise ValueError("Invalid hybrid width, dropout, or time scale")
        for name in ("context_pool", "modal_attention_pool"):
            if len(getattr(self, name)) != 2 or min(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive (height, width)")
        if any(p > n // 4 for p, n in zip(self.context_pool, self.condition_shape)):
            raise ValueError("context_pool must not exceed the quarter-resolution feature map")

    def to_dict(self):
        return asdict(self)


def _sd_components(checkout=None):
    checkout = Path(checkout or Path(__file__).resolve().parents[1] / "stable-diffusion").resolve()
    filename = checkout / "ldm/modules/diffusionmodules/openaimodel.py"
    if not filename.is_file():
        raise FileNotFoundError(f"Local Stable Diffusion checkout is missing: {filename}")
    existing = sys.modules.get("ldm")
    if existing is not None:
        locations = [Path(p).resolve() for p in getattr(existing, "__path__", ())]
        if not any(checkout in p.parents for p in locations):
            raise RuntimeError(f"Unexpected already-imported ldm package: {locations}")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    architecture = importlib.import_module("ldm.modules.diffusionmodules.openaimodel")
    attention = importlib.import_module("ldm.modules.attention")
    for module in (architecture, attention):
        if checkout not in Path(module.__file__).resolve().parents:
            raise RuntimeError(f"Wrong Stable Diffusion source imported: {module.__file__}")
    return architecture, attention


class SDPACrossAttention(nn.Module):
    """Weight-compatible execution of SD CrossAttention, including self attention."""

    def __init__(self, original):
        super().__init__()
        self.heads = original.heads
        self.scale = original.scale
        for name in ("to_q", "to_k", "to_v", "to_out"):
            setattr(self, name, getattr(original, name))

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        batch = x.shape[0]
        q, k, v = (project(value).reshape(batch, -1, self.heads,
                                         self.to_q.out_features // self.heads).transpose(1, 2)
                   for project, value in ((self.to_q, x), (self.to_k, context), (self.to_v, context)))
        allowed = None if mask is None else mask.reshape(batch, 1, 1, -1).bool()
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                  dropout_p=0.0, scale=self.scale)
        return self.to_out(attended.transpose(1, 2).reshape(batch, x.shape[1], -1))


def _configure_attention(module, attention_module, use_sdpa):
    for name, child in module.named_children():
        if isinstance(child, attention_module.BasicTransformerBlock):
            # The 2022 custom checkpoint implementation does not restore AMP
            # state on recomputation. Avoid silently incorrect BF16 gradients.
            child.checkpoint = False
        if use_sdpa and isinstance(child, attention_module.CrossAttention):
            setattr(module, name, SDPACrossAttention(child))
        else:
            _configure_attention(child, attention_module, use_sdpa)


def _normalization(channels):
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, incoming, outgoing, *, stride=1):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, stride=stride),
                         _normalization(outgoing), nn.SiLU())


class LocalHybridBlock(nn.Module):
    """Parallel dense and residual branches, averaged as in Q-MMTN AdaEncoder."""

    def __init__(self, channels, outgoing):
        super().__init__()
        growth = max(4, channels // 4)
        self.dense = nn.ModuleList([ConvNormAct(channels + i * growth, growth)
                                    for i in range(3)])
        self.dense_projection = nn.Conv2d(channels + 3 * growth, outgoing, 1)
        self.residual = nn.Sequential(ConvNormAct(channels, channels),
                                       nn.Conv2d(channels, channels, 3, padding=1))
        self.residual_projection = ConvNormAct(channels, outgoing)

    def forward(self, values):
        dense = values
        for layer in self.dense:
            dense = torch.cat((dense, layer(dense)), dim=1)
        return .5 * (self.dense_projection(dense) +
                     self.residual_projection(values + self.residual(values)))


def _positions(height, width, channels):
    row, col = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    frequency = torch.exp(-math.log(10000.) * torch.arange(channels // 4).float()
                           / (channels // 4))
    row_phase = row.flatten().float()[:, None] * frequency
    col_phase = col.flatten().float()[:, None] * frequency
    return torch.cat((row_phase.sin(), row_phase.cos(), col_phase.sin(), col_phase.cos()), -1)[None]


class SourceTransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width * 4, width), nn.Dropout(dropout))

    def forward(self, tokens):
        batch, count, width = tokens.shape
        q, k, v = self.qkv(self.attention_norm(tokens)).reshape(
            batch, count, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        attended = attended.transpose(1, 2).reshape(batch, count, width)
        tokens = tokens + self.dropout(self.projection(attended))
        return tokens + self.mlp(self.mlp_norm(tokens))


class ModalAttentionFusion(nn.Module):
    """Channel-wise softmax plus the author's 0.9 individual / 0.1 fused mix."""

    def __init__(self, incoming, outgoing, pool):
        super().__init__()
        self.pool = tuple(pool)
        descriptors = 3 * math.prod(pool)
        self.attention = nn.Sequential(nn.Linear(descriptors, max(1, descriptors // 16), bias=False),
                                       nn.ReLU(), nn.Linear(max(1, descriptors // 16), 3, bias=False),
                                       nn.Sigmoid())
        self.projection = nn.Conv2d(3 * incoming, outgoing, 1)

    def forward(self, features):
        batch, modalities, channels, height, width = features.shape
        if modalities != 3:
            raise ValueError("Condition order is T1, T2, B0")
        pooled = F.adaptive_avg_pool2d(features.reshape(batch * 3, channels, height, width), self.pool)
        descriptors = pooled.reshape(batch, 3, channels, -1).permute(0, 2, 1, 3).flatten(2)
        weights = self.attention(descriptors).softmax(-1).transpose(1, 2)
        fused = (features * weights[..., None, None]).sum(1, keepdim=True)
        mixed = .9 * features + .1 * fused
        return self.projection(mixed.reshape(batch, 3 * channels, height, width))


class QMMTNHybridEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        first, second = config.local_channels
        self.stem = nn.Sequential(ConvNormAct(1, first, stride=2),
                                   ConvNormAct(first, second, stride=2))
        self.local = LocalHybridBlock(second, config.hybrid_channels)
        self.patch_embed = nn.Conv2d(1, config.transformer_width, config.patch_size,
                                     stride=config.patch_size)
        self.register_buffer("patch_positions", _positions(
            *(n // config.patch_size for n in config.condition_shape),
            config.transformer_width), persistent=False)
        self.transformer = nn.Sequential(*[SourceTransformerBlock(
            config.transformer_width, config.transformer_heads, config.dropout)
            for _ in range(config.transformer_depth)])
        self.global_projection = nn.Conv2d(config.transformer_width, config.hybrid_channels, 1)
        self.fusion = ModalAttentionFusion(2 * config.hybrid_channels, config.context_dim,
                                           config.modal_attention_pool)
        self.register_buffer("token_positions", _positions(*config.context_pool, config.context_dim),
                             persistent=False)
        self.token_norm = nn.LayerNorm(config.context_dim)

    def forward(self, conditions):
        if conditions.ndim != 4 or conditions.shape[1] != 3 or tuple(conditions.shape[-2:]) != self.config.condition_shape:
            raise ValueError(f"Conditions must be N x 3 x {self.config.condition_shape}, ordered T1,T2,B0")
        batch, _, height, width = conditions.shape
        single_modalities = conditions.reshape(batch * 3, 1, height, width)
        local = self.local(self.stem(single_modalities))
        patches = self.patch_embed(single_modalities)
        tokens = self.transformer(patches.flatten(2).transpose(1, 2) +
                                   self.patch_positions.to(patches.dtype))
        global_features = self.global_projection(tokens.transpose(1, 2).reshape(
            batch * 3, self.config.transformer_width, *patches.shape[-2:]))
        global_features = F.interpolate(global_features, local.shape[-2:], mode="bilinear",
                                         align_corners=False)
        features = torch.cat((local, global_features), dim=1)
        features = self.fusion(features.reshape(batch, 3, *features.shape[1:]))
        tokens = F.adaptive_avg_pool2d(features, self.config.context_pool).flatten(2).transpose(1, 2)
        return self.token_norm(tokens + self.token_positions.to(tokens.dtype))


class ConditionalLatentUNet(nn.Module):
    def __init__(self, config: ModelConfig | dict | None = None, *, sd_checkout=None):
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**(config or {}))
        config = self.config
        architecture, attention = _sd_components(sd_checkout)
        self.condition_encoder = QMMTNHybridEncoder(config)
        self.unet = architecture.UNetModel(
            image_size=config.latent_shape[0], in_channels=config.latent_channels,
            out_channels=config.latent_channels, model_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=tuple(2 ** n for n in range(len(config.channel_mult))),
            channel_mult=config.channel_mult, dropout=config.dropout,
            use_checkpoint=False, use_fp16=False,
            num_head_channels=config.attention_head_dim,
            use_spatial_transformer=True, transformer_depth=1,
            context_dim=config.context_dim, legacy=False)
        _configure_attention(self.unet, attention, config.sdpa)

    def encode_condition(self, conditions):
        return self.condition_encoder(conditions)

    def forward_with_context(self, noisy, t, context):
        expected = (self.config.latent_channels, *self.config.latent_shape)
        if noisy.ndim != 4 or tuple(noisy.shape[1:]) != expected:
            raise ValueError(f"Noisy latent must be N x {expected}")
        if t.ndim != 1 or t.shape[0] != noisy.shape[0]:
            raise ValueError("t must have one scalar per batch item")
        if context.ndim != 3 or context.shape[0] != noisy.shape[0] or context.shape[-1] != self.config.context_dim:
            raise ValueError("Context must have shape N x tokens x context_dim")
        # Native flow time is 0=noise, 1=data. SD sinusoidal embedding accepts
        # continuous timesteps; scaling only sets its frequency convention.
        return self.unet(noisy, t.float() * self.config.time_scale, context=context)

    def forward(self, noisy, t, conditions):
        return self.forward_with_context(noisy, t, self.encode_condition(conditions))
