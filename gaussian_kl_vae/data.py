"""Training-only affine normalization of Gaussian coefficient channels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

NORMALIZATION_FORMAT = "gaussian_kl_vae.channel_normalization.v1"

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

class ChannelNormalization(torch.nn.Module):
    """Exact affine channel transform on active coefficients; zeros elsewhere.

    ``mask`` identifies existing Gaussian centers, not rendered output support.
    Normalization does not clip coefficients or divide DWI by B0. Inverse
    transforms preserve autograd. The finite native support is side information
    for this compression pilot; neither its storage nor its availability in a
    future unconditional generator should be assumed free.
    """

    def __init__(self, mean, std, *, basis_content_id, metadata=None, image_scale=1.):
        super().__init__()
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 1 or mean.shape != std.shape or not len(mean):
            raise ValueError("Expected channel mean/std vectors of identical nonzero length")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("Channel means/std must be finite and std positive")
        if not np.isfinite(image_scale) or image_scale <= 0:
            raise ValueError("Rendered intensity scale must be finite and positive")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("image_scale", torch.tensor(float(image_scale), dtype=torch.float32))
        self.basis_content_id = str(basis_content_id)
        self.metadata = dict(metadata or {})

    def _constants(self, x):
        if x.ndim not in (3, 4) or x.shape[-3] != self.mean.numel():
            raise ValueError("Expected coefficient [R,H,W] or [B,R,H,W]")
        shape = (1,) * (x.ndim - 3) + (-1, 1, 1)
        return self.mean.to(x).reshape(shape), self.std.to(x).reshape(shape)

    @staticmethod
    def _mask(x, mask):
        if mask is None:
            return x
        mask = torch.as_tensor(mask, device=x.device, dtype=torch.bool)
        if mask.shape != x.shape[:-3] + (1,) + x.shape[-2:]:
            raise ValueError("Mask must be [1,H,W] or [B,1,H,W], matching coefficients")
        return torch.where(mask, x, torch.zeros((), device=x.device, dtype=x.dtype))

    def normalize(self, x, mask=None):
        mean, std = self._constants(x)
        return self._mask((x - mean) / std, mask)

    def inverse(self, x, mask=None):
        mean, std = self._constants(x)
        return self._mask(x * std + mean, mask)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = dict(format=NORMALIZATION_FORMAT, basis_content_id=self.basis_content_id,
                      metadata=self.metadata, clipping=False, normalization="channel affine on active centers")
        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy(),
                 image_scale=self.image_scale.cpu().numpy(), metadata=np.array(_json(header)))
        return path

    @classmethod
    def load(cls, path, *, basis_content_id=None):
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(archive["metadata"].item())
            if header.get("format") != NORMALIZATION_FORMAT:
                raise ValueError("Unsupported coefficient normalization format")
            if basis_content_id is not None and header["basis_content_id"] != basis_content_id:
                raise ValueError("Normalization/public basis hash mismatch")
            return cls(archive["mean"].copy(), archive["std"].copy(),
                       basis_content_id=header["basis_content_id"], metadata=header["metadata"],
                       image_scale=float(archive["image_scale"]))
