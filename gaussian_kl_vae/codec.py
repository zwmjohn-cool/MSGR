"""Differentiable posterior-mean compression and decoding for a trained run."""
from __future__ import annotations

from pathlib import Path
import json

import torch
from torch import nn

from .data import ChannelNormalization, sha256_file
from .model import GaussianAutoencoderKL
from .rendering import SharedGaussianRenderer
from spatial_gaussian_dwi.shared_model import SharedBasis


class GaussianKLVAECodec(nn.Module):
    """Known support mask + fixed shared decoder; BCHW coefficients and latents.

    A generated latent can be decoded and rendered with gradients all the way
    to the latent. The supplied support is side information, not inferred here.
    No RGB pretrained weights or B0 factors are involved.
    """

    def __init__(self, model, renderer):
        super().__init__()
        self.model, self.renderer = model, renderer

    @classmethod
    def from_run(cls, run_directory, *, checkpoint='latest', weights=None, device='cpu'):
        root=Path(run_directory).resolve()
        state=torch.load(root/'checkpoints'/f'{checkpoint}.pt',map_location=device,weights_only=False)
        config=state['config']
        # A latest checkpoint's best_weights may refer to a historical best
        # checkpoint, not the weights stored at the current step.
        selected=weights or (state.get('best_weights') if checkpoint=='best' else 'online') or 'online'
        if selected not in ('online','ema'):
            raise ValueError('weights must be online or ema')
        model=GaussianAutoencoderKL(config['model']).to(device)
        model.load_state_dict(state['ema' if selected=='ema' else 'model'])
        cache=Path(config['cache_root'])
        # A moved run is portable together with its cache subdirectory.
        if (root/'cache/manifest.json').exists(): cache=root/'cache'
        manifest_path=cache/'manifest.json'
        if sha256_file(manifest_path)!=config['signature']['cache_manifest_sha256']:
            raise ValueError('Checkpoint/cache manifest identity differs')
        manifest=json.loads(manifest_path.read_text())
        # Decoding does not need the training arrays; authenticate the exact two
        # decoding assets against the frozen manifest rather than rehashing all
        # coefficient labels each time a codec is loaded.
        for name in ('basis.npz','normalization.npz'):
            if sha256_file(cache/name)!=manifest['files_sha256'][name]:
                raise ValueError(f'Checkpoint decoding asset checksum differs: {name}')
        basis=SharedBasis.load(cache/'basis.npz')
        if basis.content_id != config['signature']['basis_content_id']:
            raise ValueError('Checkpoint/public basis identity differs')
        norm=ChannelNormalization.load(cache/'normalization.npz',basis_content_id=basis.content_id)
        renderer=SharedGaussianRenderer(basis,norm,native_shape=tuple(config['native_shape'])).to(device)
        result=cls(model,renderer).eval().requires_grad_(False)
        result.checkpoint_step=state['step']
        result.weights=selected
        return result

    def encode(self, raw_coefficients, mask):
        """Return KL posterior; use .mode() for deterministic latent storage."""
        if raw_coefficients.ndim!=4 or tuple(raw_coefficients.shape[-2:]) not in (
            self.renderer.native_shape,tuple(self.model.config.padded_shape)
        ):
            raise ValueError('Raw coefficients must use native or configured padded NCHW shape')
        normalized=self.renderer.normalization.normalize(raw_coefficients,mask)
        return self.model.encode(normalized)

    def _padded_mask(self, mask, normalized):
        if mask.ndim!=4 or mask.shape[:2]!=(len(normalized),1):
            raise ValueError('Support mask must be N,1,H,W with matching latent batch size')
        if tuple(mask.shape[-2:]) not in (self.renderer.native_shape,tuple(normalized.shape[-2:])):
            raise ValueError('Support mask must use the native or configured padded shape')
        h,w=mask.shape[-2:]
        return torch.nn.functional.pad(mask,(0,normalized.shape[-1]-w,0,normalized.shape[-2]-h))

    def decode_coefficients(self, latent, mask):
        """Return native B,R,H,W raw coefficients; decoder padding is discarded."""
        normalized=self.model.decode(latent)
        mask=self._padded_mask(mask,normalized)
        return self.renderer.inverse_coefficients(normalized,mask)

    def render(self, latent, mask, directions, bvals):
        """Decode a latent to raw B,Q,H,W DWI for arbitrary positive-b queries."""
        normalized=self.model.decode(latent)
        mask=self._padded_mask(mask,normalized)
        return self.renderer(normalized,mask,directions,bvals)
