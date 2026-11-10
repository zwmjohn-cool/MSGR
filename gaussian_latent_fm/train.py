"""Frozen decoder identity and data-loader helpers for measured-DWI flow training."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from gaussian_kl_vae.codec import GaussianKLVAECodec
from gaussian_kl_vae.data import sha256_file

def frozen_codec(config, device):
    return GaussianKLVAECodec.from_run(config['vae_run'], checkpoint=config['vae_checkpoint'],
                                      weights='ema', device=device).eval().requires_grad_(False)

def validate_decoder_identity(config, cache_manifest):
    source = cache_manifest['source']
    decoder = (Path(config['vae_run'])/'checkpoints'/f"{config['vae_checkpoint']}.pt").resolve()
    if decoder != Path(source['decoder_checkpoint']).resolve():
        raise ValueError('Decoder must be the exact frozen EMA snapshot used to export target latents')
    digest = sha256_file(decoder)
    if digest != source['decoder_sha256']:
        raise ValueError('Decoder checksum differs from exported latent targets')
    return digest

def make_loader(dataset, sampler, config):
    workers = int(config.get('workers', 8))
    kw = dict(dataset=dataset, batch_sampler=sampler, num_workers=workers, pin_memory=True,
              generator=torch.Generator().manual_seed(int(config['seed']) + 91))
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=int(config.get('prefetch_factor', 2)),
                  multiprocessing_context='spawn')
    return DataLoader(**kw)
