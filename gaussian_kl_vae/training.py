"""Shared checkpoint, logging and training utilities."""
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from .data import sha256_file

ROOT = Path(__file__).resolve().parents[1]

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(state, temporary)
    temporary.replace(path)

def save_preview(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)

def brain_metrics(prediction, target, mask):
    """One subject, raw QHW images; denominator is target brain RMS."""
    active = torch.as_tensor(mask, device=prediction.device, dtype=torch.bool).reshape(*prediction.shape[-2:])
    x, y = prediction[:, active].double(), target[:, active].double()
    mse = (x - y).square().mean()
    power = y.square().mean()
    if not torch.isfinite(mse) or power <= 0:
        raise ValueError('Metrics require finite predictions and nonzero target brain signal')
    return dict(nrmse_percent=float(100 * torch.sqrt(mse / power)), rmse=float(mse.sqrt()),
                mae=float((x-y).abs().mean()))

def autocast_context(device, enabled):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if enabled and device.type == 'cuda' else nullcontext()

def freeze_sources(out):
    folders = ('gaussian_kl_vae', 'gaussian_latent_fm', 'gaussian_latent_fm_measured',
               'gaussian_latent_pmf', 'spatial_gaussian_dwi', 'stable-diffusion/ldm')
    names = sorted({path for folder in folders for path in (ROOT / folder).rglob('*.py')})
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in names}
    for path in names:
        destination = out / 'source_at_launch' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(out / 'source_hashes.json', hashes)
    return hashes
