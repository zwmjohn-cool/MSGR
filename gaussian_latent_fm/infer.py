"""Generate one cached case from source MRI and random noise only."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch

from gaussian_kl_vae.training import autocast_context, save_preview
from .data import LatentGaussianDataset, validate_cache
from .flow import sample_latent
from .model import ConditionalLatentUNet
from .train import frozen_codec, validate_decoder_identity


@torch.inference_mode()
def generate(run, identity, destination, *, weights='ema', steps=1, seed=42, device='cuda:0', method='euler'):
    run=Path(run)
    state=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    config=state['config']
    manifest=validate_cache(config['cache'])
    validate_decoder_identity(config,manifest)
    record=next((r for r in manifest['records'] if r['id']==identity),None)
    if record is None: raise ValueError(f'Case is not in this prepared cache: {identity}')
    dataset=LatentGaussianDataset(config['cache'],record['split'],records=[identity],queries_per_sample=None,manifest=manifest)
    sample=dataset[0]
    model=ConditionalLatentUNet(config['model']).to(device).eval()
    if weights not in ('ema','online'): raise ValueError('Choose ema or online weights')
    model.load_state_dict(state['ema' if weights=='ema' else 'model'])
    codec=frozen_codec(config,device)
    mean=state['latent_mean'].to(device);std=state['latent_std'].to(device)
    condition=sample['condition'][None].to(device)
    generator=torch.Generator(device=device).manual_seed(seed)
    with autocast_context(torch.device(device),config.get('mixed_precision',True)):
        generated=sample_latent(model,condition,steps=steps,generator=generator,noise_scale=config.get('noise_scale',1.),method=method)
        latent=generated.latent*std+mean
        coefficients=codec.model.decode(latent)
    mask=sample['mask'][None].to(device)
    raw_coefficients=codec.renderer.inverse_coefficients(coefficients.float(),mask)
    images=codec.renderer.forward_from_query_basis(coefficients.float(),mask,sample['query_basis'][None].to(device))
    save_preview(destination,dict(latent=latent[0].permute(1,2,0).cpu().numpy(),
                 gaussian_coefficients=raw_coefficients[0].permute(1,2,0).cpu().numpy(),
                 dwi=images[0].permute(1,2,0).cpu().numpy(),bvals=sample['bvals'].numpy(),bvecs=sample['directions'].numpy(),
                 metadata=np.array(json.dumps(dict(id=identity,weights=weights,step=state['step'],seed=seed,nfe=generated.nfe,
                 source='noise + T1/T2/B0 -> latent -> frozen VAE -> Gaussian -> raw DWI; known support mask',
                 condition_order=['T1','T2','B0'],raw_intensity=True)))))
    return Path(destination)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--case',required=True)
    p.add_argument('--output',required=True);p.add_argument('--weights',choices=['ema','online'],default='ema')
    p.add_argument('--steps',type=int,default=1);p.add_argument('--method',choices=['euler','heun'],default='euler');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();generate(a.run,a.case,a.output,weights=a.weights,steps=a.steps,seed=a.seed,method=a.method)


if __name__=='__main__':main()
