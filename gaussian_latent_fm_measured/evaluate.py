"""Complete rest-cohort monitoring for continued conditional latent Flow Matching.

Measured DWI is the primary evaluation reference and current training image
target. Fitted-Gaussian renders are a separate diagnostic evaluation target.
Each subject supplies one frozen representative slice and every acquisition.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from gaussian_kl_vae.training import (append_json, autocast_context, brain_metrics,
                                     now, save_preview, write_json)
from gaussian_latent_fm.evaluate import _seed, masked_ssim
from gaussian_latent_fm.flow import sample_latent


def validate_subject_coverage(dataset, expected_subjects=None):
    """Reject duplicate slices or omitted subjects before recording a mean."""
    if getattr(dataset, 'queries_per_sample', None) is not None:
        raise ValueError('Monitoring requires all actual acquisition queries')
    records = getattr(dataset, 'records', None)
    if records is None:
        records = [dataset[index] for index in range(len(dataset))]
    subjects = [str(record['subject']) for record in records]
    if not subjects or len(subjects) != len(dataset) or len(set(subjects)) != len(subjects):
        raise ValueError('Monitoring requires exactly one representative slice per subject')
    if expected_subjects is None:
        manifest = getattr(dataset, 'cache_manifest', {})
        expected_subjects = {str(record['subject']) for record in manifest.get('records', [])
                             if record['split'] == 'external'} or subjects
    expected = {str(subject) for subject in expected_subjects}
    if set(subjects) != expected:
        raise ValueError(f'Incomplete rest cohort: missing={sorted(expected-set(subjects))}, '
                         f'unexpected={sorted(set(subjects)-expected)}')
    return subjects


@torch.no_grad()
def evaluate(models, codec, dataset, latent_mean, latent_std, *, out, step,
             epoch, seed=42, mixed_precision=True, counts=(1, 2, 4, 8), noise_scale=1.,
             progress_callback=None, expected_subjects=None, method='euler',
             training_image_target='measured DWI'):
    """Online/EMA generation at each NFE, complete cohort, equal-subject NRMSE.

    No measured or fitted target enters the model. Conditions and fixed random
    noise determine predictions; the known support and acquisition table enter
    rendering. Progress is saved separately from complete evaluation rows.
    Use no_grad rather than inference_mode: a monitor before the first training
    forward initializes cached Gaussian geometry that later backward needs.
    """
    subjects = validate_subject_coverage(dataset, expected_subjects)
    if noise_scale <= 0 or not np.isfinite(noise_scale):
        raise ValueError('noise_scale must be finite and positive')
    if not models or any(name not in ('online', 'ema') for name in models):
        raise ValueError('Expected explicit online and/or ema models')
    counts = tuple(counts)
    if (not counts or any(isinstance(count, bool) or not isinstance(count, int)
                          or count < 1 for count in counts)):
        raise ValueError('Sampling counts must be nonempty positive integers')
    if len(set(counts)) != len(counts):
        raise ValueError('Sampling counts must not contain duplicates')
    if method != 'euler':
        raise ValueError('This monitoring protocol requires Euler so steps equal NFE')
    if any(1. / count < 1e-5 for count in counts):
        raise ValueError('Sampling intervals must exceed the FM denominator floor')
    device = next(next(iter(models.values())).parameters()).device
    mean = torch.as_tensor(latent_mean, device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    std = torch.as_tensor(latent_std, device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    if mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Frozen latent normalization is invalid')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    h, w = codec.renderer.native_shape
    states = {name: model.training for name, model in models.items()}
    codec_state = codec.training
    codec.eval()
    for model in models.values():
        model.eval()
    by_setting = {(name, count): [] for name in models for count in counts}
    if not hasattr(dataset, '_external_measured_cache'):
        dataset._external_measured_cache = {}
    measured_cache = dataset._external_measured_cache
    preview = None
    started = perf_counter()

    def progress(completed, status='evaluating', identity=None):
        state = dict(status=status, step=int(step), completed_subjects=completed,
                     total_subjects=len(subjects), weights=list(models),
                     counts=list(counts), integration_steps=list(counts), nfe=list(counts), sampler=method,
                     completed_subject_settings=sum(len(cases) for cases in by_setting.values()),
                     total_subject_settings=len(subjects) * len(by_setting),
                     current_case=identity, elapsed_seconds=perf_counter()-started,
                     updated_utc=now(), primary_metric='nrmse_measured_pct')
        write_json(out / 'monitor_progress.json', state)
        if progress_callback is not None and (completed % 25 == 0 or completed == len(subjects)):
            progress_callback(state)

    try:
        progress(0)
        for index in range(len(dataset)):
            sample = dataset[index]
            identity = str(sample['id'])
            if str(sample['subject']) != subjects[index]:
                raise ValueError('Dataset subject ordering changed during evaluation')
            x = sample['x'][None].to(device=device, dtype=torch.float32)
            mask = sample['mask'][None].to(device)
            basis = sample['query_basis'][None].to(device=device, dtype=torch.float32)
            condition = sample['condition'][None].to(device=device, dtype=torch.float32)
            clean = sample['z'][None].to(device=device, dtype=torch.float32)
            active = mask[0, 0, :h, :w].bool()
            if not active.any():
                raise ValueError(f'Empty representative slice would omit a rest subject: {identity}')
            query_count = int(len(sample['bvals']))
            if query_count < 1 or basis.shape[1] != query_count or len(sample['directions']) != query_count:
                raise ValueError(f'Acquisition table mismatch: {identity}')
            record = dataset.records[index] if hasattr(dataset, 'records') else {}
            recorded_count = record.get('query_count')
            if recorded_count is not None and int(recorded_count) != query_count:
                raise ValueError(f'Acquisitions were truncated for {identity}')
            base = dict(id=identity, subject=str(sample['subject']), slice=int(sample['slice']),
                        query_count=query_count, empty_brain=False, seed=_seed(seed, identity))
            target = codec.renderer.forward_from_query_basis(x, mask, basis)[0]
            if identity not in measured_cache:
                measured_cache[identity] = dataset.getmeasured(index).detach().cpu()
            measured = measured_cache[identity].to(device=device, dtype=torch.float32)
            if measured.shape != target.shape or measured.shape[0] != query_count:
                raise ValueError(f'Measured/GT shapes differ for {identity}')
            with autocast_context(device, mixed_precision):
                floor_coefficients = codec.model.decode(clean * std + mean)
            floor = codec.renderer.forward_from_query_basis(floor_coefficients.float(), mask, basis)[0]
            fitted_metric = brain_metrics(target, measured, active)['nrmse_percent']
            floor_fitted = brain_metrics(floor, target, active)['nrmse_percent']
            floor_measured = brain_metrics(floor, measured, active)['nrmse_percent']
            generator = torch.Generator(device=device).manual_seed(base['seed'])
            noise = torch.randn(clean.shape, device=device, generator=generator, dtype=torch.float32) * noise_scale
            preview_this_case = preview is None
            if preview_this_case:
                raw = dataset.getrawconditions(index) if hasattr(dataset, 'getrawconditions') else sample['condition'][:, :h, :w]
                raw = torch.as_tensor(raw)
                if tuple(raw.shape) != (3, h, w):
                    raise ValueError('Raw report conditions must be native [3,H,W]')
                preview = dict(id=np.array(identity), subject=np.array(sample['subject']),
                               slice=np.array(sample['slice']), step=np.array(step), seed=np.array(seed),
                               actual_noise_seed=np.array(base['seed']),
                               condition=raw.permute(1, 2, 0).cpu().numpy(),
                               measured=measured.permute(1, 2, 0).cpu().numpy(),
                               gaussian_gt=target.permute(1, 2, 0).cpu().numpy(),
                               vae_gt=floor.permute(1, 2, 0).cpu().numpy(),
                               mask=active.cpu().numpy(), bvals=sample['bvals'].cpu().numpy(),
                               bvecs=sample['directions'].cpu().numpy(),
                               query_indices=sample['query_indices'].cpu().numpy())
            for name, count in by_setting:
                model = models[name]
                sample_started = perf_counter()
                with autocast_context(device, mixed_precision):
                    result = sample_latent(model, condition, steps=count, noise=noise, method=method)
                    coefficients = codec.model.decode(result.latent.float() * std + mean)
                if result.nfe != count or result.steps != count or result.method != method:
                    raise ValueError('Euler Flow Matching must use exactly one network evaluation per interval')
                generated = codec.renderer.forward_from_query_basis(coefficients.float(), mask, basis)[0]
                diagnostic = brain_metrics(generated, target, active)
                primary = brain_metrics(generated, measured, active)
                case = dict(base, weights=name, integration_steps=count, nfe=count, sampler=method,
                            nrmse_measured_pct=primary['nrmse_percent'],
                            nrmse_gaussian_gt_pct=diagnostic['nrmse_percent'],
                            nrmse_gaussian_gt_vs_measured_pct=fitted_metric,
                            vae_floor_nrmse_gaussian_gt_pct=floor_fitted,
                            vae_floor_nrmse_measured_pct=floor_measured,
                            latent_l1=float((result.latent-clean).abs().mean()),
                            rmse_measured=primary['rmse'], mae_measured=primary['mae'],
                            ssim_measured=masked_ssim(generated, measured, active),
                            ssim_gaussian_gt=masked_ssim(generated, target, active),
                            generation_render_metrics_seconds=perf_counter()-sample_started)
                shells = torch.round(sample['bvals'].to(device) / 1000).long()
                for shell in torch.unique(shells).tolist():
                    selected = shells == shell
                    case[f'nrmse_measured_shell{shell*1000}_pct'] = brain_metrics(
                        generated[selected], measured[selected], active)['nrmse_percent']
                    case[f'nrmse_gaussian_gt_shell{shell*1000}_pct'] = brain_metrics(
                        generated[selected], target[selected], active)['nrmse_percent']
                by_setting[name, count].append(case)
                if preview_this_case:
                    preview[f'generated_{name}_nfe{count}'] = generated.permute(1, 2, 0).cpu().numpy()
            progress(index+1, identity=identity)
    except Exception:
        progress(min(len(cases) for cases in by_setting.values()), 'failed')
        raise
    finally:
        for name, model in models.items():
            model.train(states[name])
        codec.train(codec_state)
    rows = []
    for (name, count), cases in by_setting.items():
        if len(cases) != len(subjects) or {case['subject'] for case in cases} != set(subjects):
            raise ValueError('Incomplete cohort cannot publish a complete monitoring result')
        metric_keys = sorted({key for case in cases for key in case
                              if key.startswith(('nrmse_', 'vae_floor_', 'ssim_', 'rmse_', 'mae_'))
                              or key == 'latent_l1'})
        means = {}
        for key in metric_keys:
            values = [case[key] for case in cases if case.get(key) is not None]
            means[key] = float(np.mean(values)) if len(values) == len(cases) and values else None
        row = dict(step=int(step), epoch=float(epoch), weights=name, integration_steps=count,
                   nfe=count, sampler=method, seed=int(seed), created_utc=now(),
                   evaluation_counts=list(counts), evaluation_weights=list(models),
                   sampling_schedule='uniform t=0 to 1; Euler velocity=(x_theta-z)/(1-t)',
                   subjects=len(cases), nonempty_subjects=len(cases), expected_subjects=len(subjects),
                   complete_cohort=True, excluded_empty_subjects=0, used_for_selection=False,
                   protocol='all rest subjects; one frozen representative slice each; all actual queries; equal subject mean',
                   primary_target='measured DWI', primary_metric='nrmse_measured_pct',
                   training_image_target=training_image_target,
                   elapsed_seconds=perf_counter()-started, **means)
        write_json(out / 'evaluations' / f'step{step:09d}_{name}_nfe{count}.json', dict(summary=row, cases=cases))
        append_json(out / 'eval_history.jsonl', row)
        rows.append(row)
    if preview is not None:
        save_preview(out / 'preview.npz', preview)
    progress(len(subjects), 'complete')
    return rows
