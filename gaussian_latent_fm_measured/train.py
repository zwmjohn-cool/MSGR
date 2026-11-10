"""Train conditional Gaussian latent flow with measured-DWI image targets.

Time runs from noise at t=0 to clean latent at t=1. Training uses clean-latent
L1 plus rendered image L1 against measured DWI with a frozen decoder.
Historical checkpoints may carry a fitted-target objective transition record.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch
from gaussian_kl_vae.corpus_data import EpochBatchSampler
from gaussian_kl_vae.data import sha256_file
from gaussian_kl_vae.training import append_json, autocast_context, now, save_checkpoint, write_json
from gaussian_latent_fm.losses import objective
from gaussian_latent_fm.flow import interpolate, sample_time
from gaussian_latent_fm.model import ConditionalLatentUNet, ModelConfig
from gaussian_latent_fm.train import frozen_codec, make_loader, validate_decoder_identity
from .data import MeasuredLatentGaussianDataset, load_monitor_dataset
ROOT = Path(__file__).resolve().parents[1]

def reconcile_resumed_logs(out, checkpoint_step):
    """Archive work beyond the saved weights before appending resumed progress."""
    from gaussian_latent_fm.logging import _jsonl, _atomic
    out = Path(out)
    archive = out / 'resume_archive' / now().replace(':', '')
    archived = []
    for filename in ('metrics.jsonl', 'eval_history.jsonl'):
        path = out / filename
        rows = _jsonl(path)
        retained = {}
        for row in rows:
            if int(row['step']) <= checkpoint_step:
                key = (row['step'], row['weights'], row['nfe']) if filename == 'eval_history.jsonl' else row['step']
                retained[key] = row
        if len(retained) != len(rows):
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / filename)
            archived.append(filename)
            kept = sorted(retained.values(), key=lambda row: row['step'])
            _atomic(path, ''.join((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in kept)))
    preview = out / 'preview.npz'
    if preview.exists():
        with np.load(preview, allow_pickle=False) as source:
            preview_step = int(source['step'])
        if preview_step > checkpoint_step:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(preview, archive / preview.name)
            archived.append(preview.name)
    for path in (out / 'evaluations').glob('step*.json'):
        if int(path.name.split('_')[0].removeprefix('step')) > checkpoint_step:
            (archive / 'evaluations').mkdir(parents=True, exist_ok=True)
            shutil.move(path, archive / 'evaluations' / path.name)
            archived.append(str(path.relative_to(out)))
    progress = out / 'monitor_progress.json'
    if progress.exists() and json.loads(progress.read_text()).get('step', 0) > checkpoint_step:
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(progress, archive / progress.name)
        archived.append(progress.name)
    if archived:
        write_json(archive / 'recovery.json', dict(checkpoint_step=checkpoint_step, archived=archived, reason='Discarded optimizer updates are preserved here, outside resumed training history'))
    return archived

def source_signature():
    folders = ('gaussian_latent_fm_measured', 'gaussian_latent_fm', 'gaussian_kl_vae',
               'gaussian_latent_pmf', 'spatial_gaussian_dwi', 'stable-diffusion/ldm')
    paths = [p for folder in folders for p in (ROOT / folder).rglob('*.py')]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}


def training_objective(model, codec, batch, mean, scale, config, *, t=None, noise=None):
    """Original FM latent L1 plus aligned measured-DWI rendered L1.

    Decoder parameters remain frozen; its input and the rendered prediction
    retain their gradient path to the FM network and condition encoder.
    """
    if config.get('image_target') != 'measured_dwi':
        raise ValueError('This objective revision requires image_target=measured_dwi')
    target = batch['measured'].detach().float()
    if target.ndim != 4 or target.shape[:2] != batch['query_basis'].shape[:2]:
        raise ValueError('Measured DWI must match the batch and sampled query order')
    device = batch['z'].device
    clean = batch['z'].float()
    if t is None:
        t = sample_time(len(clean), device=device, mean=config.get('time_mean', -0.8), std=config.get('time_std', 0.8))
    if noise is None:
        noise = torch.randn_like(clean) * float(config.get('noise_scale', 1.0))
    noisy = interpolate(clean, noise, t)
    with autocast_context(device, config.get('mixed_precision', True)):
        predicted = model(noisy, t, batch['condition'])
        coefficients = codec.model.decode(predicted.float() * scale + mean)
    generated = codec.renderer.forward_from_query_basis(coefficients.float(), batch['mask'], batch['query_basis'])
    loss, logs = objective(predicted, clean, generated, target, batch['mask'], x_weight=config.get('x_weight', 1.0), image_weight=config.get('image_weight', 1.0), background_weight=config.get('background_weight', 0.1))
    logs['time_mean'] = t.mean().detach()
    return (loss, logs)

def make_run_config(config, training, monitoring, decoder_digest):
    """Resolve immutable continuation settings without changing FM training math."""
    config = deepcopy(config)
    if config.get('image_target') != 'measured_dwi':
        raise ValueError('Measured-DWI supervision must be explicit in the run config')
    config['model'] = asdict(ModelConfig(**config.get('model', {})))
    config['monitor_counts'] = list(config.get('monitor_counts', (1, 2, 4, 8)))
    config['monitor_sampler'] = config.get('monitor_sampler', 'euler')
    if config['monitor_counts'] != [1, 2, 4, 8] or config['monitor_sampler'] != 'euler':
        raise ValueError('This monitor uses Euler sampling with exact NFE 1, 2, 4, and 8')
    config['signature'] = dict(sources=source_signature(), decoder_sha256=decoder_digest, cache_manifest_sha256=sha256_file(Path(config['cache']) / 'manifest.json'), monitor_identity=monitoring.monitor_identity)
    config['data'] = dict(training_subjects=800, training_slices=len(training), external_monitor_subjects=len(monitoring), condition_order=['T1', 'T2', 'B0'], latent_shape=[40, 44, 8], gaussian_shape=[160, 176, 22], initialization='random initialization; resume restores the exact saved state', image_target='measured raw-intensity DWI at the sampled acquisition indices', image_input='direct x_theta -> inverse latent normalization -> frozen EMA VAE decoder', loss='standardized latent L1 + measured-DWI-RMS-scaled rendered L1 (brain + 0.1 background)', primary_metric='subject-equal mean of masked all-query NRMSE versus measured DWI', monitor_sampling='uniform Euler intervals from t=0 noise to t=1 clean; NFE=1,2,4,8; fixed per-case noise seed', primitive_support='existing known native brain support', signal='restored raw intensity, no B0 division or multiplication', used_for_selection=False)
    return config

def monitor_image_target(out, step):
    """Describe the weights' actual supervision at a recorded boundary."""
    path = Path(out) / 'objective_transitions.json'
    if not path.exists():
        return 'measured DWI'
    transitions = json.loads(path.read_text())
    if not isinstance(transitions, list) or len(transitions) != 1 or transitions[0].get('from') != 'fitted_gaussian_gt' or (transitions[0].get('to') != 'measured_dwi') or (type(transitions[0].get('first_new_step')) is not int) or (transitions[0]['first_new_step'] != transitions[0].get('last_old_step', -2) + 1):
        raise ValueError('Invalid fitted-Gaussian-to-measured-DWI transition')
    return 'fitted Gaussian GT' if step < transitions[0]['first_new_step'] else 'measured DWI'

def train(config, out, *, resume=False, monitor_on_resume=False):
    from .evaluate import evaluate
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'config.json').exists() and (not resume):
        raise ValueError('Run already exists; use --resume or a new directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(config['physical_gpu']):
        raise ValueError('CUDA_VISIBLE_DEVICES must restrict this process to the configured physical GPU')
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    seed = int(config['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(json.dumps(dict(stage='validating_data', gpu=str(config['physical_gpu']))), flush=True)
    training = MeasuredLatentGaussianDataset(config['cache'], 'train', measured_cache=config['measured_cache'], queries_per_sample=config['queries_per_sample'], seed=seed)
    if training.subject_count != 800 or len(training) != 87471:
        raise ValueError('Expected all 800 training subjects / 87471 slices')
    monitoring = load_monitor_dataset(config['monitor_cache'], training.cache_manifest, seed=seed, rest_root=config['rest_root'])
    mean = training.latent_mean.to(device).float().reshape(1, 8, 1, 1)
    scale = training.latent_std.to(device).float().reshape(1, 8, 1, 1)
    decoder_digest = validate_decoder_identity(config, training.cache_manifest)
    validate_decoder_identity(config, monitoring.cache_manifest)
    config = make_run_config(config, training, monitoring, decoder_digest)
    if resume:
        if json.loads((out / 'config.json').read_text()) != json.loads(json.dumps(config)):
            raise ValueError('Frozen config/source/cache changed; refusing silent resume')
    else:
        write_json(out / 'config.json', config)
        write_json(out / 'monitor_manifest.json', dict(identity=monitoring.monitor_identity, records=monitoring.records))
        for relative in config['signature']['sources']:
            destination = out / 'source_at_launch' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
    model = ConditionalLatentUNet(config['model']).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    codec = frozen_codec(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']), betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    step, epoch, offset, seen, elapsed_before = (0, 0, 0, 0, 0.0)
    started_utc = now()
    if resume:
        state = torch.load(out / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        if json.loads(json.dumps(state['config'])) != json.loads(json.dumps(config)):
            raise ValueError('Checkpoint does not belong to this frozen run')
        if not torch.equal(state['latent_mean'], mean.cpu()) or not torch.equal(state['latent_std'], scale.cpu()):
            raise ValueError('Checkpoint normalization differs')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        step, epoch, offset, seen = (int(state[k]) for k in ('step', 'epoch', 'sample_offset', 'seen_samples'))
        elapsed_before, started_utc = (float(state['elapsed_seconds']), state['started_utc'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device)
        del state
        reconcile_resumed_logs(out, step)
    sampler = EpochBatchSampler(len(training), int(config['batch_size']), seed=seed, epoch=epoch, offset=offset)
    loader = make_loader(training, sampler, config)
    started = time.monotonic()
    max_epochs, max_hours = (int(config['max_epochs']), float(config['max_hours']))
    total_steps = math.ceil(len(training) / int(config['batch_size'])) * max_epochs
    stop_requested, last_monitor = (False, -1)
    old_handlers, durations = ({}, [])

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def state_dict():
        return dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), config=config, step=step, epoch=epoch, sample_offset=offset, seen_samples=seen, elapsed_seconds=elapsed(), started_utc=started_utc, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device), latent_mean=mean.cpu(), latent_std=scale.cpu())

    def status(stage, **extra):
        write_json(out / 'status.json', dict(stage=stage, status=stage, step=step, total_steps=total_steps, epoch=epoch, total_epochs=max_epochs, sample_offset=offset, seen_samples=seen, equivalent_data_passes=seen / len(training), elapsed_seconds=elapsed(), max_hours=max_hours, updated_utc=now(), started_utc=started_utc, pid=os.getpid(), physical_gpu=str(config['physical_gpu']), train_subjects=800, train_slices=len(training), monitor_subjects=len(monitoring), model_parameters=sum((p.numel() for p in model.parameters())), **extra))

    def monitor():
        nonlocal last_monitor
        status('external_monitoring')
        last_refresh = time.monotonic()

        def progress_callback(progress):
            nonlocal last_refresh
            if time.monotonic() - last_refresh >= 20:
                status('external_monitoring', monitor_progress=progress)
                last_refresh = time.monotonic()
        with torch.random.fork_rng(devices=[0]):
            rows = evaluate({'online': model, 'ema': ema}, codec, monitoring, mean, scale, out=out, step=step, epoch=seen / len(training), seed=42, mixed_precision=config.get('mixed_precision', True), counts=config['monitor_counts'], noise_scale=float(config.get('noise_scale', 1.0)), method=config['monitor_sampler'], training_image_target=monitor_image_target(out, step), progress_callback=progress_callback)
        last_monitor = step
        print(json.dumps(dict(evaluation=rows), ensure_ascii=False), flush=True)
        status('training')
    try:
        status('initializing')
        model.train()
        if resume and monitor_on_resume:
            monitor()
        while epoch < max_epochs and elapsed() < max_hours * 3600 and (not stop_requested):
            sampler.set_position(epoch, offset)
            fetched = time.monotonic()
            for cpu_batch in loader:
                if stop_requested or elapsed() >= max_hours * 3600:
                    break
                tick = time.monotonic()
                wait = tick - fetched
                batch = {k: cpu_batch[k].to(device, non_blocking=True) for k in ('z', 'condition', 'measured', 'mask', 'query_basis')}
                warmup = min(1.0, (step + 1) / max(1, int(config.get('warmup_steps', 1000))))
                phase = min(1.0, elapsed() / (max_hours * 3600))
                lr = float(config['learning_rate']) * warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * phase)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss, logs = training_objective(model, codec, batch, mean, scale, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('gradient_clip', 1.0)), error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    decay = min(float(config.get('ema_decay', 0.999)), (1 + step) / (10 + step))
                    for average, current in zip(ema.parameters(), model.parameters()):
                        average.lerp_(current, 1 - decay)
                    for average, current in zip(ema.buffers(), model.buffers()):
                        average.copy_(current)
                count = len(batch['z'])
                offset, seen, step = (offset + count, seen + count, step + 1)
                if offset == len(training):
                    epoch, offset = (epoch + 1, 0)
                torch.cuda.synchronize(device)
                compute = time.monotonic() - tick
                durations.append(wait + compute)
                row = {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in logs.items()}
                row.update(step=step, epoch=epoch, sample_offset=offset, seen_samples=seen, image_target='measured_dwi', equivalent_data_passes=seen / len(training), elapsed_seconds=elapsed(), created_utc=now(), data_wait_seconds=wait, compute_seconds=compute, seconds=wait + compute, learning_rate=lr, gradient_norm=float(grad), batch_size=count, queries_per_sample=config['queries_per_sample'], gpu_allocated_gib=torch.cuda.memory_allocated() / 2 ** 30, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                append_json(out / 'metrics.jsonl', row)
                status('training', latest=row, mean_step_seconds=float(np.mean(durations[-100:])))
                if step == 1 or step % 10 == 0:
                    print(json.dumps(row), flush=True)
                if step == int(config.get('first_monitor_step', 100)) or step % int(config['monitor_every']) == 0:
                    save_checkpoint(out / 'checkpoints/latest.pt', state_dict())
                    monitor()
                if step % int(config['checkpoint_every']) == 0:
                    save_checkpoint(out / 'checkpoints/latest.pt', state_dict())
                fetched = time.monotonic()
        if not stop_requested and last_monitor != step:
            monitor()
        save_checkpoint(out / 'checkpoints/latest.pt', state_dict())
        stage = 'stopped' if stop_requested else 'complete' if epoch >= max_epochs else 'time_budget_reached'
        if not stop_requested:
            save_checkpoint(out / 'checkpoints/final.pt', state_dict())
        status(stage)
    except Exception as error:
        save_checkpoint(out / 'checkpoints/interrupted.pt', state_dict())
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if getattr(loader, '_iterator', None) is not None:
            loader._iterator._shutdown_workers()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--monitor-on-resume', action='store_true')
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.output, resume=args.resume, monitor_on_resume=args.monitor_on_resume)
if __name__ == '__main__':
    main()
