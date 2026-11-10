"""Train a KL VAE over every freshly fitted slice in the 800-subject corpus.

Training reconstructs standardized Gaussian parameters and their rendered DWI.
Both targets come from fitted Gaussian GT. The frozen external monitoring
subjects remain external tests: their metrics never select checkpoints, stop
training early, change schedules, or enter optimization. Unlike the small pilot,
an epoch is one shuffled visit of every training slice, including empty slices.
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
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .corpus_data import CorpusGaussianDataset, EpochBatchSampler, select_external_monitor, validate_corpus
from .data import sha256_file
from .corpus_losses import GaussianLossConfig, GaussianReconstructionLoss, gaussian_reconstruction_loss
from .model import GaussianAutoencoderKL, VAEConfig
from .rendering import SharedGaussianRenderer
from .training import ROOT, append_json, autocast_context, brain_metrics, freeze_sources, now, save_checkpoint, save_preview, write_json

def make_loader(dataset, sampler, config):
    workers = int(config.get('workers', 8))
    kwargs = dict(dataset=dataset, batch_sampler=sampler, num_workers=workers, pin_memory=config.get('pin_memory', True), generator=torch.Generator().manual_seed(int(config.get('seed', 20261028)) + 91))
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=int(config.get('prefetch_factor', 2)), multiprocessing_context=config.get('multiprocessing_context', 'spawn'))
    return DataLoader(**kwargs)

def generator_objective(objective, reconstructed_coefficients, target_coefficients, fake, target, mask, queries, posterior, step):
    """Gaussian/image/GAN means exclude empty cases; KL includes every case."""
    return objective.generator_loss(reconstructed_coefficients, target_coefficients, fake, target, mask, queries, posterior, step)

@torch.no_grad()
def monitor(models, renderer, dataset, measured_cache, *, step, epoch, out, mixed_precision=False):
    """All acquired queries, posterior mean, metrics excluded for empty masks."""
    device = next(iter(models.values())).get_last_layer().device
    h, w = renderer.native_shape
    summaries, preview = ([], None)
    for label, model in models.items():
        was_training = model.training
        model.eval()
        cases = []
        try:
            for index in range(len(dataset)):
                sample = dataset[index]
                x = sample['x'][None].to(device)
                mask = sample['mask'][None].to(device)
                query_basis = sample['query_basis'][None].to(device)
                with autocast_context(device, mixed_precision):
                    reconstructed, posterior = model(x, sample_posterior=False)
                reconstructed = reconstructed.float()
                generated = renderer.forward_from_query_basis(reconstructed, mask, query_basis)[0]
                target = renderer.forward_from_query_basis(x, mask, query_basis)[0]
                active = mask[0, 0, :h, :w]
                case = dict(id=sample['id'], subject=sample['subject'], query_count=len(sample['bvals']), empty_brain=not bool(active.any()), latent_mean=float(posterior.mean.mean()), latent_std=float(posterior.mean.std()), posterior_std_mean=float(posterior.std.mean()))
                if not active.any():
                    case.update(nrmse_gaussian_gt_pct=None, nrmse_measured_pct=None, nrmse_gaussian_gt_vs_measured_pct=None, gaussian_reconstruction_loss=None, metric_exclusion='empty brain mask')
                    cases.append(case)
                    continue
                if index not in measured_cache:
                    measured_cache[index] = dataset.getmeasured(index)
                measured = measured_cache[index].to(device=device, dtype=torch.float32)
                if measured.shape != target.shape:
                    raise ValueError(f"External measured/fit query or spatial shape differs: {sample['id']}")
                primary = brain_metrics(generated, target, active)
                parameter_mse, _ = gaussian_reconstruction_loss(reconstructed, x, mask, renderer.native_shape)
                reference = brain_metrics(generated, measured, active)
                fitted = brain_metrics(target, measured, active)
                case.update(nrmse_gaussian_gt_pct=primary['nrmse_percent'], gaussian_reconstruction_loss=float(parameter_mse), nrmse_measured_pct=reference['nrmse_percent'], nrmse_gaussian_gt_vs_measured_pct=fitted['nrmse_percent'], rmse_gaussian_gt=primary['rmse'], mae_gaussian_gt=primary['mae'])
                shells = torch.round(sample['bvals'].to(device) / 1000).long()
                for shell in (1, 2, 3):
                    selected = shells == shell
                    if selected.any():
                        case[f'nrmse_gaussian_gt_shell{shell * 1000}_pct'] = brain_metrics(generated[selected], target[selected], active)['nrmse_percent']
                cases.append(case)
                if preview is None or preview['id'].item() == sample['id']:
                    if preview is None:
                        preview = dict(id=np.array(sample['id']), measured=measured.permute(1, 2, 0).cpu().numpy(), gaussian_gt=target.permute(1, 2, 0).cpu().numpy(), mask=active.cpu().numpy(), bvals=sample['bvals'].numpy(), bvecs=sample['directions'].numpy(), subject=np.array(sample['subject']), slice=np.array(sample['slice']), step=np.array(step), query_indices=sample['query_indices'].numpy())
                    preview['reconstruction' if label == 'online' else 'reconstruction_ema'] = generated.permute(1, 2, 0).cpu().numpy()
        finally:
            model.train(was_training)
        usable = [case for case in cases if not case['empty_brain']]
        metric_names = ['nrmse_gaussian_gt_pct', 'nrmse_measured_pct', 'nrmse_gaussian_gt_vs_measured_pct', 'gaussian_reconstruction_loss', 'rmse_gaussian_gt', 'mae_gaussian_gt', 'latent_mean', 'latent_std', 'posterior_std_mean']
        metric_names += [f'nrmse_gaussian_gt_shell{shell}_pct' for shell in (1000, 2000, 3000)]
        averages = {}
        for key in metric_names:
            values = [case[key] for case in usable if key in case and case[key] is not None]
            averages[key] = float(np.mean(values)) if values else None
        row = dict(step=step, epoch=epoch, weights=label, created_utc=now(), subjects=len(cases), nonempty_subjects=len(usable), excluded_empty_subjects=len(cases) - len(usable), protocol='external monitoring only; all acquired queries; posterior.mode; no checkpoint selection', used_for_selection=False, **averages)
        append_json(out / 'eval_history.jsonl', row)
        write_json(out / 'evaluations' / f'step{step:09d}_{label}.json', dict(summary=row, cases=cases))
        summaries.append(row)
    if preview is not None:
        save_preview(out / 'preview.npz', preview)
    return summaries

def train_corpus(config, out, cache_root, *, resume=False):
    out, cache_root = (Path(out).resolve(), Path(cache_root).resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'checkpoints/latest.pt').exists() and (not resume):
        raise ValueError('Full-corpus checkpoint exists: use --resume or a new output directory')
    manifest = validate_corpus(cache_root, verify_shard_checksums=config.get('verify_shard_checksums', False))
    seed = int(config.get('seed', 20261028))
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    torch.manual_seed(seed)
    device = torch.device(config.get('device', 'cuda:0'))
    if device.type == 'cuda':
        device = torch.device('cuda', device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    cfg, loss_cfg = (VAEConfig(**config.get('model', {})), GaussianLossConfig(**config.get('loss', {})))
    batch_size, queries = (int(config.get('batch_size', 64)), int(config.get('queries_per_sample', 12)))
    max_epochs, max_hours = (int(config.get('max_epochs', 100)), float(config.get('max_hours', 72)))
    if max_epochs < 1 or max_hours <= 0:
        raise ValueError('Training requires positive max_epochs and max_hours')
    training = CorpusGaussianDataset(cache_root, 'train', queries_per_sample=queries, seed=seed, manifest=manifest, max_open_shards=config.get('max_open_shards', 16))
    expected_subjects = int(config.get('expected_train_subjects', 800))
    if training.subject_count != expected_subjects:
        raise ValueError(f'Training requires all {expected_subjects} subjects, found {training.subject_count}')
    if tuple(cfg.padded_shape) != training.padded_shape or cfg.input_channels != training.basis.rank:
        raise ValueError('VAE configuration differs from frozen full-corpus geometry/rank')
    monitor_records = select_external_monitor(manifest, cache_root, count=int(config.get('monitor_subjects', 20)), seed=seed)
    monitor_signature = dict(records=[record['id'] for record in monitor_records], selection_seed=seed, role='external test monitoring only', used_for_selection=False)
    monitor_path = out / 'monitor_manifest.json'
    if resume and json.loads(monitor_path.read_text()) != monitor_signature:
        raise ValueError('Frozen external monitoring cohort differs')
    if not resume:
        write_json(monitor_path, monitor_signature)
    monitoring = CorpusGaussianDataset(cache_root, 'external', records=monitor_records, manifest=manifest)
    model = GaussianAutoencoderKL(cfg).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    objective = GaussianReconstructionLoss(loss_cfg).to(device)
    renderer = SharedGaussianRenderer(training.basis, deepcopy(training.normalization), native_shape=training.native_shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 0.0001), betas=(0.9, 0.99), weight_decay=0)
    disc_optimizer = torch.optim.AdamW(objective.discriminator.parameters(), lr=config.get('discriminator_learning_rate', 0.0001), betas=(0.5, 0.9), weight_decay=0)
    signature = dict(model=asdict(cfg), loss=asdict(loss_cfg), loss_version='gaussian_reconstruction_plus_rendered_image_v1', basis_content_id=training.basis.content_id, cache_manifest_sha256=sha256_file(cache_root / 'manifest.json'), normalization_sha256=sha256_file(cache_root / 'normalization.npz'), monitor_manifest_sha256=sha256_file(monitor_path), training={key: config.get(key, default) for key, default in {'seed': 20261028, 'batch_size': 64, 'queries_per_sample': 12, 'learning_rate': 0.0001, 'discriminator_learning_rate': 0.0001, 'mixed_precision': True, 'ema_decay': 0.999, 'gradient_clip': 1.0}.items()})
    total_steps = math.ceil(len(training) / batch_size) * max_epochs
    config = dict(config, model=asdict(cfg), loss=asdict(loss_cfg), signature=signature, cache_root=str(cache_root), train_subjects=training.subject_count, train_slices=len(training), val_subjects=0, monitor_subjects=monitoring.subject_count, external_subjects=len({r['subject'] for r in manifest['records'] if r['split'] == 'external'}), external_slices=sum((r['split'] == 'external' for r in manifest['records'])), scope='all_training_subjects_all_slices', max_steps=total_steps, max_epochs=max_epochs, input_shape=[*cfg.padded_shape, cfg.input_channels], native_shape=list(training.native_shape), latent_shape=[cfg.latent_shape[1], cfg.latent_shape[2], cfg.latent_shape[0]], cohort_description=f'全部 {training.subject_count} 个训练受试，共 {len(training)} 片；每 epoch 遍历全部切片。外部 {monitoring.subject_count} 人仅监测，不选择权重、不提前停止。', target='standardized freshly fitted shared-R22 Gaussian coefficients and their rendered DWI', reconstruction_domain='standardized Gaussian coefficients; native active centers only; equal nonempty-case mean', image_domain='rendered fitted-Gaussian GT DWI; target-brain RMS scaling; brain MSE plus background term', adversarial_domain='rendered reconstructed-Gaussian DWI versus rendered fitted-Gaussian GT DWI only', signal='restored raw MRI intensity; no B0 division/multiplication', mask='external primitive-support side information, identical for fitted GT and reconstruction', normalization='all training subjects only; frozen affine channel transform; inverted before rendering', checkpoint_policy='latest and final; external monitor never selects checkpoints', empty_policy='kept in every epoch; KL only; excluded from Gaussian/image/GAN means and NRMSE', trainable_parameters=sum((parameter.numel() for parameter in model.parameters())))
    step = epoch = offset = seen_samples = 0
    elapsed_before = 0.0
    if resume:
        old_config = json.loads((out / 'config.json').read_text())
        if old_config['signature'] != json.loads(json.dumps(signature)):
            raise ValueError('Full-corpus resume model/loss/data/optimization signature differs')
        for name, digest in json.loads((out / 'source_hashes.json').read_text()).items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f'Full-corpus resume source differs: {name}')
        previous = torch.load(out / 'checkpoints/latest.pt', map_location=device, weights_only=False)
        model.load_state_dict(previous['model'])
        ema.load_state_dict(previous['ema'])
        objective.load_state_dict(previous['objective'])
        optimizer.load_state_dict(previous['optimizer'])
        disc_optimizer.load_state_dict(previous['disc_optimizer'])
        step, epoch, offset, seen_samples = (previous[key] for key in ('step', 'epoch', 'sample_offset', 'seen_samples'))
        elapsed_before = previous['elapsed_seconds']
        torch.set_rng_state(previous['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(previous['cuda_rng'].cpu(), device)
    else:
        freeze_sources(out)
    write_json(out / 'config.json', config)
    (out / 'train.pid').write_text(str(os.getpid()) + '\n')
    sampler = EpochBatchSampler(len(training), batch_size, seed=seed, epoch=epoch, offset=offset)
    loader = make_loader(training, sampler, config)
    started, started_utc = (time.monotonic(), now())
    stop_requested = False
    old_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def state():
        return dict(step=step, epoch=epoch, sample_offset=offset, seen_samples=seen_samples, model=model.state_dict(), ema=ema.state_dict(), objective=objective.state_dict(), optimizer=optimizer.state_dict(), disc_optimizer=disc_optimizer.state_dict(), config=config, elapsed_seconds=elapsed(), best_weights=None, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)

    def status(stage, **values):
        write_json(out / 'status.json', dict(stage=stage, status=stage, step=step, total_steps=total_steps, epoch=epoch, sample_offset=offset, seen_samples=seen_samples, total_epochs=max_epochs, elapsed_seconds=elapsed(), updated_utc=now(), started_utc=started_utc, pid=os.getpid(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'not restricted'), **values))
    measured_cache, durations = ({}, [])
    mixed = bool(config.get('mixed_precision', True))
    monitor_every = int(config.get('monitor_every', config.get('eval_every', 1000)))
    checkpoint_every = int(config.get('checkpoint_every', 1000))
    if monitor_every < 1 or checkpoint_every < 1:
        raise ValueError('Monitoring/checkpoint intervals must be positive')
    last_monitored = -1

    def monitoring_step():
        nonlocal last_monitored
        status('external_monitoring')
        rows = monitor({'online': model, 'ema': ema}, renderer, monitoring, measured_cache, step=step, epoch=epoch, out=out, mixed_precision=mixed)
        last_monitored = step
        print(json.dumps(dict(external_monitor=rows)), flush=True)
        status('training')
    try:
        status('initializing')
        if not resume and config.get('initial_evaluation', True):
            monitoring_step()
            save_checkpoint(out / 'checkpoints/latest.pt', state())
        model.train()
        while epoch < max_epochs and (not stop_requested) and (elapsed() < max_hours * 3600):
            sampler.set_position(epoch, offset)
            fetch_started = time.monotonic()
            for cpu_batch in loader:
                if stop_requested or elapsed() >= max_hours * 3600:
                    break
                tick = time.monotonic()
                data_wait_seconds = tick - fetch_started
                batch = {key: cpu_batch[key].to(device, non_blocking=True) for key in ('x', 'mask', 'directions', 'bvals', 'query_basis')}
                x, mask = (batch['x'], batch['mask'])
                h, w = renderer.native_shape
                query = torch.cat((batch['directions'], batch['bvals'][..., None]), -1)
                optimizer.zero_grad(set_to_none=True)
                disc_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    target = renderer.forward_from_query_basis(x, mask, batch['query_basis'])
                with autocast_context(device, mixed):
                    reconstructed, posterior = model(x, sample_posterior=True)
                reconstructed = reconstructed.float()
                fake = renderer.forward_from_query_basis(reconstructed, mask, batch['query_basis'])
                loss, logs, valid = generator_objective(objective, reconstructed, x, fake, target, mask, query, posterior, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite generator loss at step {step}')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0), error_if_nonfinite=True)
                optimizer.step()
                disc_logs = {}
                if valid.any() and objective.gan_factor(step) and loss_cfg.adversarial_weight:
                    disc_loss, disc_logs = objective.discriminator_loss(fake[valid].detach(), target[valid], mask[valid, :, :h, :w], query[valid], step)
                    if not torch.isfinite(disc_loss):
                        raise FloatingPointError(f'Nonfinite discriminator loss at step {step}')
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(objective.discriminator.parameters(), config.get('gradient_clip', 1.0), error_if_nonfinite=True)
                    disc_optimizer.step()
                with torch.no_grad():
                    decay = min(float(config.get('ema_decay', 0.999)), (1 + step) / (10 + step))
                    for average, current in zip(ema.parameters(), model.parameters()):
                        average.lerp_(current, 1 - decay)
                    for average, current in zip(ema.buffers(), model.buffers()):
                        average.copy_(current)
                seen_samples += len(x)
                offset += len(x)
                if offset == len(training):
                    epoch, offset = (epoch + 1, 0)
                step += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                compute_seconds = time.monotonic() - tick
                seconds = data_wait_seconds + compute_seconds
                durations.append(seconds)
                row = {key: float(value.detach()) if isinstance(value, torch.Tensor) else value for key, value in {**logs, **disc_logs}.items()}
                row.update(step=step, epoch=epoch, sample_offset=offset, seen_samples=seen_samples, created_utc=now(), seconds=seconds, elapsed_seconds=elapsed(), data_wait_seconds=data_wait_seconds, compute_seconds=compute_seconds, recon_loss=row['reconstruction_loss'], kl_loss=row['kl_per_scalar'], img_loss=row['image_loss'], generator_gan_loss=row['adversarial_generator'], total_loss=float(loss.detach()), gradient_norm=float(gradient), equivalent_data_passes=seen_samples / len(training), batch_size=len(x), queries_per_sample=queries, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0)
                append_json(out / 'metrics.jsonl', row)
                if step == 1 or step % 10 == 0:
                    status('training', latest=row, mean_step_seconds=float(np.mean(durations[-100:])))
                    print(json.dumps(row), flush=True)
                if step % monitor_every == 0:
                    monitoring_step()
                if step % checkpoint_every == 0:
                    save_checkpoint(out / 'checkpoints/latest.pt', state())
                fetch_started = time.monotonic()
        if not stop_requested and last_monitored != step:
            monitoring_step()
        save_checkpoint(out / 'checkpoints/latest.pt', state())
        stage = 'stopped' if stop_requested else 'complete' if epoch >= max_epochs else 'time_budget_reached'
        if stage in ('complete', 'time_budget_reached'):
            save_checkpoint(out / 'checkpoints/final.pt', state())
        status(stage)
        if stage in ('complete', 'time_budget_reached') and config.get('export_latents', False):
            status('exporting_latents', training_status=stage)
            exported = export_corpus_latents(out, cache_root, device=device, weights=config.get('export_weights', 'ema'), split=config.get('export_split', 'train'), batch_size=int(config.get('export_batch_size', config.get('batch_size', 64))), workers=int(config.get('export_workers', config.get('workers', 8))), stop_requested=lambda: stop_requested)
            status('stopped' if stop_requested else stage, latent_export_complete=exported['complete'])
    except Exception as error:
        save_checkpoint(out / 'checkpoints/interrupted.pt', state())
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if getattr(loader, '_iterator', None) is not None:
            loader._iterator._shutdown_workers()

@torch.no_grad()
def export_corpus_latents(out, cache_root, *, device=torch.device('cpu'), weights='ema', split='train', batch_size=64, workers=8, stop_requested=None):
    """Stream posterior means; bind every file to an immutable decoder snapshot."""
    if weights not in ('online', 'ema'):
        raise ValueError('Export weights must be online or ema')
    out, cache_root = (Path(out).resolve(), Path(cache_root).resolve())
    state = torch.load(out / 'checkpoints/latest.pt', map_location=device, weights_only=False)
    config = state['config']
    if sha256_file(cache_root / 'manifest.json') != config['signature']['cache_manifest_sha256']:
        raise ValueError('Export corpus differs from the trained checkpoint')
    dataset = CorpusGaussianDataset(cache_root, split, queries_per_sample=3)
    model = GaussianAutoencoderKL(config['model']).to(device).eval()
    model.load_state_dict(state['ema' if weights == 'ema' else 'model'])
    decoder = out / 'checkpoints' / f"export_step{state['step']:09d}.pt"
    if not decoder.exists():
        save_checkpoint(decoder, {key: state[key] for key in ('model', 'ema', 'config', 'step', 'best_weights')})
    folder = out / 'latents'
    folder.mkdir(exist_ok=True)
    loader_kwargs = dict(dataset=dataset, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(0))
    if workers:
        loader_kwargs['multiprocessing_context'] = 'spawn'
    loader = DataLoader(**loader_kwargs)
    records = []
    for batch in loader:
        if stop_requested is not None and stop_requested():
            break
        posterior = model.encode(batch['x'].to(device, non_blocking=True))
        latent = posterior.mode().permute(0, 2, 3, 1).cpu().numpy()
        for index in range(len(latent)):
            destination = folder / batch['subject'][index] / (batch['id'][index] + '.npz')
            destination.parent.mkdir(exist_ok=True)
            metadata = dict(id=batch['id'][index], basis_content_id=dataset.basis.content_id, decoder=os.path.relpath(decoder, destination.parent), weights=weights, step=state['step'], normalization=os.path.relpath(cache_root / 'normalization.npz', destination.parent), basis=os.path.relpath(cache_root / 'basis.npz', destination.parent), native_shape=dataset.native_shape, posterior='mode', layout='HWC')
            save_preview(destination, dict(latent=latent[index], mask=batch['mask'][index, 0].numpy(), metadata=np.array(json.dumps(metadata))))
            records.append(dict(id=batch['id'][index], path=str(destination), shape=list(latent[index].shape)))
        write_json(out / 'latent_export_status.json', dict(completed=len(records), total=len(dataset), step=state['step'], weights=weights, updated_utc=now()))
    complete = len(records) == len(dataset)
    write_json(folder / 'manifest.json', dict(records=records, checkpoint_step=state['step'], weights=weights, complete=complete, total=len(dataset), latent_layout='HWC', split=split, stored_latent_scalars=int(np.prod(model.config.latent_shape)), excludes_from_scalar_count='support mask, shared decoder, B, normalization and metadata', created_utc=now()))
    result = dict(completed=len(records), total=len(dataset), complete=complete, step=state['step'], weights=weights)
    write_json(out / 'latent_export_status.json', dict(result, updated_utc=now()))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-latents', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.export_latents:
        export_corpus_latents(args.output, args.cache, device=torch.device(config.get('device', 'cuda:0')), weights=config.get('export_weights', 'ema'), split=config.get('export_split', 'train'), batch_size=config.get('batch_size', 64), workers=config.get('workers', 8))
    else:
        train_corpus(config, args.output, args.cache, resume=args.resume)
if __name__ == '__main__':
    main()
