"""CPU-only, resumable cache preparation with explicit source identities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from gaussian_kl_vae.corpus_data import validate_corpus
from gaussian_kl_vae.data import sha256_file
from .data import FORMAT, preprocess_conditions, read_raw_conditions, source_paths

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_VAE = PROJECT / "outputs/gaussian_kl_vae_r22_full_20261028"
DEFAULT_CACHE = PROJECT / "outputs/gaussian_latent_fm_r22_20261030/cache"
DEFAULT_OLD = None


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def fingerprint(path):
    path = Path(path)
    st = path.stat()
    return dict(path=str(path.resolve()), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def _old_conditions(job, old_root):
    """Authenticate only the source-condition slab; old Gaussian labels ignored."""
    if job["split"] != "train" or old_root is None:
        return None, None
    root = Path(old_root)
    marker_path = root / "subjects" / f"{job['subject']}.json"
    if not marker_path.is_file():
        return None, None
    marker = json.loads(marker_path.read_text())
    old_records = {f"{r['subject']}_{r['slice']}": r for r in marker["records"]}
    if any(r["id"] not in old_records for r in job["records"]):
        return None, None
    # Fail closed when any source has changed: read the current MRI instead.
    for record in job["records"]:
        for name, path in source_paths(record).items():
            if fingerprint(path) != old_records[record["id"]]["sources"][name]:
                return None, None
    array = np.load(root / "conditions.npy", mmap_mode="r")
    slab = np.asarray(array[np.asarray(marker["indices"], dtype=np.int64)])
    if hashlib.sha256(slab.tobytes()).hexdigest() != marker["cache_sha256"]["conditions"]:
        raise ValueError(f"Historical condition slab checksum differs: {job['subject']}")
    positions = {index: i for i, index in enumerate(marker["indices"])}
    chosen = np.stack([slab[positions[old_records[r["id"]]["index"]]] for r in job["records"]])
    # Independent real-NIfTI parity on one complete slice per subject. Cached
    # source-only preprocessing and native orientation must be exactly equal.
    middle = len(job["records"]) // 2
    fresh = preprocess_conditions(read_raw_conditions(job["records"][middle]))
    if not np.array_equal(fresh, chosen[middle]):
        raise ValueError(f"Cached condition preprocessing/orientation changed: {job['subject']}")
    return chosen, dict(source=str(root), complete_sha256=sha256_file(marker_path),
                        slab_sha256=marker["cache_sha256"]["conditions"],
                        checked_source_fields=["t1", "t2", "b0", "mask"],
                        independent_nifti_parity=job["records"][middle]["id"],
                        legacy_gaussians_read=False)


def _prepare_subject(job, root, old_root, expected):
    started = time.monotonic()
    root = Path(root)
    folder = root / job["cache_shard"]
    folder.mkdir(parents=True, exist_ok=True)
    marker_path = folder / "complete.json"
    signature = dict(ids=[r["id"] for r in job["records"]], source_signature=expected)
    if marker_path.exists():
        completed = json.loads(marker_path.read_text())
        if completed["signature"] != signature:
            raise ValueError(f"Existing subject cache uses different sources: {folder}")
        for name, digest in completed["files_sha256"].items():
            if sha256_file(folder / name) != digest:
                raise ValueError(f"Existing subject cache corrupted: {folder / name}")
        for source in completed["source_records"]:
            for item in [source["latent"], *source["conditions"].values()]:
                current = fingerprint(item["path"])
                if any(current[key] != item[key] for key in current):
                    raise ValueError(f"Cached input source changed: {item['path']}")
        return dict(subject=job["subject"], split=job["split"], count=len(job["records"]),
                    cache_shard=job["cache_shard"], complete_sha256=sha256_file(marker_path),
                    seconds=time.monotonic()-started, resumed=True)
    count = len(job["records"])
    latents = np.lib.format.open_memmap(folder / "latents.npy", mode="w+", dtype=np.float32,
                                       shape=(count, 8, 40, 44))
    conditions = np.lib.format.open_memmap(folder / "conditions.npy", mode="w+", dtype=np.float32,
                                          shape=(count, 3, 145, 174))
    reused, provenance = _old_conditions(job, old_root)
    corpus_masks = np.load(Path(expected["corpus_root"]) / job["records"][0]["shard"] / "masks.npy",
                           mmap_mode="r")
    source_records = []
    for index, record in enumerate(job["records"]):
        path = Path(record["latent_path"])
        payload = path.read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            value = archive["latent"]
            if (meta["id"] != record["id"] or meta["step"] != expected["checkpoint_step"]
                    or meta["weights"] != "ema" or meta["posterior"] != "mode" or meta["layout"] != "HWC"
                    or meta["basis_content_id"] != expected["basis_content_id"]
                    or str((path.parent / meta["decoder"]).resolve()) != expected["decoder_checkpoint"]
                    or str((path.parent / meta["normalization"]).resolve()) != expected["gaussian_normalization"]
                    or value.shape != (40, 44, 8) or value.dtype != np.float32
                    or not np.isfinite(value).all()):
                raise ValueError(f"VAE latent identity/shape/finiteness mismatch: {path}")
            mask = archive["mask"]
            native = corpus_masks[int(record["index"]), 0]
            if (mask.shape != (160, 176) or mask.dtype != np.bool_
                    or not np.array_equal(mask[:145, :174], native)
                    or mask[145:].any() or mask[:, 174:].any()):
                raise ValueError(f"VAE latent/Gaussian target masks differ: {path}")
            latents[index] = value.transpose(2, 0, 1)
        conditions[index] = reused[index] if reused is not None else preprocess_conditions(read_raw_conditions(record))
        source_records.append(dict(id=record["id"], latent=dict(fingerprint(path), sha256=hashlib.sha256(payload).hexdigest()),
                                   conditions={name: fingerprint(p) for name, p in source_paths(record).items()}))
    if not np.isfinite(conditions).all():
        raise ValueError(f"Nonfinite condition cache: {job['subject']}")
    latents.flush()
    conditions.flush()
    completed = dict(signature=signature, subject=job["subject"], split=job["split"], count=count,
                     source_records=source_records, old_condition_reuse=provenance,
                     files_sha256={name: sha256_file(folder / name) for name in ("latents.npy", "conditions.npy")},
                     shapes={"latents": list(latents.shape), "conditions": list(conditions.shape)},
                     completed_utc=now())
    write_json(marker_path, completed)
    return dict(subject=job["subject"], split=job["split"], count=count, cache_shard=job["cache_shard"],
                complete_sha256=sha256_file(marker_path), seconds=time.monotonic()-started, reused_conditions=reused is not None)


def fit_latent_statistics(root, subjects):
    """FP64 parallel-moment merge over every latent location of training only."""
    total = 0
    mean, m2 = np.zeros(8, np.float64), np.zeros(8, np.float64)
    training_subjects, training_slices = [], 0
    for subject in sorted(subjects, key=lambda item: item["subject"]):
        if subject["split"] != "train":
            continue
        values = np.load(Path(root) / subject["cache_shard"] / "latents.npy", mmap_mode="r")
        # Accumulate bounded chunks so no full-corpus array is materialized.
        for start in range(0, len(values), 16):
            array = np.asarray(values[start:start+16], dtype=np.float64).transpose(0, 2, 3, 1).reshape(-1, 8)
            count, batch_mean = len(array), array.mean(0)
            batch_m2 = ((array-batch_mean)**2).sum(0)
            delta = batch_mean-mean
            m2 += batch_m2 + delta**2 * total * count / (total+count)
            mean += delta * count / (total+count)
            total += count
        training_subjects.append(subject["subject"])
        training_slices += len(values)
    if not total:
        raise ValueError("Cannot fit latent statistics without training samples")
    std = np.sqrt(m2 / total)
    floor = max(float(std.max())*1e-8, 1e-8)
    metadata = dict(training_subjects=training_subjects, training_slices=training_slices,
                    latent_positions=total, external_subjects_used=False, population_variance=True,
                    weighting="all training posterior-mean latent locations, including decoder-padding context",
                    std_floor=floor, clipping=False)
    np.savez(Path(root) / "latent_normalization.npz", mean=mean.astype(np.float32),
             std=np.maximum(std, floor).astype(np.float32), metadata=np.array(json.dumps(metadata)))
    write_json(Path(root) / "latent_statistics.json", dict(mean=mean.tolist(), std=std.tolist(), **metadata))


def prepare(cache=DEFAULT_CACHE, vae=DEFAULT_VAE, *, old_condition_cache=DEFAULT_OLD, workers=16):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root, vae = Path(cache).resolve(), Path(vae).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / ".prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    corpus = validate_corpus(vae / "cache")
    export = json.loads((vae / "latents/manifest.json").read_text())
    if not export["complete"] or export["weights"] != "ema" or export["total"] != len(corpus["records"]):
        raise ValueError("Require complete all-corpus final EMA posterior export")
    checkpoint = vae / "checkpoints" / f"export_step{export['checkpoint_step']:09d}.pt"
    source = dict(corpus_manifest=str(vae / "cache/manifest.json"),
                  corpus_manifest_sha256=sha256_file(vae / "cache/manifest.json"),
                  corpus_ready_sha256=sha256_file(vae / "cache/ready.json"),
                  export_manifest=str(vae / "latents/manifest.json"),
                  export_manifest_sha256=sha256_file(vae / "latents/manifest.json"),
                  decoder_checkpoint=str(checkpoint), decoder_sha256=sha256_file(checkpoint),
                  decoder_bytes=checkpoint.stat().st_size, decoder_mtime_ns=checkpoint.stat().st_mtime_ns,
                  checkpoint_step=int(export["checkpoint_step"]), weights="ema")
    export_lookup = {r["id"]: r for r in export["records"]}
    if (len(export_lookup) != len(export["records"])
            or set(export_lookup) != {r["id"] for r in corpus["records"]}):
        raise ValueError("VAE latent export does not cover every Gaussian slice exactly once")
    monitor = json.loads((vae / "monitor_manifest.json").read_text())
    monitor_ids = set(monitor["records"])
    records = [dict(r, latent_path=export_lookup[r["id"]]["path"])
               for r in corpus["records"] if r["split"] == "train" or r["id"] in monitor_ids]
    jobs = {}
    for record in records:
        shard = f"shards/{record['split']}/{record['subject']}"
        job = jobs.setdefault(record["subject"], dict(subject=record["subject"], split=record["split"],
                                                      cache_shard=shard, records=[]))
        record.update(cache_shard=shard, cache_index=len(job["records"]))
        job["records"].append(record)
    expected = dict(corpus_root=str(vae / "cache"), decoder_checkpoint=str(checkpoint),
                    decoder_sha256=source["decoder_sha256"], checkpoint_step=export["checkpoint_step"],
                    gaussian_normalization=str(vae / "cache/normalization.npz"),
                    basis_content_id=corpus["basis_content_id"], corpus_manifest_sha256=source["corpus_manifest_sha256"])
    plan = dict(format=FORMAT, source=source, record_ids=[r["id"] for r in records],
                condition_order=["t1", "t2", "b0"],
                condition_preprocessing={"t1": "positive-source p1/p99 clipped [0,1]", "t2": "positive-source p1/p99 clipped [0,1]",
                                         "b0": "raw / fixed 1000; no clipping"},
                monitor_protocol="frozen VAE external 20 representative slices; monitoring only, no selection")
    if (root / "plan.json").exists() and json.loads((root / "plan.json").read_text()) != plan:
        raise ValueError("Existing latent cache plan differs; choose another cache directory")
    write_json(root / "plan.json", plan)
    completed = []
    def progress(status, **extra):
        state = dict(status=status, completed_subjects=len(completed), total_subjects=len(jobs),
                     completed_slices=sum(s["count"] for s in completed), total_slices=len(records),
                     elapsed_seconds=time.monotonic()-started, workers=workers, pid=os.getpid(), updated_utc=now(), **extra)
        write_json(root / "status.json", state)
        print(json.dumps(state), flush=True)
    progress("preparing")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_prepare_subject, job, str(root), str(old_condition_cache) if old_condition_cache else None, expected)
                       for job in jobs.values()]
            for future in as_completed(futures):
                completed.append(future.result())
                if len(completed) % 10 == 0 or len(completed) == len(jobs):
                    progress("preparing", latest_subject=completed[-1]["subject"])
        progress("fitting_training_latent_statistics")
        fit_latent_statistics(root, completed)
        manifest = dict(plan, records=records, subjects=sorted(completed, key=lambda s: s["subject"]),
                        basis_content_id=corpus["basis_content_id"], native_shape=[145,174], padded_shape=[160,176],
                        latent_shape=[8,40,44], gaussian_shape=[22,160,176],
                        counts=dict(train_subjects=sum(s["split"] == "train" for s in completed),
                                    train_slices=sum(s["count"] for s in completed if s["split"] == "train"),
                                    monitor_subjects=sum(s["split"] == "external" for s in completed),
                                    monitor_slices=sum(s["count"] for s in completed if s["split"] == "external")),
                        files_sha256={"latent_normalization.npz": sha256_file(root / "latent_normalization.npz")},
                        source_code_sha256={str(Path(__file__).resolve()): sha256_file(__file__),
                                            str(Path(__file__).with_name("data.py")): sha256_file(Path(__file__).with_name("data.py"))},
                        created_utc=now())
        write_json(root / "manifest.json", manifest)
        write_json(root / "ready.json", dict(status="ready", manifest_sha256=sha256_file(root / "manifest.json"),
                                              training_only_latent_statistics=True, source_conditions_verified=True,
                                              completed_utc=now()))
        progress("complete", counts=manifest["counts"])
        return manifest
    except Exception as exc:
        progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--old-condition-cache", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    prepare(args.cache, args.vae, old_condition_cache=args.old_condition_cache, workers=args.workers)


if __name__ == "__main__":
    main()
