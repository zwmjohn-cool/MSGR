"""Fit every raw DWI slice to fixed native Gaussians and one frozen rank-22 B.

The full corpus is resumable at slice boundaries. Dense subject shards serve
VAE loading; each source DWI also receives a standalone active-coefficient NPZ.
Neither source intensities nor the frozen public basis are changed. Empty brain
and one-pixel brain slices are retained. Completion is gated on all labels,
source exports, train-only statistics and checksummed shard files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
from time import perf_counter
import traceback

import nibabel as nib
import numpy as np
import torch

from spatial_gaussian_dwi.radial_sh import cardinal_basis
from spatial_gaussian_dwi.shared_fitting import prepare_teacher
from spatial_gaussian_dwi.shared_model import SharedBasis, SharedGaussianDWI, fit_coefficients

from .data import ChannelNormalization, sha256_file


FORMAT = "gaussian_kl_vae.fixed_shared_corpus.v1"
SUBJECT_FORMAT = "gaussian_kl_vae.fixed_shared_subject.v1"
CONFIG = dict(width=2., sigma=.7, smooth=.0005, mp_blend=.29, patch_size=9,
              patch_offsets=4, noise_factor=1., denoise_backend="cpu",
              cg_tolerance=1e-7, cg_iterations=1000, ridge=0., require_convergence=True)
ARRAY_FILES = ("coeff.npy", "masks.npy", "directions.npy", "bvals.npy", "query_basis.npy")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _identity(manifest):
    immutable = {key: value for key, value in manifest.items()
                 if key not in {"manifest_content_id", "files_sha256", "finalized_utc"}}
    return hashlib.sha256(_canonical(immutable).encode()).hexdigest()


def load_manifest(out):
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("manifest_content_id") != _identity(manifest):
        raise ValueError("Corpus manifest identity mismatch")
    basis = SharedBasis.load(Path(out) / "basis.npz")
    if basis.content_id != manifest["basis_content_id"]:
        raise ValueError("Corpus public basis identity mismatch")
    return manifest


class CorpusGaussianDWI(SharedGaussianDWI):
    """Same fixed Gaussian field, adding explicitly supported zero-center labels.

    File format differs from the historical experiment so empty-label support
    does not silently change old loader contracts. Parameters are solely active
    [M,R] coefficients; format metadata is outside the parameter array.
    """

    def __init__(self, latent, mask, basis, metadata=None):
        mask = np.asarray(mask)
        if mask.ndim != 2 or min(mask.shape) < 1 or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Require a binary native 2D primitive mask")
        if mask.any():
            super().__init__(latent, mask, basis, metadata)
            return
        if not isinstance(basis, SharedBasis):
            raise TypeError("Require an explicit shared public basis")
        self.latent = torch.as_tensor(latent)
        if self.latent.dtype not in {torch.float32, torch.float64} or self.latent.shape != (*mask.shape, basis.rank):
            raise ValueError("Empty label must have floating native H,W,R shape")
        self.mask = mask.astype(bool, copy=True)
        self.basis = basis.to(device=self.latent.device, dtype=self.latent.dtype)
        self.config = self.basis.config
        self.metadata = dict(metadata or {})

    def evaluate_points(self, xy, directions, bvals, *, chunk_size=128):
        if self.mask.any():
            return super().evaluate_points(xy, directions, bvals, chunk_size=chunk_size)
        xy = torch.as_tensor(xy, device=self.latent.device, dtype=self.latent.dtype)
        if xy.ndim != 2 or xy.shape[-1] != 2 or not torch.isfinite(xy).all():
            raise ValueError("Points must be finite [P,2]")
        query = self._query(directions, bvals)
        # Keep a well-defined zero derivative with respect to the parameter map.
        return self._amplitudes().sum() * 0. + query.new_zeros((len(xy), len(query)))

    def save(self, path, *, basis_path=None):
        path = Path(path).resolve()
        public_path = self.basis.path if basis_path is None else Path(basis_path).resolve()
        if public_path is None:
            raise ValueError("Save the public basis before exporting a subject label")
        public = SharedBasis.load(public_path)
        if public.content_id != self.basis.content_id:
            raise ValueError("Subject/public basis content mismatch")
        header = dict(format=SUBJECT_FORMAT, rank=public.rank, basis_content_id=public.content_id,
                      basis_path=os.path.relpath(public_path, path.parent), metadata=self.metadata,
                      signal_units="restored raw MRI arbitrary units", b0_division=False,
                      multiply_b0=False, gaussian_sigma=public.config["sigma"],
                      spatial_convention="native row,column; fixed centers; full Gaussian tails",
                      active_centers=int(self.mask.sum()))
        _write_npz(path, coefficients=self.latent.detach().cpu().numpy()[self.mask].astype(np.float32),
                   mask_packbits=np.packbits(self.mask.ravel()),
                   spatial_shape=np.asarray(self.mask.shape, dtype=np.int32), metadata=np.array(_canonical(header)))
        return path

    @classmethod
    def load(cls, path, *, basis=None, device="cpu", dtype=torch.float32):
        path = Path(path).resolve()
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(archive["metadata"].item())
            if header.get("format") != SUBJECT_FORMAT:
                raise ValueError("Not a new full-corpus fixed shared-R22 label")
            shape = archive["spatial_shape"]
            if shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape <= 0):
                raise ValueError("Invalid subject spatial shape")
            shape = tuple(map(int, shape))
            packed = archive["mask_packbits"]
            if packed.dtype != np.uint8 or packed.shape != ((np.prod(shape) + 7) // 8,):
                raise ValueError("Invalid packed primitive mask")
            mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
            active = archive["coefficients"].copy()
        if active.dtype != np.float32 or active.shape != (int(mask.sum()), header["rank"]):
            raise ValueError("Active coefficients disagree with support/rank")
        if not np.isfinite(active).all() or int(mask.sum()) != header["active_centers"]:
            raise ValueError("Corrupt active coefficients or mask")
        if basis is None:
            reference = Path(header["basis_path"])
            if reference.is_absolute():
                raise ValueError("Public basis reference must be relative")
            basis = SharedBasis.load(path.parent / reference, device=device, dtype=dtype)
        if basis.content_id != header["basis_content_id"] or basis.rank != header["rank"]:
            raise ValueError("Subject/public basis content mismatch")
        latent = np.zeros((*shape, basis.rank), np.float32)
        latent[mask] = active
        return cls(torch.as_tensor(latent, device=device, dtype=dtype), mask, basis, header.get("metadata"))


def load_model(path, *, basis=None, device="cpu", dtype=torch.float32):
    return CorpusGaussianDWI.load(path, basis=basis, device=device, dtype=dtype)


def _native_dwi_shape(shape):
    if len(shape) == 4 and shape[2] == 1:
        return (shape[0], shape[1], shape[3])
    if len(shape) == 3:
        return tuple(shape)
    raise ValueError(f"Unsupported DWI slice shape: {shape}")


def fast_nifti_shape(path):
    """Read only a NIfTI-1/2 header, without constructing a data proxy/index.

    ``nib.load`` may build an indexed-gzip index when probing the compressed
    file. With thousands of 20--30 MB slices that turns an inventory scan into
    a corpus-wide image read. A standard gzip stream needs only its first
    compressed block to obtain the fixed 348/540-byte NIfTI header.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        prefix = stream.read(4)
        if len(prefix) != 4:
            raise ValueError(f"Truncated NIfTI header: {path}")
        sizes = (struct.unpack("<I", prefix)[0], struct.unpack(">I", prefix)[0])
        size = next((value for value in sizes if value in (348, 540)), None)
        if size is None:
            raise ValueError(f"Not a NIfTI-1/2 header: {path}")
        payload = prefix + stream.read(size - 4)
    if len(payload) != size:
        raise ValueError(f"Truncated NIfTI-{1 if size == 348 else 2} header: {path}")
    header_type = nib.Nifti1Header if size == 348 else nib.Nifti2Header
    header = header_type.from_fileobj(io.BytesIO(payload))
    shape = tuple(map(int, header.get_data_shape()))
    if not shape or min(shape) < 1:
        raise ValueError(f"Invalid NIfTI dimensions: {path}")
    return shape


def _acquisition(folder):
    b = np.asarray(np.loadtxt(folder / "bvals"), np.float32).reshape(-1)
    g = np.asarray(np.loadtxt(folder / "bvecs"), np.float32)
    if g.shape == (3, len(b)):
        g = g.T
    if (g.shape != (len(b), 3) or not np.isfinite(b).all() or np.any(b <= 0)
            or not np.isfinite(g).all() or np.any(np.linalg.norm(g, axis=1) < 1e-8)):
        raise ValueError(f"Require valid positive-b directions: {folder}")
    return g.copy(), b


def _stat(path):
    info = Path(path).stat()
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns)


def _scan_subject(item, native_shape):
    split, folder = item
    subject = folder.name.removesuffix("_processed")
    pattern = re.compile(re.escape(subject) + r"_processed_(\d+)_dwi\.nii\.gz$")
    sources = []
    for path in folder.glob("*_dwi.nii.gz"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized DWI filename; refusing to omit it: {path}")
        sources.append((int(match[1]), path))
    sources.sort()
    if not sources:
        raise ValueError(f"Subject has no DWI slices: {folder}")
    g, b = _acquisition(folder)
    records, shape = [], None
    shard = f"shards/{split}/{subject}"
    for index, (slice_index, path) in enumerate(sources):
        image_shape = _native_dwi_shape(fast_nifti_shape(path))
        if image_shape[-1] != len(b):
            raise ValueError(f"Acquisition/DWI query mismatch: {path}")
        if native_shape is not None and tuple(image_shape[:2]) != tuple(native_shape):
            raise ValueError(f"Unexpected native spatial shape {image_shape[:2]}: {path}")
        if shape is not None and shape != image_shape[:2]:
            raise ValueError(f"Different native shapes within subject: {folder}")
        shape = image_shape[:2]
        mask_path = path.with_name(path.name.replace("_dwi.nii.gz", "_mask.nii.gz"))
        mask_raw = np.squeeze(nib.load(mask_path).get_fdata(dtype=np.float32))
        if mask_raw.shape != shape or not np.isfinite(mask_raw).all():
            raise ValueError(f"Invalid brain mask: {mask_path}")
        mask = mask_raw > 0
        records.append(dict(id=f"{subject}_{slice_index}", subject=subject, slice=slice_index,
                            index=index, split=split, shard=shard, native_shape=list(shape),
                            query_count=len(b), mask_voxels=int(mask.sum()),
                            dwi_path=str(path.resolve()), mask_path=str(mask_path.resolve()),
                            output_npz=str(path.resolve()).removesuffix(".nii.gz") + ".npz",
                            source_stats=dict(dwi=_stat(path), mask=_stat(mask_path))))
    return dict(subject=subject, split=split, shard=shard, folder=str(folder.resolve()),
                slice_count=len(records), query_count=len(b), native_shape=list(shape),
                bvals_path=str((folder / "bvals").resolve()), bvecs_path=str((folder / "bvecs").resolve()),
                bvals_sha256=sha256_file(folder / "bvals"), bvecs_sha256=sha256_file(folder / "bvecs"),
                records=records)


def build_manifest(out, trainroot, evalroot, basispath, *, native_shape=(145, 174),
                   padded_shape=(160, 176), scan_workers=16, config=None):
    out, trainroot, evalroot = map(lambda p: Path(p).resolve(), (out, trainroot, evalroot))
    if (out / "manifest.json").exists():
        manifest = load_manifest(out)
        if manifest["train_root"] != str(trainroot) or manifest["external_root"] != str(evalroot):
            raise ValueError("Cannot reuse corpus manifest for different roots")
        if SharedBasis.load(basispath).content_id != manifest["basis_content_id"]:
            raise ValueError("Cannot replace frozen corpus basis")
        if config is not None and dict(CONFIG, **config) != manifest["fit_config"]:
            raise ValueError("Cannot replace frozen fitting configuration")
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    migrations = {}
    for split, root in (("train", trainroot), ("external", evalroot)):
        path = root / "RAW_INTENSITY_MIGRATION.json"
        value = json.loads(path.read_text())
        if value.get("status") != "complete" or value.get("target_signal_units") != "raw MRI arbitrary units":
            raise ValueError(f"Raw-intensity restoration is not verified complete: {path}")
        migrations[split] = dict(path=str(path), sha256=sha256_file(path), metadata=value)
    items = [(split, folder) for split, root in (("train", trainroot), ("external", evalroot))
             for folder in sorted(root.glob("*_processed")) if folder.is_dir()]
    if not items or not any(split == "train" for split, _ in items):
        raise ValueError("No training subjects found")
    ids = [folder.name.removesuffix("_processed") for _, folder in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Train and external subject identities overlap")
    with ThreadPoolExecutor(max_workers=int(scan_workers)) as executor:
        subjects = list(executor.map(lambda item: _scan_subject(item, native_shape), items))
    basis = SharedBasis.load(basispath)
    if basis.rank != 22 or basis.config != {"sigma": .7, "width": 2.}:
        raise ValueError("Full corpus requires the frozen native sigma=.7 shared-R22 basis")
    basis_ids = basis.metadata.get("basis_training_ids", [])
    training_subjects = {subject["subject"] for subject in subjects if subject["split"] == "train"}
    if not basis_ids or any(str(value).rsplit("_", 1)[0] not in training_subjects for value in basis_ids):
        raise ValueError("Public basis training provenance must be a subset of training subjects")
    basis.save(out / "basis.npz")
    cfg = dict(CONFIG, **(config or {}))
    if cfg["sigma"] != .7 or cfg["width"] != 2.:
        raise ValueError("Fitting geometry/radial width disagrees with frozen basis")
    records = [record for subject in subjects for record in subject["records"]]
    manifest = dict(format=FORMAT, created_utc=_now(), train_root=str(trainroot), external_root=str(evalroot),
                    rank=22, native_shape=list(native_shape) if native_shape is not None else None,
                    padded_shape=list(padded_shape), basis_content_id=basis.content_id,
                    basis_source=str(Path(basispath).resolve()), basis_source_sha256=sha256_file(basispath),
                    basis_training_ids=basis_ids, frozen_basis=True, fit_config=cfg,
                    migrations=migrations, subjects=subjects, records=records,
                    counts={split: dict(subjects=sum(s["split"] == split for s in subjects),
                                        slices=sum(r["split"] == split for r in records),
                                        empty_slices=sum(r["split"] == split and r["mask_voxels"] == 0 for r in records))
                            for split in ("train", "external")},
                    signal_units="restored raw MRI arbitrary units; no B0 division/multiplication",
                    parameter_layout="native [H,W,22]; coefficients.npy uses [slice,22,H,W]; fixed sigma=.7 and centers",
                    export_format=SUBJECT_FORMAT, export_policy="one new fixed-shared NPZ beside every source DWI; suffix only differs",
                    mask_policy="known native primitive support, explicitly retained even when empty; output Gaussian tails unmasked",
                    fit_protocol="all positive-b measurements per slice; this is compression, not held-direction prediction",
                    source_sha256={str(Path(__file__).resolve()): sha256_file(__file__)})
    manifest["manifest_content_id"] = _identity(manifest)
    manifest["files_sha256"] = {"basis.npz": sha256_file(out / "basis.npz")}
    _write(out / "manifest.json", manifest)
    return manifest


def _check_record_sources(record):
    for kind in ("dwi", "mask"):
        if _stat(record[f"{kind}_path"]) != record["source_stats"][kind]:
            raise ValueError(f"Source {kind} changed after manifest freeze: {record['id']}")


def _load_record(record):
    _check_record_sources(record)
    data = nib.load(record["dwi_path"]).get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[2] == 1:
        data = data[:, :, 0, :]
    mask = np.squeeze(nib.load(record["mask_path"]).get_fdata(dtype=np.float32)) > 0
    if data.shape != (*record["native_shape"], record["query_count"]) or mask.shape != tuple(record["native_shape"]):
        raise ValueError("Slice shape changed after manifest freeze")
    if int(mask.sum()) != record["mask_voxels"] or not np.isfinite(data[mask]).all():
        raise ValueError("Invalid or changed active brain signals/mask")
    return data, mask


def fit_record(record, directions, bvals, basis, config=None, *, device="cpu"):
    """Fit one complete raw-intensity slice, including empty and zero-signal cases."""
    start = perf_counter()
    data, mask = _load_record(record)
    cfg = dict(CONFIG, **(config or {}))
    if not mask.any() or not np.any(data[mask]):
        model = CorpusGaussianDWI(torch.zeros((*mask.shape, basis.rank)), mask, basis)
        details = dict(empty_brain=not mask.any(), zero_active_signal=True, teacher_seconds=0.,
                       fit_seconds=0., active_centers=int(mask.sum()), fitted_signal_square_sum=0.,
                       fitted_signal_values=int(mask.sum()) * len(bvals))
    else:
        teacher, _, teacher_details = prepare_teacher(data, directions, bvals, mask, np.arange(len(bvals)), cfg)
        source, fit = fit_coefficients(teacher, directions, bvals, mask, basis, cfg, device=device)
        model = CorpusGaussianDWI(source.latent, source.mask, source.basis)
        with torch.no_grad():
            field = model.latent_field()[torch.as_tensor(mask, device=model.latent.device)].double()
            query = model._query(directions, bvals).double()
            power_sum = float(((field @ (query.T @ query)) * field).sum().clamp_min(0))
        details = dict(empty_brain=False, zero_active_signal=False,
                       teacher_seconds=teacher_details["total_seconds"], fit_seconds=fit["fit_seconds"],
                       cg_iterations=fit["spatial"]["iterations"],
                       cg_relative_residual=fit["spatial"]["relative_normal_residual"],
                       active_centers=int(mask.sum()), query_condition_number=fit["query_condition_number"],
                       fitted_signal_square_sum=power_sum, fitted_signal_values=int(mask.sum()) * len(bvals))
    details.update(total_seconds=perf_counter() - start, queries=len(bvals), public_basis_updated=False)
    model.metadata = dict(id=record["id"], subject=record["subject"], slice=record["slice"],
                          split=record["split"], fit=details, config=cfg, all_queries=True)
    return model, details


def _open_arrays(folder, subject, basis):
    shape = (subject["slice_count"], basis.rank, *subject["native_shape"])
    specifications = {"coeff.npy": (np.float32, shape),
                      "masks.npy": (np.bool_, (shape[0], 1, *shape[-2:]))}
    arrays = {}
    for name, (dtype, expected) in specifications.items():
        path = folder / name
        if path.exists():
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
            if array.shape != expected or array.dtype != dtype:
                raise ValueError(f"Existing subject shard has wrong shape/dtype: {path}")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=expected)
        arrays[name] = array
    return arrays


def _array_slice_hash(coefficients, mask):
    digest = hashlib.sha256(np.asarray(coefficients, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def fit_subject(out, subject, manifest, basis, *, device="cpu", write_source_npz=True, stop_after=None):
    """One subject lock, one atomically acknowledged slice at a time."""
    out = Path(out)
    folder = out / subject["shard"]
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / ".fit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        progress_path = folder / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(
            manifest_content_id=manifest["manifest_content_id"], basis_content_id=basis.content_id,
            subject=subject["subject"], split=subject["split"], slices={}, errors={})
        if progress["manifest_content_id"] != manifest["manifest_content_id"] or progress["basis_content_id"] != basis.content_id:
            raise ValueError("Subject progress belongs to a different corpus/configuration")
        if sha256_file(subject["bvals_path"]) != subject["bvals_sha256"] or sha256_file(subject["bvecs_path"]) != subject["bvecs_sha256"]:
            raise ValueError("Acquisition table changed after manifest freeze")
        g, b = _acquisition(Path(subject["folder"]))
        public = basis.to(device="cpu", dtype=torch.float32)
        query = cardinal_basis(torch.from_numpy(g), b, public.config["width"]) @ public.matrix
        for name, values in (("directions.npy", g), ("bvals.npy", b), ("query_basis.npy", query.numpy())):
            path = folder / name
            if path.exists():
                if not np.array_equal(np.load(path, allow_pickle=False), values):
                    raise ValueError(f"Existing acquisition cache differs: {path}")
            else:
                np.save(path, values)
        arrays = _open_arrays(folder, subject, basis)
        processed = 0
        for record in subject["records"]:
            _check_record_sources(record)
            index, key = record["index"], str(record["index"])
            previous = progress["slices"].get(key)
            if previous is not None:
                actual = _array_slice_hash(arrays["coeff.npy"][index], arrays["masks.npy"][index])
                if actual != previous["shard_slice_sha256"]:
                    raise ValueError(f"Completed shard slice is corrupt: {record['id']}")
                export = Path(record["output_npz"])
                if not write_source_npz or (export.exists() and sha256_file(export) == previous.get("source_npz_sha256")):
                    continue
                # Missing source export can be regenerated from verified fitted coefficients.
                model = CorpusGaussianDWI(torch.from_numpy(arrays["coeff.npy"][index].copy()).permute(1, 2, 0),
                                          arrays["masks.npy"][index, 0].copy(), basis,
                                          dict(id=record["id"], fit=previous["fit"], all_queries=True))
                model.save(export, basis_path=out / "basis.npz")
                previous["source_npz_sha256"] = sha256_file(export)
                previous["source_npz_bytes"] = export.stat().st_size
                _write(progress_path, progress)
                continue
            try:
                model, details = fit_record(record, g, b, basis, manifest["fit_config"], device=device)
                coefficients = model.latent.detach().cpu().numpy().transpose(2, 0, 1)
                arrays["coeff.npy"][index] = np.where(model.mask[None], coefficients, 0.)
                arrays["masks.npy"][index, 0] = model.mask
                for array in arrays.values():
                    array.flush()
                export = Path(record["output_npz"])
                if write_source_npz:
                    model.save(export, basis_path=out / "basis.npz")
                result = dict(id=record["id"], completed_utc=_now(), fit=details,
                              shard_slice_sha256=_array_slice_hash(arrays["coeff.npy"][index], arrays["masks.npy"][index]),
                              source_npz_sha256=sha256_file(export) if write_source_npz else None,
                              source_npz_bytes=export.stat().st_size if write_source_npz else 0)
                progress["slices"][key] = result
                progress["errors"].pop(key, None)
                progress["updated_utc"] = _now()
                _write(progress_path, progress)
                processed += 1
            except Exception as error:
                progress["errors"][key] = dict(id=record["id"], error=f"{type(error).__name__}: {error}",
                                               traceback=traceback.format_exc(), updated_utc=_now())
                _write(progress_path, progress)
                raise
            if stop_after is not None and processed >= stop_after:
                break
        complete = len(progress["slices"]) == subject["slice_count"] and not progress["errors"]
        if complete:
            files = {name: sha256_file(folder / name) for name in ARRAY_FILES}
            _write(folder / "complete.json", dict(status="complete", subject=subject["subject"],
                slice_count=subject["slice_count"], manifest_content_id=manifest["manifest_content_id"],
                basis_content_id=basis.content_id, files_sha256=files,
                source_exports_complete=bool(write_source_npz), progress_sha256=sha256_file(progress_path),
                updated_utc=_now()))
        return dict(subject=subject["subject"], completed=len(progress["slices"]),
                    total=subject["slice_count"], newly_fitted=processed, complete=complete)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def fit_worker(out, *, worker_index=0, worker_count=1, device="cpu", torch_threads=1,
               write_source_npz=True, max_new_slices=None):
    out = Path(out).resolve()
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid worker shard assignment")
    torch.set_num_threads(int(torch_threads))
    manifest = load_manifest(out)
    basis = SharedBasis.load(out / "basis.npz", device=device)
    assigned = manifest["subjects"][worker_index::worker_count]
    status_path = out / "workers" / f"worker{worker_index:04d}.json"
    fitted, finished = 0, 0
    for subject in assigned:
        _write(status_path, dict(status="fitting", pid=os.getpid(), device=device, worker_index=worker_index,
            worker_count=worker_count, subjects_finished=finished, subjects_assigned=len(assigned),
            newly_fitted=fitted, current_subject=subject["subject"], updated_utc=_now()))
        try:
            result = fit_subject(out, subject, manifest, basis, device=device, write_source_npz=write_source_npz,
                                 stop_after=None if max_new_slices is None else max_new_slices - fitted)
        except Exception as error:
            _write(status_path, dict(status="failed", pid=os.getpid(), current_subject=subject["subject"],
                                     error=f"{type(error).__name__}: {error}", updated_utc=_now()))
            raise
        fitted += result["newly_fitted"]
        finished += int(result["complete"])
        print(json.dumps(result), flush=True)
        if max_new_slices is not None and fitted >= max_new_slices:
            break
    status = dict(status="complete" if finished == len(assigned) else "partial", newly_fitted=fitted,
                  subjects_finished=finished, subjects_assigned=len(assigned), updated_utc=_now(), pid=os.getpid())
    _write(status_path, status)
    return status


def verify_inventory(manifest):
    expected = {record["dwi_path"] for record in manifest["records"]}
    actual = {str(path.resolve()) for root in (manifest["train_root"], manifest["external_root"])
              for folder in Path(root).glob("*_processed") for path in folder.glob("*_dwi.nii.gz")}
    if expected != actual:
        raise ValueError(f"Source inventory changed: missing={len(expected-actual)}, added={len(actual-expected)}")


def finalize(out, *, verify_checksums=True, standardized_cache=False):
    """Freeze all-training statistics only after every requested source is fitted."""
    out = Path(out).resolve()
    manifest = load_manifest(out)
    verify_inventory(manifest)
    if (out / "ready.json").exists():
        existing = json.loads((out / "ready.json").read_text())
        if existing["manifest_sha256"] != sha256_file(out / "manifest.json"):
            raise ValueError("Ready corpus manifest changed")
        return existing
    count = 0
    mean, m2 = np.zeros(22, np.float64), np.zeros(22, np.float64)
    image_power, image_count = 0., 0
    training_ids, completed_hashes = [], {}
    for subject_index, subject in enumerate(manifest["subjects"]):
        folder = out / subject["shard"]
        complete = json.loads((folder / "complete.json").read_text())
        if (complete["manifest_content_id"] != manifest["manifest_content_id"]
                or not complete["source_exports_complete"] or complete["slice_count"] != subject["slice_count"]):
            raise ValueError(f"Subject incomplete or belongs to another corpus: {subject['subject']}")
        progress = json.loads((folder / "progress.json").read_text())
        if complete["progress_sha256"] != sha256_file(folder / "progress.json"):
            raise ValueError("Subject completion/progress checksum mismatch")
        if verify_checksums:
            for name, expected in complete["files_sha256"].items():
                if sha256_file(folder / name) != expected:
                    raise ValueError(f"Subject shard checksum mismatch: {folder/name}")
        coefficients = np.load(folder / "coeff.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(folder / "masks.npy", mmap_mode="r", allow_pickle=False)
        for record in subject["records"]:
            _check_record_sources(record)
            result = progress["slices"][str(record["index"])]
            export = Path(record["output_npz"])
            if (not export.exists() or export.stat().st_size != result["source_npz_bytes"]
                    or (verify_checksums and sha256_file(export) != result["source_npz_sha256"])):
                raise ValueError(f"Source label missing/changed: {export}")
            if subject["split"] != "train":
                continue
            training_ids.append(record["id"])
            active = coefficients[record["index"]].transpose(1, 2, 0)[masks[record["index"], 0]].astype(np.float64)
            if not np.isfinite(active).all():
                raise ValueError("Nonfinite training coefficients")
            if len(active):
                batch_count, batch_mean = len(active), active.mean(0)
                delta = batch_mean - mean
                m2 += ((active - batch_mean) ** 2).sum(0) + delta ** 2 * count * batch_count / (count + batch_count)
                mean += delta * batch_count / (count + batch_count)
                count += batch_count
            image_power += result["fit"]["fitted_signal_square_sum"]
            image_count += result["fit"]["fitted_signal_values"]
        completed_hashes[subject["shard"]] = sha256_file(folder / "complete.json")
        _write(out / "finalize_status.json", dict(stage="training_statistics_and_verification",
            checked_subjects=subject_index + 1, total_subjects=len(manifest["subjects"]),
            training_centers=count, updated_utc=_now()))
    if count == 0:
        raise ValueError("Entire training corpus has no active Gaussian centers")
    std = np.sqrt(m2 / count)
    floor = max(float(std.max()) * 1e-8, 1e-8)
    normalization = ChannelNormalization(mean, np.maximum(std, floor), basis_content_id=manifest["basis_content_id"],
        image_scale=max(np.sqrt(image_power / max(image_count, 1)), 1e-8),
        metadata=dict(training_ids=training_ids, training_subjects=manifest["counts"]["train"]["subjects"],
            training_slices=len(training_ids), active_centers=count, protocol="full raw positive-b measurements",
            held_subjects_used=False, external_subjects_used=False,
            weighting="active-center count; FP64 population moments", std_floor=floor,
            image_scale_definition="pooled all-training fitted Gaussian brain RMS; no external labels used"))
    normalization.save(out / "normalization.npz")
    if standardized_cache:
        for subject in manifest["subjects"]:
            folder = out / subject["shard"]
            raw = np.load(folder / "coeff.npy", mmap_mode="r", allow_pickle=False)
            masks = np.load(folder / "masks.npy", mmap_mode="r", allow_pickle=False)
            dest = np.lib.format.open_memmap(folder / "normalized.npy", mode="w+", dtype=np.float32, shape=raw.shape)
            for index in range(len(raw)):
                dest[index] = normalization.normalize(torch.from_numpy(raw[index].copy()), torch.from_numpy(masks[index].copy())).numpy()
            dest.flush()
    manifest["files_sha256"]["normalization.npz"] = sha256_file(out / "normalization.npz")
    manifest["finalized_utc"] = _now()
    _write(out / "manifest.json", manifest)
    ready = dict(status="ready", manifest_content_id=manifest["manifest_content_id"],
                 manifest_sha256=sha256_file(out / "manifest.json"), basis_content_id=manifest["basis_content_id"],
                 normalization_sha256=sha256_file(out / "normalization.npz"), counts=manifest["counts"],
                 complete_sha256=completed_hashes, standardized_cache=bool(standardized_cache),
                 source_exports_verified=True, source_exports_sha256_checked=bool(verify_checksums),
                 normalization_training_only=True, updated_utc=_now())
    _write(out / "ready.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("manifest")
    scan.add_argument("--out", type=Path, required=True)
    scan.add_argument("--train-root", type=Path, required=True)
    scan.add_argument("--external-root", type=Path, required=True)
    scan.add_argument("--basis", type=Path, required=True)
    scan.add_argument("--scan-workers", type=int, default=16)
    worker = sub.add_parser("fit")
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, default=0)
    worker.add_argument("--worker-count", type=int, default=1)
    worker.add_argument("--device", default="cpu")
    worker.add_argument("--torch-threads", type=int, default=1)
    worker.add_argument("--max-new-slices", type=int)
    finish = sub.add_parser("finalize")
    finish.add_argument("--out", type=Path, required=True)
    finish.add_argument("--standardized-cache", action="store_true")
    args = parser.parse_args()
    if args.command == "manifest":
        result = build_manifest(args.out, args.train_root, args.external_root, args.basis, scan_workers=args.scan_workers)
        print(json.dumps(dict(counts=result["counts"], manifest_content_id=result["manifest_content_id"])), flush=True)
    elif args.command == "fit":
        fit_worker(args.out, worker_index=args.worker_index, worker_count=args.worker_count,
                   device=args.device, torch_threads=args.torch_threads, max_new_slices=args.max_new_slices)
    else:
        print(json.dumps(finalize(args.out, standardized_cache=args.standardized_cache)), flush=True)


if __name__ == "__main__":
    main()
