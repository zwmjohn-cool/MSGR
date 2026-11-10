"""Aligned, memory-mapped MRI conditions and final-EMA VAE latent targets.

Only the training cohort enters latent channel statistics. Gaussian targets
reuse the authenticated fixed-shared R22 corpus; no old C3/OT target is read.
All scalar MRI arrays retain native row/column orientation and bottom/right
padding. Measured DWI is an evaluation-only reference in raw intensity units.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from gaussian_kl_vae.corpus_data import CorpusGaussianDataset, EpochBatchSampler
from gaussian_kl_vae.data import sha256_file


FORMAT = "gaussian_latent_fm.aligned_corpus.v1"
CONDITION_ORDER = ("t1", "t2", "b0")


class LatentNormalization(torch.nn.Module):
    """Frozen training-only affine transform, invertible with autograd."""
    def __init__(self, mean, std, metadata=None):
        super().__init__()
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if (mean.ndim != 1 or mean.shape != std.shape or not len(mean)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0)):
            raise ValueError("Latent channel mean/std must be finite vectors with positive std")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.metadata = dict(metadata or {})

    def _constants(self, x):
        if x.ndim not in (3, 4) or x.shape[-3] != len(self.mean):
            raise ValueError("Expected latent [C,H,W] or [N,C,H,W]")
        shape = (1,) * (x.ndim - 3) + (-1, 1, 1)
        return self.mean.to(x).reshape(shape), self.std.to(x).reshape(shape)

    def normalize(self, x):
        mean, std = self._constants(x)
        return (x - mean) / std

    def inverse(self, x):
        mean, std = self._constants(x)
        return x * std + mean

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())
            if metadata.get("external_subjects_used") is not False:
                raise ValueError("Latent statistics must exclude external subjects")
            return cls(archive["mean"].copy(), archive["std"].copy(), metadata)


def validate_cache(root, *, verify_shards=False):
    """Authenticate small frozen assets; optional full array checksums.

    Preparation validates every source latent and every reused condition slab.
    Startup avoids rereading the multi-GB arrays, but authenticates all subject
    completion records, the decoder, and the original Gaussian corpus identity.
    """
    root = Path(root).resolve()
    ready = json.loads((root / "ready.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if ready.get("status") != "ready" or manifest.get("format") != FORMAT:
        raise ValueError("Latent/MRI cache is incomplete or incompatible")
    if sha256_file(root / "manifest.json") != ready["manifest_sha256"]:
        raise ValueError("Latent cache manifest identity differs")
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Latent cache decoding asset changed: {name}")
    source = manifest["source"]
    for key, hash_key in (("corpus_manifest", "corpus_manifest_sha256"),
                          ("export_manifest", "export_manifest_sha256")):
        if sha256_file(source[key]) != source[hash_key]:
            raise ValueError(f"Frozen source identity changed: {key}")
    corpus_root = Path(source["corpus_manifest"]).parent
    if sha256_file(corpus_root / "ready.json") != source["corpus_ready_sha256"]:
        raise ValueError("Frozen Gaussian corpus completion identity changed")
    corpus_manifest = json.loads(Path(source["corpus_manifest"]).read_text())
    for name in ("basis.npz", "normalization.npz"):
        if sha256_file(corpus_root / name) != corpus_manifest["files_sha256"][name]:
            raise ValueError(f"Frozen Gaussian decoding asset changed: {name}")
    checkpoint = Path(source["decoder_checkpoint"])
    st = checkpoint.stat()
    if st.st_size != source["decoder_bytes"] or st.st_mtime_ns != source["decoder_mtime_ns"]:
        if sha256_file(checkpoint) != source["decoder_sha256"]:
            raise ValueError("Frozen VAE decoder changed")
    train = {r["subject"] for r in manifest["records"] if r["split"] == "train"}
    external = {r["subject"] for r in manifest["records"] if r["split"] == "external"}
    if train & external or not train:
        raise ValueError("Train/external subjects overlap or training cohort is empty")
    for subject in manifest["subjects"]:
        folder = root / subject["cache_shard"]
        if sha256_file(folder / "complete.json") != subject["complete_sha256"]:
            raise ValueError(f"Subject cache marker changed: {subject['subject']}")
        if verify_shards:
            marker = json.loads((folder / "complete.json").read_text())
            for name, digest in marker["files_sha256"].items():
                if sha256_file(folder / name) != digest:
                    raise ValueError(f"Subject cache array changed: {folder / name}")
    return manifest


def source_paths(record):
    path = Path(record["dwi_path"])
    prefix = path.name.removesuffix("_dwi.nii.gz")
    return {name: path.with_name(f"{prefix}_{name}.nii.gz") for name in (*CONDITION_ORDER, "mask")}


def read_raw_conditions(record):
    """Read original aligned T1/T2/B0 only, for offline report panels."""
    import nibabel as nib
    paths = source_paths(record)
    mask_image = nib.load(paths["mask"])
    result = []
    for name in CONDITION_ORDER:
        image = nib.load(paths[name])
        array = np.asarray(image.dataobj, dtype=np.float32).squeeze()
        if (array.shape != tuple(record["native_shape"]) or not np.isfinite(array).all()
                or not np.allclose(image.affine, mask_image.affine, atol=1e-5, rtol=0)):
            raise ValueError(f"Condition shape/affine/values differ: {paths[name]}")
        result.append(array)
    return np.stack(result, axis=0)


def preprocess_conditions(raw):
    """Source-only p1/p99 T1/T2; fixed B0/1000 preserves raw amplitude."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected finite MRI [3,H,W]")
    scaled = np.zeros_like(raw)
    for channel in (0, 1):
        positive = raw[channel] > 0
        values = raw[channel][positive].astype(np.float64)
        if values.size:
            low, high = np.percentile(values, [1, 99])
            if high > low:
                scaled[channel][positive] = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    scaled[2] = raw[2] / np.float32(1000.)
    return scaled


class LatentGaussianDataset(CorpusGaussianDataset):
    """Final EMA posterior means + source MRI + current Gaussian GT.

    ``z`` is channel-standardized [8,40,44]; ``x`` is the VAE's standardized
    Gaussian target [22,160,176]. ``condition`` is [3,160,176], ordered T1/T2/B0.
    ``queries_per_sample=None`` exposes all actual measured directions; there
    is no 270-direction truncation. Epoch-index pairs give resumable sampling.
    """
    def __init__(self, root, split="train", *, records=None, queries_per_sample=12,
                 seed=20261030, max_open_shards=24, manifest=None):
        self.cache_root = Path(root).resolve()
        self.cache_manifest = manifest or validate_cache(self.cache_root)
        if split not in ("train", "external", "all"):
            raise ValueError("Unknown subject split")
        selected = [r for r in self.cache_manifest["records"] if split == "all" or r["split"] == split]
        if records is not None:
            lookup = {r["id"]: r for r in selected}
            selected = [lookup[r if isinstance(r, str) else r["id"]] for r in records]
        corpus_manifest = json.loads(Path(self.cache_manifest["source"]["corpus_manifest"]).read_text())
        super().__init__(Path(self.cache_manifest["source"]["corpus_manifest"]).parent,
                         split, records=selected, queries_per_sample=queries_per_sample,
                         seed=seed, max_open_shards=max_open_shards, manifest=corpus_manifest)
        # The parent canonicalizes records against the frozen corpus. Attach only
        # cache addresses, keeping the authoritative Gaussian shard/index intact.
        extra = {r["id"]: r for r in selected}
        self.records = [dict(r, cache_shard=extra[r["id"]]["cache_shard"],
                             cache_index=extra[r["id"]]["cache_index"]) for r in self.records]
        self.latent_normalization = LatentNormalization.load(self.cache_root / "latent_normalization.npz")
        self._latent_shards = OrderedDict()

    @property
    def latent_mean(self):
        return self.latent_normalization.mean

    @property
    def latent_std(self):
        return self.latent_normalization.std

    def __getstate__(self):
        state = super().__getstate__()
        state["_latent_shards"] = OrderedDict()
        return state

    def _latent_arrays(self, shard):
        if shard not in self._latent_shards:
            folder = self.cache_root / shard
            self._latent_shards[shard] = {name: np.load(folder / f"{name}.npy", mmap_mode="r")
                                         for name in ("latents", "conditions")}
            if len(self._latent_shards) > self.max_open_shards:
                self._latent_shards.popitem(last=False)
        self._latent_shards.move_to_end(shard)
        return self._latent_shards[shard]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        record = self.records[sample["index"]]
        arrays = self._latent_arrays(record["cache_shard"])
        slot = record["cache_index"]
        latent = torch.from_numpy(arrays["latents"][slot].copy())
        conditions = torch.from_numpy(arrays["conditions"][slot].copy())
        h, w = self.native_shape
        pad = (0, self.padded_shape[1] - w, 0, self.padded_shape[0] - h)
        sample.update(z=self.latent_normalization.normalize(latent), condition=F.pad(conditions, pad),
                      queries=torch.cat((sample["directions"], sample["bvals"][:, None]), dim=1))
        return sample

    def getrawconditions(self, index):
        return torch.from_numpy(read_raw_conditions(self.records[index]).copy())


__all__ = ["LatentGaussianDataset", "LatentNormalization", "EpochBatchSampler", "validate_cache"]
