"""Stream all fixed-shared Gaussian slices from authenticated subject shards.

The corpus is fitted afresh by :mod:`gaussian_kl_vae.corpus`. No legacy C3/OT
labels or pilot normalization enters this dataset. Query selection happens
before collation, so subjects with different acquisition lengths can share a
training batch. All image geometry remains in the native coordinate system.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .data import ChannelNormalization, sha256_file
from spatial_gaussian_dwi.shared_model import SharedBasis


FORMAT = "gaussian_kl_vae.fixed_shared_corpus.v1"


def validate_corpus(root, *, verify_shard_checksums=False):
    """Require a completely refitted corpus and frozen training-only statistics.

    Small decoding assets and the manifest are always authenticated. Rehashing
    every coefficient shard is optional because it rereads the entire corpus;
    all subject completion markers and array shapes are checked in either case.
    """
    root = Path(root).resolve()
    ready_path = root / "ready.json"
    if not ready_path.is_file():
        raise ValueError("Full refit and training-only normalization must finish before VAE training: ready.json missing")
    ready = json.loads(ready_path.read_text())
    if (ready.get("status") != "ready" or not ready.get("source_exports_verified")
            or not ready.get("normalization_training_only")):
        raise ValueError("Corpus refit/source export/training-only normalization gate is incomplete")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Expected the freshly fitted fixed-shared Gaussian corpus")
    if ready.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Corpus ready marker/manifest identity differs")
    for name in ("basis.npz", "normalization.npz"):
        if sha256_file(root / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Frozen corpus decoding asset checksum differs: {name}")
    basis = SharedBasis.load(root / "basis.npz")
    if basis.content_id != manifest["basis_content_id"]:
        raise ValueError("Corpus public basis identity differs")
    normalization = ChannelNormalization.load(root / "normalization.npz", basis_content_id=basis.content_id)
    records = manifest["records"]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Corpus slice IDs are not unique")
    roles = {record["split"] for record in records}
    if not roles <= {"train", "external"}:
        raise ValueError("Corpus contains an unknown subject role")
    training_subjects = {str(record["subject"]) for record in records if record["split"] == "train"}
    external_subjects = {str(record["subject"]) for record in records if record["split"] == "external"}
    if training_subjects & external_subjects:
        raise ValueError("Training and external subject cohorts overlap")
    training_ids = [record["id"] for record in records if record["split"] == "train"]
    if normalization.metadata.get("training_ids") != training_ids:
        raise ValueError("Frozen normalization does not cover exactly the training slices")
    if normalization.metadata.get("external_subjects_used", False):
        raise ValueError("External subjects entered normalization statistics")
    by_shard = {}
    for record in records:
        by_shard.setdefault(record["shard"], []).append(record)
    if set(by_shard) != {subject["shard"] for subject in manifest["subjects"]}:
        raise ValueError("Corpus record/shard inventory differs")
    for subject in manifest["subjects"]:
        subject_records = by_shard[subject["shard"]]
        if (sorted(int(record["index"]) for record in subject_records) != list(range(subject["slice_count"]))
                or any(record["split"] != subject["split"] or str(record["subject"]) != str(subject["subject"])
                       for record in subject_records)):
            raise ValueError(f"Corpus does not cover every subject slice exactly once: {subject['subject']}")
        shard = root / subject["shard"]
        marker = shard / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Subject refit is incomplete: {subject['subject']}")
        completed = json.loads(marker.read_text())
        expected = ready["complete_sha256"].get(subject["shard"])
        if not expected or sha256_file(marker) != expected:
            raise ValueError(f"Subject completion marker changed: {subject['subject']}")
        if (completed.get("status") != "complete" or not completed.get("source_exports_complete")
                or completed.get("slice_count") != subject["slice_count"]
                or completed.get("basis_content_id") != basis.content_id):
            raise ValueError(f"Subject refit/source export gate is incomplete: {subject['subject']}")
        shapes = {
            "coeff.npy": (subject["slice_count"], basis.rank, *subject["native_shape"]),
            "masks.npy": (subject["slice_count"], 1, *subject["native_shape"]),
            "directions.npy": (subject["query_count"], 3),
            "bvals.npy": (subject["query_count"],),
            "query_basis.npy": (subject["query_count"], basis.rank),
        }
        for name, shape in shapes.items():
            array = np.load(shard / name, mmap_mode="r", allow_pickle=False)
            if array.shape != shape:
                raise ValueError(f"Subject array shape differs: {shard / name}")
            expected_dtype = np.bool_ if name == "masks.npy" else np.float32
            if array.dtype != expected_dtype:
                raise ValueError(f"Subject array dtype differs: {shard / name}")
        if verify_shard_checksums:
            for name, digest in completed["files_sha256"].items():
                if sha256_file(shard / name) != digest:
                    raise ValueError(f"Subject array checksum differs: {shard / name}")
    return manifest


def shell_query_indices(bvals, count, *, seed, epoch, identity):
    """Equal-shell queries, deterministic for this case and true data epoch."""
    if count < 3 or count % 3:
        raise ValueError("queries_per_sample must be a positive multiple of three")
    shell = np.rint(np.asarray(bvals) / 1000).astype(np.int64)
    if set(shell.tolist()) != {1, 2, 3}:
        raise ValueError("Expected measured positive-b shells near 1000, 2000, 3000")
    digest = hashlib.blake2b(f"{seed}/{epoch}/{identity}".encode(), digest_size=8).digest()
    generator = np.random.default_rng(int.from_bytes(digest, "little"))
    selected = []
    for label in (1, 2, 3):
        available = np.flatnonzero(shell == label)
        if len(available) < count // 3:
            raise ValueError(f"Insufficient shell {label} measurements for {count} queries")
        selected.append(generator.choice(available, count // 3, replace=False))
    return np.concatenate(selected)


class CorpusGaussianDataset(Dataset):
    """Read raw FP32 coefficients, normalize with frozen corpus stats, then pad.

    Training indices can be ``(epoch,index)`` pairs. This makes shell sampling
    independent of worker scheduling and prefetch, including after resuming.
    ``queries_per_sample=None`` exposes every measured direction for evaluation.
    """
    def __init__(self, root, split="train", *, records=None, queries_per_sample=None,
                 seed=20261028, max_open_shards=16, manifest=None,
                 verify_shard_checksums=False):
        if split not in ("train", "external", "all"):
            raise ValueError("split must be train, external or all")
        self.root = Path(root).resolve()
        self.manifest = manifest or validate_corpus(self.root, verify_shard_checksums=verify_shard_checksums)
        self.basis = SharedBasis.load(self.root / "basis.npz")
        self.normalization = ChannelNormalization.load(self.root / "normalization.npz", basis_content_id=self.basis.content_id)
        eligible = [record for record in self.manifest["records"] if split == "all" or record["split"] == split]
        if records is not None:
            lookup = {record["id"]: record for record in eligible}
            if any(record["id"] not in lookup for record in records):
                raise ValueError("Selected monitoring records do not belong to the requested corpus split")
            self.records = [lookup[record["id"]] for record in records]
        else:
            self.records = eligible
        if not self.records:
            raise ValueError("Requested corpus split contains no slices")
        shapes = {tuple(subject["native_shape"]) for subject in self.manifest["subjects"]}
        if len(shapes) != 1:
            raise ValueError("The convolutional VAE requires a common native spatial shape")
        self.native_shape = shapes.pop()
        self.padded_shape = tuple(self.manifest.get("padded_shape", (160, 176)))
        if any(native > padded for native, padded in zip(self.native_shape, self.padded_shape)):
            raise ValueError("Padded shape cannot crop native anatomy")
        self.subject_count = len({record["subject"] for record in self.records})
        self.queries_per_sample = queries_per_sample
        self.seed = int(seed)
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards = OrderedDict()
        self._queries = {}

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shards"] = OrderedDict()
        state["_queries"] = {}
        return state

    def _subject_arrays(self, shard_name):
        if shard_name not in self._shards:
            folder = self.root / shard_name
            self._shards[shard_name] = {
                "coeff": np.load(folder / "coeff.npy", mmap_mode="r", allow_pickle=False),
                "mask": np.load(folder / "masks.npy", mmap_mode="r", allow_pickle=False),
            }
            if len(self._shards) > self.max_open_shards:
                self._shards.popitem(last=False)
        self._shards.move_to_end(shard_name)
        if shard_name not in self._queries:
            folder = self.root / shard_name
            self._queries[shard_name] = {
                key: np.load(folder / f"{key}.npy", allow_pickle=False)
                for key in ("directions", "bvals", "query_basis")
            }
        return self._shards[shard_name], self._queries[shard_name]

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        record = self.records[int(index)]
        arrays, queries = self._subject_arrays(record["shard"])
        slot = int(record["index"])
        coefficients = torch.from_numpy(arrays["coeff"][slot].copy())
        mask = torch.from_numpy(arrays["mask"][slot].copy())
        normalized = self.normalization.normalize(coefficients, mask)
        h, w = self.native_shape
        pad = (0, self.padded_shape[1] - w, 0, self.padded_shape[0] - h)
        selected = np.arange(len(queries["bvals"]))
        if self.queries_per_sample is not None:
            selected = shell_query_indices(queries["bvals"], self.queries_per_sample,
                                           seed=self.seed, epoch=epoch, identity=record["id"])
        return dict(x=F.pad(normalized, pad), mask=F.pad(mask, pad),
                    directions=torch.from_numpy(queries["directions"][selected].copy()),
                    bvals=torch.from_numpy(queries["bvals"][selected].copy()),
                    query_basis=torch.from_numpy(queries["query_basis"][selected].copy()),
                    query_indices=torch.from_numpy(selected.copy()),
                    id=record["id"], subject=str(record["subject"]), slice=int(record["slice"]),
                    index=int(index), epoch=int(epoch), empty_brain=not bool(mask.any()))

    def getmeasured(self, index):
        """Raw DWI for monitoring only; never used by the training objective."""
        import nibabel as nib
        record = self.records[index]
        measured = np.asanyarray(nib.load(record["dwi_path"]).dataobj, dtype=np.float32).squeeze()
        if measured.ndim != 3 or measured.shape[:2] != self.native_shape:
            raise ValueError(f"Measured DWI shape differs for {record['id']}")
        return torch.from_numpy(np.array(measured, dtype=np.float32, copy=True)).permute(2, 0, 1)


class EpochBatchSampler(Sampler):
    """An exact shuffled visit of every slice, with resumable sample offset."""
    def __init__(self, count, batch_size, *, seed=20261028, epoch=0, offset=0):
        if count <= 0 or batch_size <= 0:
            raise ValueError("Dataset count and batch size must be positive")
        self.count, self.batch_size, self.seed = int(count), int(batch_size), int(seed)
        self.set_position(epoch, offset)

    def set_position(self, epoch, offset=0):
        if epoch < 0 or not 0 <= offset <= self.count:
            raise ValueError("Invalid epoch/sample offset")
        self.epoch, self.offset = int(epoch), int(offset)

    def __len__(self):
        return (self.count - self.offset + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % (2**63 - 1))
        permutation = torch.randperm(self.count, generator=generator).tolist()
        for start in range(self.offset, self.count, self.batch_size):
            yield [(self.epoch, index) for index in permutation[start:start + self.batch_size]]


def select_external_monitor(manifest, root, *, count=20, seed=20261028):
    """Freeze one central nonempty slice for each seeded external subject.

    Selection inspects masks only, never measured or fitted DWI intensities.
    These subjects retain their external-test role and cannot select checkpoints.
    """
    subjects = sorted([entry for entry in manifest["subjects"] if entry["split"] == "external"],
                      key=lambda entry: str(entry["subject"]))
    if count <= 0 or count > len(subjects):
        raise ValueError("External monitor count must fit the available subject cohort")
    generator = np.random.default_rng(seed)
    selected_subjects = sorted(generator.choice(len(subjects), count, replace=False))
    by_subject = {}
    for record in manifest["records"]:
        if record["split"] == "external":
            by_subject.setdefault(str(record["subject"]), []).append(record)
    result = []
    for index in selected_subjects:
        subject = subjects[index]
        cases = sorted(by_subject[str(subject["subject"])], key=lambda entry: int(entry["slice"]))
        masks = np.load(Path(root) / subject["shard"] / "masks.npy", mmap_mode="r", allow_pickle=False)
        active = [record for record in cases if masks[int(record["index"])].any()]
        eligible = active or cases
        result.append(eligible[len(eligible) // 2])
    return result
