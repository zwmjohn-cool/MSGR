"""Reuse frozen training targets and authenticate all external representatives.

The 265-person monitor cache contains no training records, so it cannot use
the mixed-cohort validator from ``gaussian_latent_fm.data``. Its statistics,
decoder and Gaussian labels must instead match an already validated training
manifest. No statistic is fitted or updated by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from gaussian_kl_vae.data import sha256_file
from gaussian_latent_fm.data import (
    FORMAT, LatentGaussianDataset, LatentNormalization, EpochBatchSampler,
    validate_cache,
)


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_RULE = "middle nonempty native slice per subject, sorted by slice index; all actual diffusion queries"


def external_subject_inventory(rest_root):
    """Count raw/processed/slices directories once, and expose missing pairs."""
    root = Path(rest_root).resolve()
    original, processed = set(), set()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if re.fullmatch(r"[0-9]+", path.name):
            original.add(path.name)
        elif match := re.fullmatch(r"([0-9]+)_(?:processed|slices)", path.name):
            processed.add(match.group(1))
    if not original or original != processed:
        raise ValueError(
            "External raw/processed subject inventories differ: "
            f"unprocessed={sorted(original - processed)}, missing_raw={sorted(processed - original)}"
        )
    return sorted(original)


def representative_records(corpus_records):
    """The existing Q-MMTN/representative265 rule, independent of model error."""
    by_subject = {}
    for record in corpus_records:
        if record["split"] == "external":
            by_subject.setdefault(str(record["subject"]), []).append(record)
    result = []
    for subject in sorted(by_subject):
        cases = sorted(by_subject[subject], key=lambda record: int(record["slice"]))
        nonempty = [record for record in cases if int(record["mask_voxels"]) > 0]
        eligible = nonempty or cases
        result.append(eligible[len(eligible) // 2])
    return result


def _check_digest(path, expected, description):
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"{description} checksum differs: {path}")


def _check_source_stat(path, expected, description):
    actual = Path(path).stat()
    size = expected.get("bytes", expected.get("size"))
    if actual.st_size != size or actual.st_mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"{description} source changed: {path}")


def validate_monitor_cache(cache, training_manifest, *, rest_root=ROOT / "hcp_datasets_rest",
                           expected_subjects=265, verify_shards=True):
    """Authenticate the complete held-out cohort against frozen training assets.

    ``training_manifest`` must come from ``validate_cache``. Full hashes of
    the small per-representative arrays are checked by default; these amount
    to only one condition/latent slice per external subject.
    """
    root = Path(cache).resolve()
    rest_root = Path(rest_root).resolve()
    ready = json.loads((root / "ready.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if (ready.get("status") != "ready" or ready.get("training_only_latent_statistics") is not True
            or manifest.get("format") != FORMAT):
        raise ValueError("External representative cache is incomplete or incompatible")
    _check_digest(root / "manifest.json", ready.get("manifest_sha256"), "Monitor manifest")
    for name, digest in manifest["files_sha256"].items():
        _check_digest(root / name, digest, "Monitor asset")
    if (manifest["files_sha256"].get("latent_normalization.npz") !=
            training_manifest["files_sha256"].get("latent_normalization.npz")):
        raise ValueError("Monitor latent statistics differ from training-only statistics")
    normalization = LatentNormalization.load(root / "latent_normalization.npz")
    train_subjects = {str(record["subject"]) for record in training_manifest["records"]
                      if record["split"] == "train"}
    if (not train_subjects or set(normalization.metadata.get("training_subjects", ())) != train_subjects):
        raise ValueError("Monitor statistics must name exactly the training subjects")
    for name in ("source", "basis_content_id", "native_shape", "padded_shape", "latent_shape",
                 "gaussian_shape", "condition_order", "condition_preprocessing"):
        if manifest.get(name) != training_manifest.get(name):
            raise ValueError(f"Monitor/training cache {name} differs")

    source = manifest["source"]
    corpus_path = Path(source["corpus_manifest"])
    _check_digest(corpus_path, source["corpus_manifest_sha256"], "Gaussian corpus")
    _check_digest(corpus_path.parent / "ready.json", source["corpus_ready_sha256"], "Gaussian completion")
    _check_digest(source["export_manifest"], source["export_manifest_sha256"], "VAE latent export")
    _check_digest(source["decoder_checkpoint"], source["decoder_sha256"], "Frozen VAE decoder")
    corpus = json.loads(corpus_path.read_text())
    for name in ("basis.npz", "normalization.npz"):
        _check_digest(corpus_path.parent / name, corpus["files_sha256"][name], "Gaussian decoding asset")

    records = manifest["records"]
    ids = [record["id"] for record in records]
    subjects = [str(record["subject"]) for record in records]
    inventory = external_subject_inventory(rest_root)
    if (len(records) != expected_subjects or len(set(subjects)) != expected_subjects
            or len(set(ids)) != len(ids) or sorted(subjects) != inventory
            or any(record["split"] != "external" for record in records)):
        raise ValueError("Monitor must contain exactly one representative of every external subject")
    if train_subjects & set(subjects):
        raise ValueError("Training and monitoring subjects overlap")
    expected = representative_records(corpus["records"])
    if ids != [record["id"] for record in expected] or manifest.get("record_ids") != ids:
        raise ValueError("Monitor representatives differ from the frozen middle-slice rule")
    for actual, canonical in zip(records, expected):
        if any(actual.get(key) != value for key, value in canonical.items()):
            raise ValueError(f"Monitor Gaussian record identity differs: {actual['id']}")
        if rest_root not in Path(actual["dwi_path"]).resolve().parents:
            raise ValueError(f"Monitor DWI is outside the requested external root: {actual['id']}")
        for kind, path_key in (("dwi", "dwi_path"), ("mask", "mask_path")):
            _check_source_stat(actual[path_key], actual["source_stats"][kind], "Measured monitor")

    markers = manifest["subjects"]
    if len(markers) != expected_subjects or sorted(str(item["subject"]) for item in markers) != inventory:
        raise ValueError("Monitor completion records do not cover the external inventory")
    by_subject = {str(record["subject"]): record for record in records}
    expected_signature = dict(
        corpus_root=str(corpus_path.parent), decoder_checkpoint=source["decoder_checkpoint"],
        decoder_sha256=source["decoder_sha256"], checkpoint_step=int(source["checkpoint_step"]),
        gaussian_normalization=str(corpus_path.parent / "normalization.npz"),
        basis_content_id=manifest["basis_content_id"],
        corpus_manifest_sha256=source["corpus_manifest_sha256"],
    )
    for subject in markers:
        record = by_subject[str(subject["subject"])]
        folder = root / subject["cache_shard"]
        marker_path = folder / "complete.json"
        _check_digest(marker_path, subject["complete_sha256"], "Monitor subject completion")
        marker = json.loads(marker_path.read_text())
        if (subject.get("count") != 1 or subject.get("split") != "external"
                or record["cache_shard"] != subject["cache_shard"] or record["cache_index"] != 0
                or marker.get("count") != 1 or marker.get("subject") != record["subject"]
                or marker.get("split") != "external"
                or marker.get("signature") != {"ids": [record["id"]], "source_signature": expected_signature}):
            raise ValueError(f"Monitor shard identity differs: {record['id']}")
        shapes = {"conditions.npy": (1, 3, *manifest["native_shape"]),
                  "latents.npy": (1, *manifest["latent_shape"])}
        # The aligned cache and its manifest both use latent CHW order.
        for name, shape in shapes.items():
            array = np.load(folder / name, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f"Monitor array shape/dtype differs: {folder / name}")
            if verify_shards:
                _check_digest(folder / name, marker["files_sha256"].get(name), "Monitor array")
        source_records = marker.get("source_records", [])
        if len(source_records) != 1 or source_records[0].get("id") != record["id"]:
            raise ValueError(f"Monitor source identity differs: {record['id']}")
        original = source_records[0]
        for item in (original["latent"], *original["conditions"].values()):
            _check_source_stat(item["path"], item, "Monitor cached input")
    return manifest


def load_monitor_dataset(cache, training_manifest, *, seed=20261101,
                         rest_root=ROOT / "hcp_datasets_rest", expected_subjects=265,
                         verify_shards=True):
    manifest = validate_monitor_cache(cache, training_manifest, rest_root=rest_root,
                                      expected_subjects=expected_subjects, verify_shards=verify_shards)
    dataset = LatentGaussianDataset(cache, "external", queries_per_sample=None,
                                   manifest=manifest, seed=seed)
    ids = [record["id"] for record in manifest["records"]]
    dataset.monitor_identity = dict(
        cache=str(Path(cache).resolve()), manifest_sha256=sha256_file(Path(cache) / "manifest.json"),
        representative_ids=ids,
        representative_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        subjects=len(ids), rest_root=str(Path(rest_root).resolve()), selection=REPRESENTATIVE_RULE,
        statistics="frozen training-only; no external update", used_for_selection=False,
    )
    return dataset


__all__ = ["LatentGaussianDataset", "LatentNormalization", "EpochBatchSampler", "validate_cache",
           "validate_monitor_cache", "load_monitor_dataset", "external_subject_inventory",
           "representative_records"]
