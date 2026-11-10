"""Raw measured-DWI supervision aligned to the frozen Gaussian latent queries.

The existing Q-MMTN query archive is only a lossless I/O cache. Its read-only
format checks are adapted from qmmtn_baseline/data.py; no Q-MMTN preprocessing,
augmentation, signal normalization, cache writer, or dependency is imported.
Only the exact query_indices chosen by LatentGaussianDataset are extracted.
"""
from __future__ import annotations

from collections import OrderedDict
import gzip
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
import torch

from gaussian_kl_vae.data import sha256_file
from gaussian_latent_fm.data import LatentGaussianDataset


QUERY_CACHE_FORMAT = 'qmmtn.lossless_measured_queries.v1'


def _source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _verify_stat(path, expected):
    actual = _source_identity(path)
    if actual['size'] != expected['size'] or actual['mtime_ns'] != expected['mtime_ns']:
        raise ValueError(f'Frozen measured source changed: {path}')
    return actual


def read_query_cache(path, source, affine, shape, count, selected):
    """Read selected raw FP32 planes, preserving their requested order exactly."""
    try:
        with zipfile.ZipFile(path, 'r') as archive:
            metadata = json.loads(archive.read('metadata.json'))
            if (metadata.get('format') != QUERY_CACHE_FORMAT or metadata.get('source') != source
                    or metadata.get('shape') != [int(count), *shape]
                    or metadata.get('dtype') != 'float32'):
                raise ValueError('cache format/source/shape identity differs')
            cache_affine = np.asarray(metadata['affine'], dtype=np.float64)
            if cache_affine.shape != (4, 4) or not np.allclose(cache_affine, affine, atol=1e-5, rtol=0):
                raise ValueError('cache affine differs from original DWI and mask')
            expected = {'metadata.json', *(f'q{i:04d}.npy' for i in range(count))}
            names = archive.namelist()
            if len(names) != len(expected) or set(names) != expected:
                raise ValueError('cache query inventory is incomplete or duplicated')
            planes = []
            for index in selected:
                if not 0 <= int(index) < count:
                    raise ValueError('requested query lies outside the acquisition')
                with archive.open(f'q{int(index):04d}.npy', 'r') as stream:
                    plane = np.load(stream, allow_pickle=False)
                if plane.dtype != np.float32 or plane.shape != tuple(shape) or not np.isfinite(plane).all():
                    raise ValueError(f'invalid cached query {index}')
                planes.append(plane)
            return np.stack(planes)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as error:
        raise ValueError(f'Invalid measured-query cache {path}: {error}') from error


class MeasuredLatentGaussianDataset(LatentGaussianDataset):
    """Training-only adapter adding raw measured [Q,H,W] image targets.

    MRI conditions and latent/Gaussian targets are inherited unchanged. Cached
    measured planes are never padded, divided by B0, clipped, or resampled.
    Missing archives fall back to original NIfTI without writing any cache.
    """

    def __init__(self, root, split='train', *, measured_cache=None, **kwargs):
        if split != 'train':
            raise ValueError('Measured supervision is restricted to the training split')
        super().__init__(root, split, **kwargs)
        if any(record['split'] != 'train' for record in self.records):
            raise ValueError('External subjects cannot enter measured supervision')
        self.measured_cache = Path(measured_cache).resolve() if measured_cache is not None else None
        self._measured_subjects = {str(subject['subject']): subject for subject in self.manifest['subjects']}
        self._measured_query_tables = {}
        self._measured_headers = OrderedDict()
        self.measured_target_identity = dict(
            target='original measured DWI in raw FP32 intensity',
            measured_cache=str(self.measured_cache) if self.measured_cache else None,
            cache_format=QUERY_CACHE_FORMAT, cache_read_only=True,
            missing_cache='read original NIfTI without creating an archive',
            alignment='exact parent query_indices; original acquisition hashes and native affine verified',
        )

    def __getstate__(self):
        state = super().__getstate__()
        state['_measured_query_tables'] = {}
        state['_measured_headers'] = OrderedDict()
        return state

    def _query_table(self, record):
        subject = self._measured_subjects[str(record['subject'])]
        identity = str(record['subject'])
        if identity not in self._measured_query_tables:
            for name in ('bvals', 'bvecs'):
                if sha256_file(subject[f'{name}_path']) != subject[f'{name}_sha256']:
                    raise ValueError(f'Frozen acquisition table changed: {subject[name+"_path"]}')
            bvals = np.asarray(np.loadtxt(subject['bvals_path']), dtype=np.float32).reshape(-1)
            directions = np.asarray(np.loadtxt(subject['bvecs_path']), dtype=np.float32)
            if directions.shape == (3, len(bvals)):
                directions = directions.T
            if (directions.shape != (len(bvals), 3) or len(bvals) != int(record['query_count'])
                    or not np.isfinite(bvals).all() or not np.isfinite(directions).all()):
                raise ValueError(f'Malformed measured acquisition table: {identity}')
            folder = self.root / record['shard']
            frozen_b = np.load(folder / 'bvals.npy', allow_pickle=False)
            frozen_g = np.load(folder / 'directions.npy', allow_pickle=False)
            if not np.array_equal(bvals, frozen_b) or not np.array_equal(directions, frozen_g):
                raise ValueError(f'Measured acquisition order differs from Gaussian targets: {identity}')
            stats = {name: _source_identity(subject[f'{name}_path']) for name in ('bvals', 'bvecs')}
            self._measured_query_tables[identity] = (directions.copy(), bvals, stats)
        directions, bvals, stats = self._measured_query_tables[identity]
        for name, expected in stats.items():
            _verify_stat(subject[f'{name}_path'], expected)
        return directions, bvals

    def _source_header(self, record):
        # Source stats are checked on every call even when a small header is cached.
        dwi_identity = _verify_stat(record['dwi_path'], record['source_stats']['dwi'])
        _verify_stat(record['mask_path'], record['source_stats']['mask'])
        identity = record['id']
        if identity not in self._measured_headers:
            dwi, mask = nib.load(record['dwi_path']), nib.load(record['mask_path'])
            shape = tuple(dwi.shape)
            mask_shape = tuple(mask.shape)
            if mask_shape == (*self.native_shape, 1):
                mask_shape = mask_shape[:2]
            if (shape != (*self.native_shape, int(record['query_count']))
                    or mask_shape != tuple(self.native_shape)
                    or not np.allclose(dwi.affine, mask.affine, atol=1e-5, rtol=0)):
                raise ValueError(f'Measured DWI and native mask shape/affine differ: {identity}')
            self._measured_headers[identity] = np.asarray(dwi.affine, dtype=np.float64).copy()
            if len(self._measured_headers) > 128:
                self._measured_headers.popitem(last=False)
        self._measured_headers.move_to_end(identity)
        return dwi_identity, self._measured_headers[identity]

    def _read_measured(self, record, sample):
        selected = np.asarray(sample['query_indices'], dtype=np.int64)
        if (selected.ndim != 1 or not len(selected) or np.any(selected < 0)
                or np.any(selected >= int(record['query_count']))):
            raise ValueError(f'Invalid measured query selection: {record["id"]}')
        directions, bvals = self._query_table(record)
        if (not np.array_equal(np.asarray(sample['bvals']), bvals[selected])
                or not np.array_equal(np.asarray(sample['directions']), directions[selected])):
            raise ValueError(f'Sampled Gaussian queries and measured columns differ: {record["id"]}')
        source, affine = self._source_header(record)
        cache_path = (self.measured_cache / 'train' / str(record['subject']) / f'{record["id"]}.npz'
                      if self.measured_cache is not None else None)
        if cache_path is not None and cache_path.is_file():
            measured = read_query_cache(cache_path, source, affine, self.native_shape,
                                        int(record['query_count']), selected)
        else:
            path = Path(record['dwi_path'])
            if path.name.endswith('.gz'):
                with gzip.open(path, 'rb') as stream:
                    image = nib.Nifti1Image.from_bytes(stream.read())
            else:
                image = nib.load(path)
            array = np.asarray(image.dataobj, dtype=np.float32)
            if (array.shape != (*self.native_shape, int(record['query_count']))
                    or not np.allclose(image.affine, affine, atol=1e-5, rtol=0)):
                raise ValueError(f'Measured source shape/affine changed: {record["id"]}')
            measured = np.moveaxis(array, -1, 0)[selected].copy()
        if not np.isfinite(measured).all():
            raise ValueError(f'Nonfinite measured training target: {record["id"]}')
        if _source_identity(record['dwi_path']) != source:
            raise ValueError(f'Measured source changed while reading: {record["id"]}')
        return torch.from_numpy(np.ascontiguousarray(measured, dtype=np.float32))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        record = self.records[sample['index']]
        sample['measured'] = self._read_measured(record, sample)
        return sample
