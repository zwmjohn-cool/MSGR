"""Reuse authenticated measured targets and the complete representative cohort.

Only data readers are shared with the pMF package. Flow training and sampling
remain the original Gaussian latent FM implementation.
"""
import numpy as np
import torch
from torch.nn import functional as F

from gaussian_kl_vae.corpus_data import shell_query_indices
from gaussian_latent_pmf.measured_data import MeasuredLatentGaussianDataset as _MeasuredDataset
from gaussian_latent_pmf.data import load_monitor_dataset


class MeasuredLatentGaussianDataset(_MeasuredDataset):
    """Read the original aligned inputs without unused Gaussian coefficients.

    The parent readers still authenticate measured DWI and acquisition order.
    Fitted coefficients are only needed for monitoring diagnostics; training
    already has their frozen VAE latents, so it need not copy or normalize x.
    """

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        record = self.records[int(index)]
        arrays, queries = self._subject_arrays(record['shard'])
        mask = torch.from_numpy(arrays['mask'][int(record['index'])].copy())
        h, w = self.native_shape
        pad = (0, self.padded_shape[1] - w, 0, self.padded_shape[0] - h)
        selected = np.arange(len(queries['bvals']))
        if self.queries_per_sample is not None:
            selected = shell_query_indices(queries['bvals'], self.queries_per_sample,
                                           seed=self.seed, epoch=epoch, identity=record['id'])
        sample = dict(mask=F.pad(mask, pad),
                      directions=torch.from_numpy(queries['directions'][selected].copy()),
                      bvals=torch.from_numpy(queries['bvals'][selected].copy()),
                      query_basis=torch.from_numpy(queries['query_basis'][selected].copy()),
                      query_indices=torch.from_numpy(selected.copy()),
                      id=record['id'], subject=str(record['subject']), slice=int(record['slice']),
                      index=int(index), epoch=int(epoch), empty_brain=not bool(mask.any()))
        aligned = self._latent_arrays(record['cache_shard'])
        slot = record['cache_index']
        latent = torch.from_numpy(aligned['latents'][slot].copy())
        conditions = torch.from_numpy(aligned['conditions'][slot].copy())
        sample.update(z=self.latent_normalization.normalize(latent), condition=F.pad(conditions, pad),
                      queries=torch.cat((sample['directions'], sample['bvals'][:, None]), dim=1))
        sample['measured'] = self._read_measured(record, sample)
        return sample


__all__ = ['MeasuredLatentGaussianDataset', 'load_monitor_dataset']
