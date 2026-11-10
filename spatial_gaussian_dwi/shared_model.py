"""Native Gaussian amplitudes with a single explicit cross-subject decoder.

The public basis is stored once. Subject archives contain only their active
amplitudes, primitive mask, shape, and an authenticated relative reference.
No learned decoder is duplicated in a subject archive. Fixed native centers
and full-tail isotropic spatial Gaussians retain the local model convention.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .local_operator import fit_latent
from .model import field_points, gaussian_convolve
from .radial_sh import cardinal_basis


BASIS_FORMAT = "spatial_gaussian_dwi.shared_cardinal_basis.v1"
FORMAT = "spatial_gaussian_dwi.shared_cardinal_subject.v1"


def _config(config):
    source = dict(config or {})
    result = {"width": float(source.get("width", 2.)),
              "sigma": float(source.get("sigma", .7))}
    if not np.isfinite(list(result.values())).all() or min(result.values()) <= 0:
        raise ValueError("Spatial sigma and radial width must be finite and positive")
    return result


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SharedBasis:
    """An explicit [84,R] physical decoder, common to every subject.

    The content ID hashes canonical FP32 coefficients and the numerical
    decoder configuration, independent of file location and descriptive
    metadata. Rendering preserves tensor gradients. Persisted IDs refer to
    FP32 storage precision; changing B during training requires saving a new
    public basis and subject references before distributing the model.
    """

    def __init__(self, matrix, config=None, metadata=None):
        self.matrix = torch.as_tensor(matrix)
        if (self.matrix.dtype not in (torch.float32, torch.float64)
                or self.matrix.ndim != 2 or self.matrix.shape[0] != 84
                or not 1 <= self.matrix.shape[1] <= 84
                or not torch.isfinite(self.matrix).all()):
            raise ValueError("Shared basis must be finite float32/float64 [84,R], 1<=R<=84")
        self.config = _config(config)
        self.metadata = dict(metadata or {})
        self.path = None

    @property
    def rank(self):
        return self.matrix.shape[1]

    @property
    def parameter_count(self):
        return self.matrix.numel()

    @property
    def content_id(self):
        array = np.asarray(self.matrix.detach().cpu().numpy(), dtype="<f4", order="C")
        descriptor = dict(format=BASIS_FORMAT, shape=list(array.shape), config=self.config,
                          dtype="little-endian float32", dictionary="radial_sh.cardinal_basis84")
        digest = hashlib.sha256(_json(descriptor).encode("utf8"))
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path):
        path = Path(path).resolve()
        header = dict(format=BASIS_FORMAT, config=self.config, metadata=self.metadata,
                      content_id=self.content_id, coefficient_dimension=84, rank=self.rank)
        _write_npz(path, matrix=self.matrix.detach().cpu().numpy().astype(np.float32),
                   metadata=np.array(_json(header)))
        self.path = path
        return path

    @classmethod
    def load(cls, path, *, device="cpu", dtype=torch.float32):
        path = Path(path).resolve()
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(archive["metadata"].item())
            if header["format"] != BASIS_FORMAT:
                raise ValueError("Unsupported public basis format")
            matrix = archive["matrix"].copy()
            if matrix.dtype != np.float32:
                raise ValueError("Public basis storage must use FP32")
        result = cls(torch.as_tensor(matrix, device=device, dtype=dtype),
                     header["config"], header.get("metadata"))
        if (result.content_id != header["content_id"] or result.rank != header["rank"]
                or header["coefficient_dimension"] != 84):
            raise ValueError("Public basis content hash or dimensions do not match")
        result.path = path
        return result

    def to(self, *, device=None, dtype=None):
        result = SharedBasis(self.matrix.to(device=device, dtype=dtype), self.config, self.metadata)
        result.path = self.path
        return result


class SharedGaussianDWI:
    """A native [H,W,R] amplitude map and an explicit shared public basis.

    Shared B permits G(z B.T)=G(z) B.T, so rendering convolves only R channels.
    Background values are ignored, even if NaN. Gaussian output tails remain
    visible outside the primitive mask. No intensity normalization, B0 factor,
    nonlinear network, quantization, or fitted-signal cache is applied.
    """

    def __init__(self, latent, mask, basis, metadata=None):
        if not isinstance(basis, SharedBasis):
            raise TypeError("basis must be an explicit SharedBasis")
        self.latent = torch.as_tensor(latent)
        if self.latent.dtype not in (torch.float32, torch.float64):
            raise ValueError("latent must be float32/float64 [H,W,R]")
        support = np.asarray(mask)
        if (support.ndim != 2 or min(support.shape) < 1 or not support.any()
                or not np.all((support == 0) | (support == 1))):
            raise ValueError("Require a nonempty binary native primitive mask")
        self.mask = support.astype(bool, copy=True)
        if self.latent.shape != (*self.mask.shape, basis.rank):
            raise ValueError("Latent spatial shape/rank disagrees with mask or public basis")
        active = torch.as_tensor(self.mask, device=self.latent.device)
        if not torch.isfinite(self.latent[active]).all():
            raise ValueError("Active amplitudes must be finite")
        self.basis = basis.to(device=self.latent.device, dtype=self.latent.dtype)
        self.config = self.basis.config
        self.metadata = dict(metadata or {})

    def _amplitudes(self):
        active = torch.as_tensor(self.mask, device=self.latent.device)
        return torch.where(active[..., None], self.latent, 0.)

    def _query(self, directions, bvals):
        phi = cardinal_basis(torch.as_tensor(directions, device=self.latent.device,
                                             dtype=self.latent.dtype), bvals,
                             width=self.config["width"])
        return phi @ self.basis.matrix

    def latent_field(self):
        return gaussian_convolve(self._amplitudes(), self.config["sigma"])

    def coefficient_field(self):
        return self.latent_field() @ self.basis.matrix.T

    def evaluate_grid(self, directions, bvals):
        return self.latent_field() @ self._query(directions, bvals).T

    def prepare_fast_renderer(self, *, compile=True):
        """Create a reusable, differentiable full-tail native-grid renderer.

        Validate an acquisition table once with ``renderer.validate_queries``;
        then call ``renderer(self.latent, g, b)``. Current amplitudes and query
        values are read on every call. Only fixed geometry/dictionary constants
        are retained. First-call optional compilation is setup, not rendering.
        The original ``evaluate_grid`` remains the strict reference path.
        """
        from .fast_rendering import FastSharedGaussianRenderer

        renderer = FastSharedGaussianRenderer(self.basis, self.mask)
        if compile:
            renderer.enable_compilation()
        return renderer

    def evaluate_points(self, xy, directions, bvals, *, chunk_size=128):
        field = field_points(self._amplitudes(), self.mask, xy, self.config["sigma"],
                             chunk_size=chunk_size)
        return field @ self._query(directions, bvals).T

    @property
    def parameter_count(self):
        """Subject-specific parameters; public B is counted separately."""
        return int(self.mask.sum()) * self.basis.rank

    @property
    def shared_parameter_count(self):
        return self.basis.parameter_count

    def storage(self, *, cohort_size=1):
        if type(cohort_size) is not int or cohort_size < 1:
            raise ValueError("cohort_size must be a positive integer")
        mask_bytes = int(np.packbits(self.mask.ravel()).nbytes)
        reference = 45 * int(self.mask.sum())
        return dict(subject_parameters=self.parameter_count,
                    shared_parameters=self.shared_parameter_count,
                    single_subject_including_shared=self.parameter_count + self.shared_parameter_count,
                    amortized_parameters=self.parameter_count + self.shared_parameter_count / cohort_size,
                    dense_subject_parameters=self.latent.numel(),
                    active_gaussians=int(self.mask.sum()), rank=self.basis.rank,
                    fp32_subject_parameter_bytes=self.parameter_count * 4,
                    fp32_shared_parameter_bytes=self.shared_parameter_count * 4,
                    mask_packbits_bytes=mask_bytes, sh_active_parameters=reference,
                    subject_parameter_ratio=self.parameter_count / reference,
                    single_subject_including_shared_ratio=(self.parameter_count + self.shared_parameter_count) / reference,
                    amortized_parameter_ratio=(self.parameter_count + self.shared_parameter_count / cohort_size) / reference)

    def save(self, path, *, basis_path=None):
        """Reference an existing verified public basis; never embed B here."""
        path = Path(path).resolve()
        source = self.basis.path if basis_path is None else Path(basis_path).resolve()
        if source is None:
            raise ValueError("Save the public basis first or provide basis_path")
        public = SharedBasis.load(source)
        if public.content_id != self.basis.content_id:
            raise ValueError("Saved public basis differs from this model's basis")
        header = dict(format=FORMAT, metadata=self.metadata, rank=self.basis.rank,
                      basis_path=os.path.relpath(source, path.parent),
                      basis_content_id=public.content_id,
                      signal_units="restored raw MRI a.u.", signal_normalization="none",
                      multiply_b0=False, spatial_convention="native row,column; fixed centers; full tails")
        _write_npz(path, latent=self.latent.detach().cpu().numpy()[self.mask].astype(np.float32),
                   mask_packbits=np.packbits(self.mask.ravel()),
                   spatial_shape=np.asarray(self.mask.shape, dtype=np.int32),
                   metadata=np.array(_json(header)))
        return path

    @classmethod
    def load(cls, path, *, basis=None, device="cpu", dtype=torch.float32):
        path = Path(path).resolve()
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(archive["metadata"].item())
            if header["format"] != FORMAT:
                raise ValueError("Unsupported shared Gaussian archive")
            shape_values = archive["spatial_shape"]
            if shape_values.shape != (2,) or shape_values.dtype.kind not in "iu":
                raise ValueError("Invalid saved spatial shape")
            shape = tuple(int(value) for value in shape_values)
            if min(shape) < 1:
                raise ValueError("Invalid saved spatial shape")
            packed = archive["mask_packbits"]
            if (packed.dtype != np.uint8 or packed.ndim != 1
                    or packed.size != (int(np.prod(shape)) + 7) // 8):
                raise ValueError("Invalid packed primitive mask")
            mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
            active = archive["latent"].copy()
            if active.dtype != np.float32 or active.shape != (int(mask.sum()), header["rank"]):
                raise ValueError("Saved FP32 amplitudes do not match support or rank")
        if basis is None:
            reference = Path(header["basis_path"])
            if reference.is_absolute():
                raise ValueError("Subject archives require a relative public basis path")
            basis = SharedBasis.load(path.parent / reference, device=device, dtype=dtype)
        if not isinstance(basis, SharedBasis) or basis.content_id != header["basis_content_id"]:
            raise ValueError("Subject/public basis content hash mismatch")
        latent = np.zeros((*shape, active.shape[-1]), dtype=np.float32)
        latent[mask] = active
        return cls(torch.as_tensor(latent, device=device, dtype=dtype), mask, basis,
                   header.get("metadata"))


@torch.no_grad()
def fit_coefficients(teacher, directions_train, bvals_train, mask, basis, config=None, *, device="cpu"):
    """Fit fixed B to an externally prepared TRAINING-ONLY [H,W,84] teacher.

    The caller must restrict acquisition rows before constructing the teacher
    or public basis. No held signals or global-acquisition table is accepted.
    For ridge=0, QR of Phi_train B separates q-space and spatial least squares
    exactly; an audited Gaussian PCG then handles only R channels. Positive
    ridge uses the full physical-amplitude coupled operator to retain its
    intended meaning. Fits are FP64; the returned model is explicitly FP32.
    """
    began = perf_counter()
    cfg = dict(config or {})
    public = basis.to(device=device, dtype=torch.float64)
    for key in ("width", "sigma"):
        if key in cfg and float(cfg[key]) != public.config[key]:
            raise ValueError(f"Fit {key} disagrees with public basis")
    target = torch.as_tensor(teacher, device=device, dtype=torch.float64)
    support = np.asarray(mask)
    if (support.ndim != 2 or not support.any()
            or not np.all((support == 0) | (support == 1))):
        raise ValueError("Require a nonempty binary primitive mask")
    if target.shape != (*support.shape, 84):
        raise ValueError("Teacher must have shape [H,W,84]")
    selected = torch.as_tensor(support.astype(bool), device=device)
    if not torch.isfinite(target[selected]).all():
        raise ValueError("Selected training teacher values must be finite")
    target = torch.where(selected[..., None], target, 0.)
    query = cardinal_basis(torch.as_tensor(directions_train, device=device, dtype=torch.float64),
                           bvals_train, width=public.config["width"])
    groups = torch.where(selected, 0, -1)
    ridge = float(cfg.get("ridge", 0.))
    options = dict(mask=selected, ridge=ridge, tolerance=float(cfg.get("cg_tolerance", 1e-7)),
                   max_iterations=int(cfg.get("cg_iterations", 1000)),
                   require_convergence=bool(cfg.get("require_convergence", True)))
    design = query @ public.matrix
    singular = torch.linalg.svdvals(design)
    if len(singular) < public.rank or singular[-1] <= singular[0] * 1e-12:
        raise ValueError("Training queries do not identify every public-basis channel")
    if ridge == 0.:
        orthogonal, triangular = torch.linalg.qr(design, mode="reduced")
        projected = target @ (query.T @ orthogonal)
        identity = torch.eye(public.rank, dtype=torch.float64, device=device)[None]
        whitened, spatial = fit_latent(projected, identity, groups, public.config["sigma"],
                                       target_space="signal", **options)
        latent = torch.linalg.solve_triangular(triangular, whitened.reshape(-1, public.rank).T,
                                               upper=True).T.reshape(*support.shape, public.rank)
        algorithm = "exact shared-query QR separation; rank-channel spatial Gaussian PCG"
    else:
        latent, spatial = fit_latent(target, public.matrix[None], groups, public.config["sigma"],
                                    query, target_space="coefficients", **options)
        algorithm = "physical-amplitude ridge; full shared-decoder Gaussian PCG"
    details = dict(spatial=spatial, algorithm=algorithm, raw_intensity=True,
                   b0_normalization=False, public_basis_updated=False,
                   rank=public.rank, shared_basis_content_id=public.content_id,
                   training_queries=len(query), query_condition_number=float(singular[0] / singular[-1]),
                   subject_parameters=int(selected.sum()) * public.rank,
                   shared_parameters=public.parameter_count, fit_seconds=perf_counter()-began)
    return SharedGaussianDWI(latent.float(), support.astype(bool), public.to(dtype=torch.float32), details), details
