"""InverseSR-style inversion with losses on acquired DWIs only.

The only reconstruction targets accepted here are ``[Q_observed,H,W]`` DWIs.
Neither clean test latents nor fitted test coefficient maps are inputs. A
renderer closure supplies the observation operator (decode, Gaussian render,
and selection of acquired encodings); a second closure can render all queries
after the solve. All intensities returned by either closure remain in raw units.

The perceptual implementation follows BioMedAI-UCSC/InverseSR commit
6ccc1fa25be26d063400d39af968e90e422f606e, project/utils/utils.py::getVggFeatures:
grayscale is repeated into RGB, then frozen TorchScript VGG16 is called with
resize_images=False and return_lpips=True. This is distinct from the AlexNet
LPIPS metric used to evaluate image quality. No added x255, ImageNet transform,
image clipping, latent penalty, or missing-direction image enters the loss.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _freeze(module: nn.Module) -> nn.Module:
    # ScriptModules deliberately disallow Module.requires_grad_(), although
    # their tensors support it. Freeze tensors individually for the official
    # TorchScript VGG16 as well as ordinary eager decoder/flow modules.
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def load_inversesr_vgg16(path, *, device="cpu", expected_sha256=None):
    """Load an explicit official feature checkpoint; never download or substitute.

    The caller records/locks the checkpoint identity. If an expected hash is
    provided, reject mismatches before loading any executable TorchScript.
    """
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("InverseSR VGG16 checkpoint SHA256 differs")
    model = _freeze(torch.jit.load(str(path), map_location=device))
    model.checkpoint_sha256 = digest
    return model


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    perceptual: torch.Tensor
    mae: torch.Tensor
    per_case_total: torch.Tensor
    per_case_perceptual: torch.Tensor
    per_case_mae: torch.Tensor


class InverseSRObservationLoss(nn.Module):
    """Perceptual + MAE with a single acquired-only affine intensity transform.

    One reconstruction may jointly contain observations from multiple shells.
    Its min and max are computed once over those permitted images only. Batched
    input ``[B,Q,H,W]`` has independent statistics and losses for every case;
    no intensity information is shared between slices/subjects. The scalar
    objective sums case losses, preserving each case's serial Adam gradient.
    Both reductions average native pixels/features and observed directions.
    ``feature_batch_size`` changes memory use, not the loss.
    """

    def __init__(self, observed: torch.Tensor, perceptual_model: nn.Module, *,
                 lambda_perc=1e4, lambda_mae=1.0, feature_batch_size=4,
                 checkpoint_features=True, cache_features_on_cpu=False,
                 allow_tf32_features=False):
        super().__init__()
        if observed.ndim not in (3, 4) or min(observed.shape) < 1:
            raise ValueError("Observed DWIs must be nonempty [Q,H,W] or [B,Q,H,W]")
        if not torch.isfinite(observed).all():
            raise ValueError("Observed DWIs must be finite")
        if (not 0 <= lambda_perc < float("inf")
                or not 0 <= lambda_mae < float("inf")
                or lambda_perc + lambda_mae <= 0):
            raise ValueError("Loss weights must be finite, nonnegative and nonzero")
        if (isinstance(feature_batch_size, bool)
                or not isinstance(feature_batch_size, int) or feature_batch_size < 1):
            raise ValueError("feature_batch_size must be a positive integer")
        self.perceptual_model = _freeze(perceptual_model)
        self.lambda_perc, self.lambda_mae = float(lambda_perc), float(lambda_mae)
        self.feature_batch_size = feature_batch_size
        self.checkpoint_features = bool(checkpoint_features)
        self.cache_features_on_cpu = bool(cache_features_on_cpu)
        self.allow_tf32_features = bool(allow_tf32_features)
        target = observed.detach().float().clone()
        self.batch_size = 1 if target.ndim == 3 else len(target)
        self.query_count = target.shape[-3]
        self.register_buffer("observed", target)
        cases = target.reshape(self.batch_size, -1)
        self.register_buffer("intensity_min", cases.amin(1).reshape(self.batch_size, 1, 1, 1))
        self.register_buffer("intensity_range", (cases.amax(1) - cases.amin(1)).clamp_min(1e-6).reshape(self.batch_size, 1, 1, 1))
        self.register_buffer("scaled_observed", self.scale(target))
        self._feature_buffer_names = []
        if self.lambda_perc:
            with torch.no_grad():
                images_flat = self.scaled_observed.reshape(-1, *target.shape[-2:])
                for index, images in enumerate(images_flat.split(feature_batch_size)):
                    features = self._features(images)
                    name = f"observed_features_{index}"
                    features = features.detach().clone()
                    if self.cache_features_on_cpu:
                        features = features.cpu()
                    self.register_buffer(name, features)
                    self._feature_buffer_names.append(name)

    def scale(self, images):
        batched = images.float().reshape(self.batch_size, self.query_count, *images.shape[-2:])
        return ((batched - self.intensity_min) / self.intensity_range).reshape_as(images)

    def _features(self, images):
        rgb = images[:, None].repeat(1, 3, 1, 1)
        # The official network divides its input by255 internally. With the
        # released InverseSR [0,1] convention, feature differences are small;
        # TF32 convolution roundoff measurably changes this loss across batch
        # shapes. Use IEEE FP32 for VGG only, restoring caller backend flags.
        precision = nullcontext()
        if images.device.type == "cuda":
            precision = torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=self.allow_tf32_features)
        with precision, torch.autocast(device_type=images.device.type, enabled=False):
            features = self.perceptual_model(rgb, resize_images=False, return_lpips=True)
        if not isinstance(features, torch.Tensor) or len(features) != len(images):
            raise ValueError("Official VGG16 must return a feature tensor with image batch first")
        return features.float()

    def forward(self, predicted: torch.Tensor) -> LossTerms:
        if predicted.shape != self.observed.shape:
            raise ValueError("Predictions must contain exactly the acquired images")
        scaled = self.scale(predicted)
        per_case_mae = (scaled - self.scaled_observed).abs().reshape(self.batch_size, -1).mean(1)
        per_case_perceptual = scaled.new_zeros((self.batch_size,))
        if self.lambda_perc:
            image_errors = []
            images_flat = scaled.reshape(-1, *scaled.shape[-2:])
            for images, name in zip(images_flat.split(self.feature_batch_size), self._feature_buffer_names):
                def feature_error(values, feature_name=name):
                    features = self._features(values)
                    target = getattr(self, feature_name).to(device=features.device)
                    return (features - target).abs().flatten(1).mean(1)

                if self.checkpoint_features and torch.is_grad_enabled() and images.requires_grad:
                    # Checkpoint through feature subtraction/reduction too.
                    # LPIPS feature vectors can be very large; retaining every
                    # feature difference would defeat VGG activation savings.
                    errors = checkpoint(feature_error, images, use_reentrant=False)
                else:
                    errors = feature_error(images)
                image_errors.append(errors)
            per_case_perceptual = torch.cat(image_errors).reshape(self.batch_size, self.query_count).mean(1)
        per_case_total = self.lambda_perc * per_case_perceptual + self.lambda_mae * per_case_mae
        return LossTerms(per_case_total.sum(), per_case_perceptual.mean(), per_case_mae.mean(),
                         per_case_total, per_case_perceptual, per_case_mae)

    def metadata(self):
        return dict(lambda_perc=self.lambda_perc, lambda_mae=self.lambda_mae,
                    intensity_min=self.intensity_min.flatten().tolist(),
                    intensity_range=self.intensity_range.flatten().tolist(),
                    scale_source="per case, all permitted acquired images only; affine; no clipping",
                    perceptual="official InverseSR TorchScript VGG16 absolute feature difference",
                    resize_images=False, return_lpips=True,
                    batch_size=self.batch_size, observed_directions=self.query_count,
                    reduction="mean within each case; SUM case objectives for independent Adam updates",
                    feature_batch_size=self.feature_batch_size,
                    cache_features_on_cpu=self.cache_features_on_cpu,
                    VGG_allow_tf32=self.allow_tf32_features,
                    VGG_precision="FP32; decoder mixed precision is independent")


@dataclass(frozen=True)
class FlowEvaluation:
    latent: torch.Tensor
    nfe: int
    steps: int
    method: str = "heun_last_euler"


def differentiable_flow_eval(model, conditions, noise, *, steps=1, context=None,
                             checkpoint_steps=True, t_eps=1e-5,
                             autocast_factory: Callable = nullcontext,
                             evaluation_counter: dict | None = None) -> FlowEvaluation:
    """Differentiate the entire noise-to-data trajectory, without changing flow.py.

    Mirrors ``gaussian_latent_fm.flow.sample_latent(method='heun')`` exactly:
    uniform t=0 noise to t=1 data, clean prediction parameterization, Heun for
    every interval except the final Euler interval. Ten intervals use nineteen
    logical forward network evaluations. Checkpoint recomputation during
    backward is additional work, not another integration interval/NFE.

    Fixed MRI tokens may be precomputed and passed as ``context``. Detaching
    those tokens is valid because the only optimization variable is ``noise``;
    intermediate latent states are never detached.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if t_eps <= 0 or 1. / steps < t_eps:
        raise ValueError("t_eps must be positive and no larger than the final interval")
    if noise.ndim < 2 or conditions is None or len(noise) != len(conditions):
        raise ValueError("Noise and fixed MRI condition batches must match")
    if noise.device != conditions.device:
        raise ValueError("Noise and MRI conditions must use the same device")
    if context is None:
        with torch.no_grad(), autocast_factory():
            context = model.encode_condition(conditions.detach()).detach()
    state = noise.float()
    nfe = 0

    def field(values, scalar_time):
        nonlocal nfe
        t = torch.full((len(values),), scalar_time, device=values.device, dtype=torch.float32)

        def clean_prediction(x, time=t):
            if evaluation_counter is not None:
                evaluation_counter["model_forward_calls"] = evaluation_counter.get("model_forward_calls", 0) + 1
            with autocast_factory():
                return model.forward_with_context(x, time, context)

        if checkpoint_steps and torch.is_grad_enabled() and values.requires_grad:
            clean = checkpoint(clean_prediction, values, use_reentrant=False)
        else:
            clean = clean_prediction(values)
        nfe += 1
        divisor = (1 - t).clamp_min(t_eps).reshape(len(values), *([1] * (values.ndim - 1)))
        return (clean.float() - values.float()) / divisor

    for index in range(steps):
        t, next_t = index / steps, (index + 1) / steps
        dt = next_t - t
        first = field(state, t)
        proposed = state + dt * first
        if index < steps - 1:
            second = field(proposed, next_t)
            state = state + (.5 * dt) * (first + second)
        else:
            state = proposed
    return FlowEvaluation(state, nfe=nfe, steps=steps)


@dataclass(frozen=True)
class InversionConfig:
    mode: str = "latent"
    updates: int = 600
    learning_rate: float = .07
    flow_steps: int = 1
    source_seed: int = 42
    source_seeds: tuple[int, ...] | None = None
    noise_scale: float = 1.0
    checkpoint_flow: bool = True

    def __post_init__(self):
        if self.mode not in ("latent", "flow"):
            raise ValueError("mode must be 'latent' or 'flow'")
        for name in ("updates", "flow_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not 0 < self.learning_rate < float("inf")
                or not 0 < self.noise_scale < float("inf")):
            raise ValueError("Learning rate and source-noise scale must be finite and positive")


@dataclass(frozen=True)
class InversionResult:
    latent: torch.Tensor
    optimized_input: torch.Tensor
    initial_input: torch.Tensor
    observed_prediction: torch.Tensor
    prediction: torch.Tensor | None
    history: list[dict]
    metadata: dict


def solve_inversion(loss: InverseSRObservationLoss, render_observed: Callable, *,
                    latent_shape, config: InversionConfig = InversionConfig(),
                    flow_model=None, conditions=None,
                    frozen_modules: Iterable[nn.Module] = (),
                    final_render: Callable | None = None,
                    callback: Callable | None = None,
                    checkpoint_callback: Callable | None = None,
                    checkpoint_updates: Iterable[int] = (),
                    autocast_factory: Callable = nullcontext) -> InversionResult:
    """Optimize standardized zero latents or source noise against sparse DWIs.

    ``render_observed(z)`` receives a standardized clean latent and must return
    raw native acquired images. ``final_render(z)`` optionally renders the
    complete query table after optimization. Neither closure may depend on
    hidden target DWIs. Pass every pretrained decoder/renderer in
    ``frozen_modules``; only one new latent/noise tensor is passed to Adam.
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError("Inversion requires gradients; do not wrap it in no_grad/inference_mode")
    shape = tuple(latent_shape)
    if len(shape) < 2 or any(int(n) != n or n < 1 for n in shape) or shape[0] != loss.batch_size:
        raise ValueError("Use a positive standardized latent shape matching the observed batch")
    device = loss.observed.device
    checkpoint_updates = set(checkpoint_updates)
    if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= config.updates
           for n in checkpoint_updates):
        raise ValueError("Checkpoint updates must be positive integers within the solve budget")
    for module in frozen_modules:
        _freeze(module)
    _freeze(loss.perceptual_model)
    context = None
    seeds = None
    if config.mode == "flow":
        if flow_model is None or conditions is None:
            raise ValueError("Flow inversion requires a frozen flow and fixed MRI conditions")
        _freeze(flow_model)
        fixed_conditions = conditions.detach().to(device=device).clone()
        if len(fixed_conditions) != shape[0]:
            raise ValueError("Fixed MRI batch must match the observed cases")
        with torch.no_grad(), autocast_factory():
            context = flow_model.encode_condition(fixed_conditions).detach()
        seeds = tuple(config.source_seeds or (config.source_seed + index for index in range(shape[0])))
        if len(seeds) != shape[0]:
            raise ValueError("source_seeds must contain exactly one fixed seed per case")
        initial = torch.cat([torch.randn((1, *shape[1:]), device=device, dtype=torch.float32,
                             generator=torch.Generator(device=device).manual_seed(seed))
                             for seed in seeds]) * config.noise_scale
    else:
        if flow_model is not None or conditions is not None:
            raise ValueError("Direct latent inversion must not receive MRI or a conditional flow")
        initial = torch.zeros(shape, device=device, dtype=torch.float32)
    variable = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([variable], lr=config.learning_rate)
    history = []
    started = perf_counter()
    logical_nfe = 0
    evaluation_counter = {"model_forward_calls": 0}

    def clean_latent():
        nonlocal logical_nfe
        if config.mode == "latent":
            return variable
        result = differentiable_flow_eval(
            flow_model, fixed_conditions, variable, steps=config.flow_steps,
            context=context, checkpoint_steps=config.checkpoint_flow,
            autocast_factory=autocast_factory, evaluation_counter=evaluation_counter)
        logical_nfe += result.nfe
        return result.latent

    for index in range(config.updates):
        optimizer.zero_grad(set_to_none=True)
        latent = clean_latent()
        with autocast_factory():
            prediction = render_observed(latent)
        terms = loss(prediction.float())
        if not torch.isfinite(terms.total):
            raise FloatingPointError(f"Nonfinite observed loss at update {index}")
        terms.total.backward()
        if variable.grad is None or not torch.isfinite(variable.grad).all():
            raise FloatingPointError(f"Missing/nonfinite optimized-input gradient at update {index}")
        row = dict(update=index, state="before_optimizer_update",
                   loss=float(terms.total.detach()), perceptual=float(terms.perceptual.detach()),
                   mae=float(terms.mae.detach()), grad_norm=float(variable.grad.norm()),
                   per_case_loss=terms.per_case_total.detach().tolist(),
                   per_case_perceptual=terms.per_case_perceptual.detach().tolist(),
                   per_case_mae=terms.per_case_mae.detach().tolist(),
                   elapsed_seconds=perf_counter() - started)
        optimizer.step()
        history.append(row)
        if callback is not None:
            callback(dict(row))
        if checkpoint_callback is not None and index + 1 in checkpoint_updates:
            with torch.no_grad():
                checkpoint_callback(index + 1, variable.detach().clone(), optimizer.state_dict())
        # Do not retain any previous trajectory/decoder/perceptual graph.
        del latent, prediction, terms

    # Re-evaluate the final UPDATED variable. The image from the last loss is
    # one Adam step stale and must never be returned as the final reconstruction.
    with torch.no_grad():
        latent = clean_latent()
        with autocast_factory():
            observed_prediction = render_observed(latent).float()
            all_prediction = None if final_render is None else final_render(latent).float()
        terms = loss(observed_prediction)
        if not torch.isfinite(terms.total) or (all_prediction is not None
                                               and not torch.isfinite(all_prediction).all()):
            raise FloatingPointError("Nonfinite final reconstruction")
        final_row = dict(update=config.updates, state="final_updated_input",
                         loss=float(terms.total), perceptual=float(terms.perceptual),
                         mae=float(terms.mae),
                         per_case_loss=terms.per_case_total.tolist(),
                         per_case_perceptual=terms.per_case_perceptual.tolist(),
                         per_case_mae=terms.per_case_mae.tolist(),
                         elapsed_seconds=perf_counter() - started)
        history.append(final_row)
    metadata = dict(mode="Latent Inversion" if config.mode == "latent" else "Flow Inversion",
                    updates=config.updates, learning_rate=config.learning_rate,
                    source_seed=config.source_seed if config.mode == "flow" else None,
                    source_seeds=list(seeds) if seeds is not None else None,
                    batch_size=loss.batch_size,
                    initialization="standardized zero" if config.mode == "latent" else "fixed Gaussian source noise",
                    input_scope="sparse DWI+B0" if config.mode == "latent" else "sparse DWI+B0+T1+T2 (MRI-assisted)",
                    flow_steps=config.flow_steps if config.mode == "flow" else 0,
                    flow_nfe_per_trajectory=2 * config.flow_steps - 1 if config.mode == "flow" else 0,
                    logical_flow_forward_nfe_including_final=logical_nfe,
                    flow_model_forward_calls_including_checkpoint_recomputation=evaluation_counter["model_forward_calls"],
                    flow_backward_trajectories=config.updates if config.mode == "flow" else 0,
                    checkpoint_backward_recomputation_additional=bool(config.checkpoint_flow and config.mode == "flow"),
                    frozen_networks=True, fixed_MRI_conditions=config.mode == "flow",
                    final_state_recomputed=True, loss=loss.metadata())
    return InversionResult(latent.detach().clone(), variable.detach().clone(), initial,
                           observed_prediction.detach(),
                           None if all_prediction is None else all_prediction.detach(),
                           history, metadata)
