"""JiT time convention and ODE integration with the requested clean-latent L1.

JiT/denoiser.py uses t=0 at noise, t=1 at data; interpolation is
z_t=t*x+(1-t)*noise and velocity=(x_pred-z_t)/(1-t). Its official objective
is velocity MSE. This experiment deliberately uses unweighted clean-latent
L1 instead, as requested, plus a separately supplied differentiable image L1.
The native JiT defaults P_mean=-0.8 and P_std=0.8 give logit-normal times.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


def sample_time(n, *, device, generator=None, mean=-.8, std=.8):
    if n <= 0 or std <= 0:
        raise ValueError("Batch size and logit-normal std must be positive")
    return (torch.randn(n, device=device, generator=generator) * std + mean).sigmoid()


def interpolate(clean, noise, t):
    if clean.shape != noise.shape or t.ndim != 1 or len(t) != len(clean):
        raise ValueError("Clean/noise shapes must match, with one time per item")
    time = t.reshape(len(t), *([1] * (clean.ndim - 1)))
    return time * clean + (1 - time) * noise


def x_loss(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("x-pred and clean latent must have the same shape")
    return (prediction.float() - target.float()).abs().mean()


def velocity(prediction, noisy, t, *, t_eps=1e-5):
    if prediction.shape != noisy.shape or t.ndim != 1 or len(t) != len(noisy):
        raise ValueError("Invalid velocity input shapes")
    if t_eps <= 0:
        raise ValueError("t_eps must be positive")
    denominator = (1 - t.float()).clamp_min(t_eps).reshape(len(t), *([1] * (noisy.ndim - 1)))
    return (prediction.float() - noisy.float()) / denominator


@dataclass(frozen=True)
class SampleResult:
    latent: torch.Tensor
    nfe: int
    steps: int
    method: str


@torch.no_grad()
def sample_latent(model, conditions, *, steps=50, method="heun", generator=None,
                  noise=None, noise_scale=1., t_eps=1e-5):
    """Integrate noise -> clean latent, caching condition tokens exactly once.

    Heun uses Euler for the last interval, matching JiT and never querying the
    singular t=1 velocity. Thus Euler NFE=steps, Heun NFE=2*steps-1. The default
    denominator floor differs deliberately from JiT's training-oriented .05:
    1e-5 does not alter any interval in normal 1/10/50-step evaluation. The
    final Euler update is exactly the last clean prediction whenever dt>=eps.
    NFE counts U-Net calls, not the single reusable condition encoder call.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if method not in ("euler", "heun") or noise_scale <= 0 or t_eps <= 0:
        raise ValueError("Invalid sampling method/noise scale/denominator floor")
    if 1. / steps < t_eps:
        raise ValueError("t_eps must not exceed the final integration interval")
    if noise is None:
        shape = (len(conditions), model.config.latent_channels, *model.config.latent_shape)
        state = torch.randn(shape, device=conditions.device, dtype=torch.float32,
                             generator=generator) * noise_scale
    else:
        if len(noise) != len(conditions) or noise.device != conditions.device:
            raise ValueError("Initial noise must match condition batch and device")
        state = noise.float().clone()
    context = model.encode_condition(conditions)
    nfe = 0

    def field(state, scalar_time):
        nonlocal nfe
        t = torch.full((len(state),), scalar_time, device=state.device, dtype=torch.float32)
        predicted_clean = model.forward_with_context(state, t, context)
        nfe += 1
        return velocity(predicted_clean, state, t, t_eps=t_eps)

    for index in range(steps):
        t, t_next = index / steps, (index + 1) / steps
        dt = t_next - t
        first = field(state, t)
        proposed = state + dt * first
        if method == "heun" and index < steps - 1:
            second = field(proposed, t_next)
            state = state + (.5 * dt) * (first + second)
        else:
            state = proposed
    return SampleResult(latent=state, nfe=nfe, steps=steps, method=method)
