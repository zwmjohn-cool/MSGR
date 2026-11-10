"""Memory-bounded derivative of the frozen InverseSR observation objective.

The original loss implementation and experiment sources remain unchanged. Each
VGG image chunk is evaluated and differentiated once, its graph is released,
and the assembled image derivative is passed through the original renderer,
decoder and flow graph. VGG chunk boundaries and all FP32 reductions match the
original loss. This optimization supports first-order inversion only.
"""
from __future__ import annotations

import torch

from gaussian_asr.inversion import InverseSRObservationLoss, LossTerms


class _AttachPredictionGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predicted, value, gradient):
        ctx.save_for_backward(gradient)
        return value.clone()

    @staticmethod
    def backward(ctx, incoming):
        (gradient,) = ctx.saved_tensors
        return gradient * incoming, None, None


def value_and_prediction_gradient(loss, predicted):
    """Return original diagnostics and d(sum of case objectives)/d(predicted).

The gradient entering each per-image feature error follows the original
sum -> lambda multiplication -> per-case query mean sequence. The separate
MAE branch uses its original full-case reduction. Their FP32 derivatives add
before the acquired-only affine scaling derivative is applied.
"""
    if predicted.shape != loss.observed.shape:
        raise ValueError("Predictions must contain exactly the acquired images")
    with torch.enable_grad():
        scaled = loss.scale(predicted.detach()).detach().requires_grad_(True)
        per_case_mae = (scaled - loss.scaled_observed).abs().reshape(loss.batch_size, -1).mean(1)
        mae_value = per_case_mae.detach()
        # Preserve PyTorch's reduction/multiplication backward order, including
        # zero MAE weight. No analytic sign shortcut changes rounding at ties.
        scaled_gradient, = torch.autograd.grad((loss.lambda_mae * per_case_mae).sum(), scaled)
        images_flat = scaled.detach().reshape(-1, *scaled.shape[-2:])
        gradient_flat = scaled_gradient.reshape_as(images_flat)
        errors = []
        if loss.lambda_perc:
            for offset, name in enumerate(loss._feature_buffer_names):
                start = offset * loss.feature_batch_size
                images = images_flat[start:start + loss.feature_batch_size].detach().requires_grad_(True)
                features = loss._features(images)
                target = getattr(loss, name).to(device=features.device)
                image_errors = (features - target).abs().flatten(1).mean(1)
                # This is exactly the backward of per_case_total.sum(), then
                # multiplication by lambda_perc, then mean over Q observations.
                upstream = torch.ones_like(image_errors) * loss.lambda_perc
                upstream = upstream / loss.query_count
                image_gradient, = torch.autograd.grad(image_errors, images, grad_outputs=upstream)
                gradient_flat[start:start + len(images)].add_(image_gradient)
                errors.append(image_errors.detach())
                del images, features, target, image_errors, upstream, image_gradient
            per_case_perceptual = torch.cat(errors).reshape(loss.batch_size, loss.query_count).mean(1)
        else:
            per_case_perceptual = scaled.new_zeros((loss.batch_size,))
        per_case_total = loss.lambda_perc * per_case_perceptual + loss.lambda_mae * mae_value
        terms = LossTerms(per_case_total.sum(), per_case_perceptual.mean(), mae_value.mean(),
                          per_case_total, per_case_perceptual, mae_value)
        gradient = (scaled_gradient.reshape(loss.batch_size, loss.query_count, *scaled.shape[-2:])
                    / loss.intensity_range).reshape_as(predicted)
        return terms, gradient.detach()


class StreamingObservationLoss(InverseSRObservationLoss):
    """Drop-in first-order objective preserving original final-loss evaluation."""

    def forward(self, predicted):
        if not torch.is_grad_enabled() or not predicted.requires_grad:
            return super().forward(predicted)
        terms, gradient = value_and_prediction_gradient(self, predicted)
        total = _AttachPredictionGradient.apply(predicted, terms.total, gradient)
        return LossTerms(total, terms.perceptual, terms.mae, terms.per_case_total,
                         terms.per_case_perceptual, terms.per_case_mae)

    def metadata(self):
        return dict(super().metadata(),
                    gradient_execution="streamed detached prediction chunks; original FP32 VGG chunk size",
                    VGG_checkpoint_recomputation=False, gradient_order=1)


def install_streaming_observation_loss():
    """Opt in from an independently audited execution wrapper only."""
    import gaussian_asr.inversion as inversion
    inversion.InverseSRObservationLoss = StreamingObservationLoss
