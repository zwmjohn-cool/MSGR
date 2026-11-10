# MSGR

Method implementation of **Multi-Shell Gaussian Representation for MRI-to-DWI Synthesis and Angular Super-Resolution**.

The pipeline fits a continuous Gaussian diffusion field, compresses its coefficients with a KL autoencoder, and learns an MRI-conditioned latent flow. The frozen decoder and renderer also support reconstruction from sparse diffusion measurements.

## Method

1. **Shared Gaussian field.** Fixed native-grid centers and full-tail spatial Gaussians with `sigma = 0.7` pixels carry 22 coefficients per active center. A shared `84 × 22` basis maps these coefficients to three cardinal radial functions times 28 even spherical harmonics of orders 0, 2, 4, and 6. Queries use gradient directions and positive b-values; outputs retain raw MRI intensity units without B0 multiplication or division.
2. **KL autoencoder.** A 22-channel coefficient canvas is padded from `145 × 174` to `160 × 176` and compressed to an `8 × 40 × 44` latent. Training combines active-center coefficient MSE, rendered-DWI MSE, a rendered-image PatchGAN term, and KL regularization. Reconstruction targets and discriminator real images in this stage come from the fitted Gaussian field.
3. **Conditional flow.** T1, T2, and B0 images condition a latent U-Net through cross-attention. The network predicts the clean standardized latent from `z_t = t*z + (1-t)*noise`, with `t = sigmoid(N(-0.8, 0.8²))`. Training uses latent L1 plus rendered L1 against **measured DWI**, keeping the VAE, Gaussian basis, and normalization statistics frozen.
4. **Angular super-resolution (ASR).** Latent Inversion optimizes the decoder's standardized latent from zero. Flow Inversion optimizes input noise through the frozen MRI-conditioned flow. Both fit acquired directions jointly using a fixed VGG16 perceptual loss and pixel MAE.

The VAE rendered loss uses target-brain RMS scaling, brain MSE, and `0.1 × background MSE`; the main flow image loss uses the same scaling convention with L1. Default VAE weights are coefficient `1`, image `1`, adversarial `0.01` with warmup, and per-scalar KL `1e-4`. The two outer flow loss weights are both `1`.

## Repository layout

| Directory | Purpose |
| --- | --- |
| [spatial_gaussian_dwi/](spatial_gaussian_dwi/) | Shared basis learning, teacher preparation, coefficient fitting, and differentiable rendering |
| [gaussian_kl_vae/](gaussian_kl_vae/) | Corpus fitting, training-only normalization, KL autoencoder, losses, and codec |
| [gaussian_latent_fm/](gaussian_latent_fm/) | Conditional backbone, latent preparation, samplers, and cached-case inference |
| [gaussian_latent_fm_measured/](gaussian_latent_fm_measured/) | Main flow training with measured-DWI supervision |
| [gaussian_latent_pmf/](gaussian_latent_pmf/) | Reused data readers only; no pMF generator |
| [gaussian_asr/](gaussian_asr/) | Frozen runtime, observed-only losses, and latent/flow inversion |
| [stable-diffusion/](stable-diffusion/) | Required autoencoder and U-Net architecture sources |
| [configs/](configs/) | VAE and measured-DWI flow configurations |

## Installation

Use Python 3.11 or newer. Training requires a CUDA-enabled PyTorch installation; install the PyTorch build suitable for your device before installing the remaining requirements. The training entry points use Unix file locking.

```bash
git clone https://github.com/zwmjohn-cool/MSGR.git
cd MSGR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from the repository root. Required Stable Diffusion architecture sources are included; the networks are trained from scratch, and no RGB pretrained weights are bundled.

## Data and model assets

This repository contains source code and configurations. Data, fitted Gaussian labels, the learned basis, normalization statistics, model checkpoints, and VGG16 weights must be supplied separately.

The corpus reader expects processed subject directories such as:

```text
TRAIN_ROOT/
  RAW_INTENSITY_MIGRATION.json
  SUBJECT_processed/
    bvals
    bvecs
    SUBJECT_processed_SLICE_dwi.nii.gz
    SUBJECT_processed_SLICE_mask.nii.gz
    SUBJECT_processed_SLICE_t1.nii.gz
    SUBJECT_processed_SLICE_t2.nii.gz
    SUBJECT_processed_SLICE_b0.nii.gz
```

The external corpus uses the same processed-file layout. For flow monitoring, `rest_root` must also contain a raw numeric `SUBJECT/` directory for every `SUBJECT_processed/` (or `SUBJECT_slices/`) directory; the loader checks that the two subject inventories match. Subject IDs in this workflow are numeric.

The completion marker verifies the original raw-intensity migration; the repository does not perform that migration. Acquisitions must contain valid positive b-values paired with their directions. MRI conditions must already be aligned to the native DWI grid.

The supplied workflow retains the original cohort assumptions. In particular, the main flow trainer checks for **800 training subjects and 87,471 slices**, and requires an external monitoring cache. Adapting the workflow to another cohort requires updating those checks and preparing matching manifests. Basis learning and normalization must use training data only.

For decoding, `GaussianKLVAECodec.from_run(..., weights="ema")` needs a VAE checkpoint and its matching `manifest.json`, `basis.npz`, and `normalization.npz`. The codec finds these assets in the recorded corpus cache, or in the run's `cache/` directory. It verifies their identities and hashes; decoding alone does not require all training coefficient arrays.

## Training workflow

These commands assume the native data and required external assets are already available. Replace the example paths with your own.

### 1. Fit Gaussian coefficient maps

Supply a training-only shared R22 basis as `assets/basis.npz`. The basis-learning API is [prepare_teacher / learn_common_basis](spatial_gaussian_dwi/shared_fitting.py); the corpus commands below keep that basis fixed.

```bash
python -m gaussian_kl_vae.corpus manifest \
  --out outputs/vae/cache \
  --train-root /path/to/train \
  --external-root /path/to/external \
  --basis assets/basis.npz

python -m gaussian_kl_vae.corpus fit \
  --out outputs/vae/cache --device cuda:0

python -m gaussian_kl_vae.corpus finalize \
  --out outputs/vae/cache
```

The fitting command also exports fitted `*_dwi.npz` labels beside the input slices. For multiple fitting workers, assign distinct `--worker-index` values with a shared `--worker-count`.

### 2. Train the VAE and export posterior means

```bash
python -m gaussian_kl_vae.corpus_training \
  --config configs/gaussian_kl_vae_r22_full.json \
  --output outputs/vae --cache outputs/vae/cache
```

The supplied configuration exports EMA posterior-mean latents after training completes. To export them separately from an existing run:

```bash
python -m gaussian_kl_vae.corpus_training \
  --config configs/gaussian_kl_vae_r22_full.json \
  --output outputs/vae --cache outputs/vae/cache --export-latents
```

Add `--resume` to the training command to restore an existing run. Resume checks the saved source, configuration, and data identities.

### 3. Prepare the latent and MRI cache

```bash
python -m gaussian_latent_fm.prepare \
  --vae outputs/vae --cache outputs/flow_cache --workers 16
```

This requires the completed latent export, fitted corpus cache, and aligned MRI files. It builds training-only latent statistics and retains the exact decoder identity.

### 4. Train the conditional flow

Update [gaussian_latent_fm_measured_r22.json](configs/gaussian_latent_fm_measured_r22.json) to match the prepared assets:

| Key | Value for the example workflow |
| --- | --- |
| `cache` | `outputs/flow_cache` |
| `vae_run` | `outputs/vae` |
| `vae_checkpoint` | Actual exported checkpoint stem, without `.pt` |
| `physical_gpu` | Physical GPU index, for example `"1"` |
| `monitor_cache` | Path to the prepared external monitoring cache |
| `rest_root` | External root containing matching raw and processed subject inventories |
| `measured_cache` | Optional lossless measured-query cache, or `null` to read native NIfTI files |

The checked-in `export_step000136700` checkpoint name and historical cache paths refer to the original run; change them when using newly trained assets.

The main flow trainer requires a separate **265-subject external monitoring cache**, with one representative middle nonempty slice per subject, using exactly the training cache's statistics, decoder, and source identities. Supply this cache externally or implement its preparation against the validation contract in [gaussian_latent_pmf/data.py](gaussian_latent_pmf/data.py). Its full builder is not included. The 20-subject monitoring subset produced by `gaussian_latent_fm.prepare` cannot directly replace it.

```bash
CUDA_VISIBLE_DEVICES=1 python -m gaussian_latent_fm_measured.train \
  --config configs/gaussian_latent_fm_measured_r22.json \
  --output outputs/flow
```

`CUDA_VISIBLE_DEVICES` must equal the configuration's `physical_gpu`; the process uses logical `cuda:0`. Add `--resume` to restore a saved run.

## MRI-to-DWI synthesis

The CLI generates a case already present in the prepared cache. Replace `SUBJECT_SLICE` with its manifest case ID:

```bash
python -m gaussian_latent_fm.infer \
  --run outputs/flow --case SUBJECT_SLICE \
  --weights ema --steps 1 --method euler \
  --output generated.npz
```

It loads `checkpoints/latest.pt` and the matching frozen VAE assets. The archive contains the generated latent, Gaussian coefficients, raw-intensity DWI, acquisition table, and metadata. This is a cached-case entry point; arbitrary MRI file ingestion is not implemented by this CLI.

One Euler step uses one network evaluation (1 NFE). Continuous positive-b query matrices can also be constructed through `codec.renderer.query_matrix(directions, bvals)`.

## Angular super-resolution

ASR is exposed as a Python API. Supply the trained flow model and codec, their frozen latent mean/std, native B0, the complete acquisition table, acquired indices, and **only the acquired DWI values** as the loss target. `conditions` has channel order **T1, T2, B0**, padded to `160 × 176` using the training preprocessing.

```python
from gaussian_asr import (
    FrozenGaussianRuntime, InversionConfig, InverseSRObservationLoss,
    load_inversesr_vgg16, solve_inversion,
)

# flow_model, codec, latent_mean/std, and input tensors are supplied by the caller.
# b0: [B,145,174]; observed: [B,Q_observed,145,174]
# directions: [Q,3]; bvals: [Q]; observed_indices: [Q_observed]
runtime = FrozenGaussianRuntime(
    model=flow_model, codec=codec,
    latent_mean=latent_mean, latent_std=latent_std,
    b0=b0, query_basis=codec.renderer.query_matrix(directions, bvals),
    observed_indices=observed_indices,
)
vgg = load_inversesr_vgg16("/path/to/vgg16.pt", device=runtime.device)
loss = InverseSRObservationLoss(observed, vgg, lambda_perc=1e4, lambda_mae=1.0)
result = solve_inversion(
    loss, runtime.render_observed,
    latent_shape=runtime.latent_shape,
    config=InversionConfig(mode="flow", updates=600, learning_rate=0.07, flow_steps=1),
    flow_model=flow_model, conditions=conditions,
    frozen_modules=(codec,), final_render=runtime.render_all,
    autocast_factory=runtime.autocast,
)
prediction = result.prediction
```

The feature extractor is the external InverseSR-compatible TorchScript VGG16, not a substituted torchvision classifier. Keep tensors and frozen modules on the same device, and leave autograd enabled during inversion.

For **Latent Inversion**, use `InversionConfig(mode="latent")` and omit `flow_model` and `conditions` from `solve_inversion`. The runtime still uses the supplied model's configuration to determine the latent shape. Both modes default to 600 Adam updates at learning rate `0.07`; Flow Inversion uses one NFE **per trajectory**, with backward recomputation and a final forward evaluation in addition to the optimization steps. For a completed acquisition, retain measured values at acquired indices and use predictions for missing directions.

## Third-party sources

The included Stable Diffusion architecture sources retain their [upstream license](stable-diffusion/LICENSE). The MRI condition encoder adaptation is implemented in [gaussian_latent_fm/model.py](gaussian_latent_fm/model.py); a separate Q-MMTN checkout is not required.
