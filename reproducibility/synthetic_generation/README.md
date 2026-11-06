# Synthetic Image Generation

This directory contains the script used to reproduce the synthetic tomato-leaf
generation workflow described in the manuscript.

## Generation settings

- Model: `runwayml/stable-diffusion-v1-5`
- Pipeline: `StableDiffusionImg2ImgPipeline`
- Scheduler: `DPMSolverMultistepScheduler`
- Input/output size: 512 × 512
- Strength: 0.65
- Guidance scale: 8.0
- Inference steps: 25
- Target images per class: 2000
- Source images: `real_validation`
- Output split: `synthetic_train`

The class-specific positive prompts and the shared negative prompt are defined
directly in `generate_synthetic_img2img.py`.

## Reproducibility note

The experiment-era generation workflow did not set an explicit diffusion RNG
seed. Therefore, exact pixel-level regeneration of the synthetic images is not
claimed.

## Usage

```bash
python generate_synthetic_img2img.py \
  --source-dir /path/to/real_validation \
  --output-dir /path/to/synthetic_train
```
