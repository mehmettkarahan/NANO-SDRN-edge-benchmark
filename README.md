# NANO-SDRN — Lightweight image classification on edge devices

Research code and recorded evaluation data for **NANO-SDRN** and seven lightweight baselines: MobileNetV4, RepViT, SHViT, StarNet, FastViT, GhostNetV2 and EdgeNeXt.

Eight models are trained from scratch using common settings for three tomato leaf classes. Deployment comparisons cover **NVIDIA Jetson Orin Nano Super** (PyTorch/TensorRT GPU, ONNX Runtime CPU) and **Raspberry Pi 5** (ONNX Runtime CPU).

## Experimental conditions

| Device | Backend | Precisions | Batch sizes |
|---|---|---|---|
| Jetson Orin Nano Super GPU | TensorRT | FP32, FP16, INT8-PTQ | 1, 4, 8, 16 |
| Jetson Orin Nano Super GPU | PyTorch | FP32, FP16 | 1, 4, 8, 16 |
| Jetson Orin Nano Super CPU | ONNX Runtime, 4 threads | FP32, INT8-PTQ | 1, 4, 8, 16 |
| Raspberry Pi 5 CPU | ONNX Runtime, 4 threads | FP32, INT8-PTQ | 1, 4, 8, 16 |

Classification/complexity and TOPSIS results are separately archived. Classification is reported for training seeds 42, 123 and 2026 where available; deployment uses the seed 42 checkpoint.

## Find results quickly

| What | Location |
|---|---|
| 8-model classification and complexity (three training seeds) | [`results/model_classification_complexity.csv`](results/model_classification_complexity.csv) |
| Seed-42, precision-specific accuracy/macro-F1 and model complexity | [`results/precision_accuracy_model_complexity.csv`](results/precision_accuracy_model_complexity.csv) |
| Deployment file sizes (ONNX, TensorRT, checkpoint) | [`results/deployment_file_sizes.csv`](results/deployment_file_sizes.csv) |
| Seed-42, 9-condition batch 1/4/8/16 **E2E** latency, accuracy, macro-F1, images/s | [`results/cross_platform_e2e_batch1_4_8_16.csv`](results/cross_platform_e2e_batch1_4_8_16.csv) |
| NANO-SDRN class-wise three-seed precision/recall/F1 | [`results/nano_sdrn_classwise_three_seeds.csv`](results/nano_sdrn_classwise_three_seeds.csv) |
| Nine-condition TOPSIS: matrices, weighting, scores, mean ranks | [`results/topsis/`](results/topsis/) |
| PyTorch E2E batch-level timing samples and full summaries | [`reproducibility/deployment/pytorch_e2e/`](reproducibility/deployment/pytorch_e2e/) |
| Reproduce TOPSIS | `python scripts/verify_topsis.py` |
| Recreate NANO-SDRN test confusion matrix and ROC | `python scripts/plot_classification_figures.py` |
| Model-specific experiments, training, TensorRT, CPU, ablation | `MODEL_NAME/` (e.g. [`NANO-SDRN/`](NANO-SDRN/)) |
| Dataset manifests, image generation, runtime and deployment methods | [`reproducibility/`](reproducibility/) |

## What is meant by latency?

**GPU-forward** measures model execution on the GPU with CUDA events (without image loading, preprocessing or transfers). **Pipeline E2E** measures image loading, preprocessing, inference and prediction; GPU E2E also includes host/device transfers. The two timings describe **different scopes**, including at batch size 1. For `batch > 1`, the tabulated `ms/image` is the *batch processing time divided by the batch size*, not the latency seen by a single image request.


## TOPSIS evaluation

TOPSIS compares **macro-F1 (%)** (benefit), **E2E ms/image** (cost), and **deployment file size (MiB)** (cost), using **Equal**, **Entropy**, and **CRITIC** weights. Each of the nine batch-1 hardware/runtime conditions uses its own **8 × 3 decision matrix**. For PyTorch FP16 the checkpoint file on disk has the same size as FP32, because FP16 conversion happens in memory. See [`results/topsis/README.md`](results/topsis/README.md).

## Running the workflows

All experiment scripts require the actual datasets and trained checkpoints, which are not included in this repository. Training/export, TensorRT inference, and ONNX Runtime utilities are organized by model. To run the 64 PyTorch E2E configurations with the supplied multi-batch benchmark implementation:

```bash
python scripts/jetson_pytorch_E2E_FP32_FP16_B1_B4_B8_B16.py \
  --weights-dir /path/to/checkpoints \
  --real-test /path/to/real_test \
  --output-dir /path/to/results \
  --batches 1 4 8 16 \
  --precisions FP32 FP16
```

For exact experimental limitations and software environments, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The original PlantVillage image files and generated synthetic images are not redistributed here. The dataset split manifests and the Stable Diffusion img2img generation settings are preserved.

## Scope of archived results

Only classification quality (accuracy, macro-F1 and related classification metrics), model complexity (parameters and GMACs), measured inference latency/nominal throughput, on-disk model/engine size, ablation analyses and nine-condition TOPSIS are reported. The recorded results remain those of the original experiments; excluded columns were not substituted with newly estimated values.
