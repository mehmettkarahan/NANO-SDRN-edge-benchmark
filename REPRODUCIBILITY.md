# Reproducibility and measurement definitions

The repository includes the eight architecture definitions or timm identifiers, a shared training protocol, reference image split manifests, on-device benchmark scripts, recorded outputs and TOPSIS decision data.

## Dataset and learning

- Original images: PlantVillage, classes `Tomato__Early_blight`, `Tomato__Late_blight`, and `Tomato__healthy`.
- 6,000 synthetic training images (2,000/class); 1,500 real validation and 1,500 real test images (500/class each).
- Real validation served as both source images for synthetic image-to-image generation and model selection; it is not independent of the synthetic generation procedure.
- Five-fold synthetic-domain CV; final training seeds **42, 123, 2026**, selected on validation macro-F1. Deployment measurements use **seed 42**.
- Synthetic-domain CV used stratified folds without source-image grouping; images generated from a shared source may therefore appear in different folds. These CV estimates are descriptive and not an independent source-disjoint validation.
- Cross-split auditing of real validation and real test identified one exact content match and six perceptual-hash candidates; the saved sensitivity analysis is at `reproducibility/dataset/test_overlap_sensitivity_analysis.csv` (relative to repository root, `reproducibility/dataset/`).
- The diffusion generation code preserves prompts and settings but no per-image RNG seeds were recorded; identical pixel-level regenerated synthetic images are not guaranteed.

## Timing scopes (do not substitute between them)

| Scope | Included stages | Configurations | Files |
|---|---|---|---|
| GPU-forward | CUDA-event GPU execution, excluding image I/O and host processing | Jetson PyTorch FP32/FP16, batch 1 | `results/pytorch_gpu_forward_batch1.csv` |
| GPU-forward | TensorRT CUDA-event compute; engine build excluded | Jetson TensorRT FP32/FP16/INT8-PTQ | `MODEL/results/jetson_gpu/tensorrt_*.csv` |
| Pipeline E2E | Image loading, preprocessing, inference, prediction; GPU configurations also include memory transfers | Nine device/backend/precision conditions × batches 1/4/8/16 | `results/cross_platform_e2e_batch1_4_8_16.csv` |

When batch > 1, `ms/image` is **amortized processing time** (`batch time / batch size`), not single-request latency. `images/s` is nominal throughput (`1000 / ms_per_image`). CPU comparisons use **4 threads** on Jetson and Raspberry Pi 5. CPU FP16 is not presented as a native FP16 configuration.


The source benchmark conditions for GPU-forward and E2E are separately documented. Their means should not be combined as though the timed operations were identical. The published paper uses rounded presentation values while the CSV files retain numerical precision.

## Runtime

- Training: Python 3.10.20, PyTorch 2.14.0+cu130, CUDA 13.0.
- Jetson PyTorch inference: Python 3.10.20, PyTorch 2.14.0+cu132, CUDA 13.2.
- TensorRT: Python 3.12.3, TensorRT 10.16.2.10.
- ONNX Runtime CPU: 1.23.2, CPUExecutionProvider (4 threads).

Training example: `python MODEL/code/train.py --data-root /path/to/dataset --output-dir /path/to/output` from the model directory. PyTorch pipeline E2E benchmark: `python scripts/jetson_pytorch_E2E_FP32_FP16_B1_B4_B8_B16.py --weights-dir /path/to/checkpoints --real-test /path/to/real_test --output-dir /path/to/measurements` (the companion Python file must be in the same scripts directory).

Model weights, images, ONNX graphs and TensorRT engine binaries are not distributed. Checkpoint and engine path strings embedded in archived provenance records describe the experiment machine; they are not public download paths. Checkpoint SHA-256 hashes are retained where supplied. Stable Diffusion settings and dataset audits are under `reproducibility/`.

The main accuracy/complexity and E2E tables are in `results/`; the nine-condition TOPSIS data and independent verification script are in `results/topsis/` and `scripts/verify_topsis.py`.

The seed-42 confusion matrix and OvR ROC images can be regenerated from the saved NANO-SDRN test predictions with `python scripts/plot_classification_figures.py`.
