# Data-to-manuscript map

- **Three-seed classification**: `results/model_classification_complexity.csv`; seed-level reports under each model `results/training/`.
- **Batch-1 PyTorch/TensorRT GPU compute latency**: `results/pytorch_gpu_forward_batch1.csv` and `MODEL/results/jetson_gpu/tensorrt_*.csv`. These report GPU-forward execution, not pipeline latency.
- **Accuracy / macro-F1 by precision and model complexity**: `results/precision_accuracy_model_complexity.csv` (seed 42 precision-specific values), `results/model_classification_complexity.csv` (three-seed summaries).
- **Cross-platform E2E comparison**: `results/cross_platform_e2e_batch1_4_8_16.csv` (all nine configurations, all four batch sizes).
- **Deployment artifact sizes**: `results/deployment_file_sizes.csv` (per-format sizes).
- **Ablation**: `NANO-SDRN/ablation/` with tri-scale reference and component controls.
- **Classwise error analysis**: `results/nano_sdrn_classwise_three_seeds.csv` and representative seed-42 confusion matrix under `NANO-SDRN/results/training/figures/`.
- **TOPSIS**: `results/topsis/` with nine decision matrices, weights and independent score verification.

The repository does not present removed calibration/reliability, selective classification, paired-bootstrap, CPU thread-scaling, or training-time comparison as manuscript analyses. Device-specific results contain the four-thread CPU comparison conditions.

## Accuracy versus deployment trade-off panels

- Storage: use `results/precision_accuracy_model_complexity.csv` with `results/deployment_file_sizes.csv` for FP32 ONNX and TensorRT INT8-PTQ.
- E2E latency: use `results/cross_platform_e2e_batch1_4_8_16.csv`, selecting seed 42 and batch size 1. Jetson PyTorch FP32 / TensorRT FP16 and TensorRT INT8-PTQ / Raspberry Pi 5 CPU ONNX Runtime INT8-PTQ provide the four panel conditions.
- Accuracy values used in these panels are from the seed 42 checkpoints, not three-seed means.
