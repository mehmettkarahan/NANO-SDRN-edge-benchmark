# Device inference data and scripts

- `scripts/`: PyTorch GPU-forward batch=1 and TensorRT CUDA-event benchmark scripts, with engine/calibration configuration.
- `raw_latency_samples/`: per-model PyTorch GPU-forward batch=1 distributions.
- `pytorch_e2e/`: PyTorch FP32/FP16 batch 1, 4, 8, 16 pipeline timings and per-condition summaries.
- `results_summaries/`: device benchmark summaries and GPU-forward latency statistics.
- `metadata/`: prepared model, engine build and INT8-PTQ validation calibration manifests.
- `environments/`: training/deployment environment details.

GPU-forward timings exclude image I/O and host preparation. E2E timings include image read, preprocessing, inference and prediction; GPU E2E also includes transfers. The hardware and methodology of the recorded measurements must be kept distinct. The target comparison uses the seed 42 checkpoint, batch 1/4/8/16 for pipeline E2E, and four threads for CPU.
