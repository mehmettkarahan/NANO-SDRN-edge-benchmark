# EdgeNeXt

Model code and training outputs for **EdgeNeXt** in the NANO-SDRN edge-inference benchmark.

- `code/model.py`, `code/train.py`: model configuration and three-seed training.
- `code/export_onnx.py`, `code/quantize_int8.py`: portable ONNX utilities. TensorRT INT8-PTQ calibration is documented separately under `reproducibility/deployment/`.
- `results/training/`: train/validation/test records and classification reports.
- `results/jetson_gpu/pytorch_fp32.csv`, `pytorch_fp16.csv`: **batch=1 GPU-forward** latency statistics.
- `results/jetson_gpu/pytorch_e2e_fp32_batch1_4_8_16.csv`, `pytorch_e2e_fp16_batch1_4_8_16.csv`: PyTorch **pipeline E2E**, four batch sizes.
- `results/jetson_gpu/tensorrt_*.csv`: batch 1, 4, 8 and 16 TensorRT inference results.
- `results/jetson_cpu/`, `results/raspberry_pi5/`: ONNX Runtime FP32/INT8-PTQ, **4 CPU threads**, batch 1, 4, 8, 16.
- `../reproducibility/deployment/pytorch_e2e/EdgeNeXt/`: PyTorch FP32/FP16 **pipeline E2E**, batch 1, 4, 8, 16 (per-run summary and timing samples).

The unified platform comparison is `../results/cross_platform_e2e_batch1_4_8_16.csv`.
Training example: `python code/train.py --data-root /path/to/dataset --output-dir training_output`.
Model weights and image datasets are not included in this release.
