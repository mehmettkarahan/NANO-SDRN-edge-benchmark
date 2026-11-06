# Training environment and model definition

The archived training environment reports Python 3.10.20, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, timm 1.0.29, CUDA 13.0, and MAXN_SUPER performance mode on Jetson Orin Nano.

`experiment_config.json` records the common training protocol, including 224x224 inputs, batch size 16, 5-fold CV, 20 CV epochs, up to 30 final epochs, AdamW (3e-4, weight decay 1e-4), CosineAnnealingLR, early stopping patience 7, seeds 42/123/2026, FP32 training, and ONNX opset 17.

`nano_sdrn_model.py` is the final NANO-SDRN architecture used in the manuscript. `baseline_variants.json` records the exact timm model identifiers and `pretrained=false` setting.

Training and final deployment were performed in separate software environments. CUDA 13.0 is the training runtime.
