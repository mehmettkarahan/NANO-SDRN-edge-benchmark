"""
SCI/SCIE-oriented Jetson Orin Nano PyTorch FP16 inference benchmark.

Purpose
-------
Create the FP16 counterpart of the existing PyTorch FP32 deployment benchmark
without retraining and without changing checkpoints, preprocessing, warm-up,
timing policy, or representative seed.

Fair-comparison rules
---------------------
1) Uses the SAME seed-42 validation-selected checkpoints as the FP32/TensorRT runs.
2) Uses the SAME real_test evaluation preprocessing:
       Resize(256) -> CenterCrop(224) -> ToTensor() -> ImageNet Normalize
3) Uses the SAME PyTorch B1 forward benchmark policy:
       50 warm-up runs
       >=300 timed runs
       >=5 s benchmark duration
       <=5000 timed runs
       CUDA Events for GPU forward latency
4) Uses the SAME 100-run H2D + forward + postprocess benchmark.
6) TF32 is disabled.
7) FP16 is explicit: the FP32 checkpoint is loaded, then model.half() is applied.
   No autocast/mixed-precision fallback is used.
8) No training is performed. The original checkpoint files are never modified.
9) Speed benchmark remains batch size 1, matching the existing PyTorch FP32 benchmark.

Default eight-model set
-----------------------
MobileNetV4, RepViT, SHViT, StarNet, FastViT, GhostNetV2, EdgeNeXt,
NANO-SDRN.

Run
---
source ~/torch_gpu_env/bin/activate

python -u ~/Desktop/sci_jetson_gpu_pytorch_fp16.py   2>&1 | tee ~/Desktop/sci_jetson_gpu_pytorch_fp16.log"""
from __future__ import annotations
import os
import sys
import gc
import re
import json
import time
import math
import hashlib
import argparse
import platform
import threading
import subprocess
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score, roc_auc_score
IMG_SIZE = 224
REPRESENTATIVE_SEED = 42
PYTORCH_WARMUP_RUNS = 50
PYTORCH_MIN_TIMED_RUNS = 300
PYTORCH_MAX_TIMED_RUNS = 5000
PYTORCH_MIN_BENCHMARK_SECONDS = 5.0
PIPELINE_BENCHMARK_RUNS = 100
EVAL_BATCH_SIZE = 16
EVAL_NUM_WORKERS = 2
MODEL_ORDER = ['MobileNetV4', 'RepViT', 'SHViT', 'StarNet', 'FastViT', 'GhostNetV2', 'EdgeNeXt', 'NANO-SDRN']
BASELINE_MODELS = set(MODEL_ORDER[:-1])
NANO_MODEL = 'NANO-SDRN'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HOME = Path.home()
DESKTOP = HOME / 'Desktop'
DATASET_ROOT = DESKTOP / 'dataset'
DEFAULT_REAL_TEST = DATASET_ROOT / 'real_test'
DEFAULT_OUTPUT_DIR = DESKTOP / 'jetson_gpu_realtest_sci_pytorch_fp16_B1'
BASELINE_SOURCE_CANDIDATES = [DESKTOP / 'jetson_lightweight_academic_benchmark.py', DESKTOP / 'jetson 9 model egitim sonuclari' / 'jetson_lightweight_academic_benchmark.py']
NANO_SOURCE_CANDIDATES = [DESKTOP / 'nano_sdrn_v' / 'nano_sdrn_v11_detailed_ablation_study_fair.py', DESKTOP / 'jetson 9 model egitim sonuclari' / 'nano_sdrn_v11_detailed_ablation_study_fair.py', DESKTOP / 'nano_sdrn_v11_detailed_ablation_study_fair.py']
CHECKPOINTS = {'MobileNetV4': DATASET_ROOT / 'comparison_results' / 'MobileNetV4' / 'MobileNetV4_seed_42_best_real_validation.pth', 'RepViT': DATASET_ROOT / 'comparison_results' / 'RepViT' / 'RepViT_seed_42_best_real_validation.pth', 'SHViT': DATASET_ROOT / 'comparison_results' / 'SHViT' / 'SHViT_seed_42_best_real_validation.pth', 'StarNet': DATASET_ROOT / 'comparison_results' / 'StarNet' / 'StarNet_seed_42_best_real_validation.pth', 'FastViT': DATASET_ROOT / 'comparison_results' / 'FastViT' / 'FastViT_seed_42_best_real_validation.pth', 'GhostNetV2': DATASET_ROOT / 'comparison_results' / 'GhostNetV2' / 'GhostNetV2_seed_42_best_real_validation.pth', 'EdgeNeXt': DATASET_ROOT / 'comparison_results' / 'EdgeNeXt' / 'EdgeNeXt_seed_42_best_real_validation.pth', 'NANO-SDRN': DATASET_ROOT / 'comparison_results_nano_sdrn_v11_ablation_fair' / 'NANO-SDRN' / 'NANO-SDRN_seed_42_best_real_validation.pth'}

def safe_name(name: str) -> str:
    return re.sub('[^A-Za-z0-9._-]+', '_', str(name))

def sha256_file(path: Path, chunk_size: int=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def first_existing(candidates):
    for p in candidates:
        p = Path(p).expanduser()
        if p.is_file():
            return p.resolve()
    return None

def load_module(path: Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Python source not found: {path}')
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import source module: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_state_dict(path: Path):
    path = Path(path).expanduser().resolve()
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and obj and all((torch.is_tensor(v) for v in obj.values())):
        return obj
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    raise RuntimeError(f'{path}: checkpoint does not contain a recognizable state_dict.')

def set_deployment_mode():
    torch.manual_seed(REPRESENTATIVE_SEED)
    np.random.seed(REPRESENTATIVE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(REPRESENTATIVE_SEED)
        torch.cuda.manual_seed_all(REPRESENTATIVE_SEED)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision('highest')
    except Exception:
        pass

def summarize_latencies_ms(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {'mean_ms': np.nan, 'median_p50_ms': np.nan, 'p90_ms': np.nan, 'p95_ms': np.nan, 'p99_ms': np.nan, 'std_ms': np.nan, 'min_ms': np.nan, 'max_ms': np.nan, 'fps_from_mean': np.nan, 'timed_runs': 0}
    mean_ms = float(np.mean(arr))
    return {'mean_ms': mean_ms, 'median_p50_ms': float(np.percentile(arr, 50)), 'p90_ms': float(np.percentile(arr, 90)), 'p95_ms': float(np.percentile(arr, 95)), 'p99_ms': float(np.percentile(arr, 99)), 'std_ms': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 'min_ms': float(np.min(arr)), 'max_ms': float(np.max(arr)), 'fps_from_mean': 1000.0 / mean_ms if mean_ms > 0 else np.nan, 'timed_runs': int(arr.size)}

def build_real_test(real_test_root: Path):
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    dataset = datasets.ImageFolder(str(real_test_root), transform=eval_tf)
    if len(dataset) == 0:
        raise RuntimeError(f'real_test is empty: {real_test_root}')
    return dataset

def construct_model(model_name, num_classes, baseline_module, nano_module):
    if model_name in BASELINE_MODELS:
        return baseline_module.get_model(model_name, num_classes)
    if model_name == NANO_MODEL:
        return nano_module.get_model(model_name, num_classes)
    raise ValueError(f'Unknown model: {model_name}')

def convert_to_explicit_fp16(model, device):
    """
    Explicit FP16 inference:
      FP32 checkpoint -> model.half() -> CUDA.
    No autocast is used, so this is not labelled mixed precision.
    """
    model = model.eval()
    model = model.half()
    model = model.to(device)
    bad = []
    for name, p in model.named_parameters():
        if p.is_floating_point() and p.dtype != torch.float16:
            bad.append((f'parameter:{name}', str(p.dtype)))
    for name, b in model.named_buffers():
        if b.is_floating_point() and b.dtype != torch.float16:
            bad.append((f'buffer:{name}', str(b.dtype)))
    if bad:
        raise RuntimeError('Explicit FP16 conversion audit failed; floating tensors remain non-FP16: ' + str(bad[:20]))
    return model

def benchmark_pytorch_fp16_forward(model, sample_chw, device, output_dir):
    set_deployment_mode()
    model.eval()
    x = sample_chw.unsqueeze(0).to(device=device, dtype=torch.float16, non_blocking=False).contiguous()
    with torch.inference_mode():
        for _ in range(PYTORCH_WARMUP_RUNS):
            out = model(x)
        torch.cuda.synchronize()
        output_dtype = str(out.dtype)
        latencies = []
        wall_start = time.perf_counter()
        while True:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(float(starter.elapsed_time(ender)))
            elapsed = time.perf_counter() - wall_start
            enough_runs = len(latencies) >= PYTORCH_MIN_TIMED_RUNS
            enough_time = elapsed >= PYTORCH_MIN_BENCHMARK_SECONDS
            if enough_runs and enough_time:
                break
            if len(latencies) >= PYTORCH_MAX_TIMED_RUNS:
                break
    latency_stats = summarize_latencies_ms(latencies)
    pd.DataFrame({'latency_ms': latencies}).to_csv(output_dir / 'pytorch_fp16_forward_latency_samples.csv', index=False)
    return {'Precision': 'FP16', 'FP16 Mode': 'explicit model.half(); no autocast', 'Batch Size': 1, 'Input Dtype': str(x.dtype), 'Output Dtype': output_dtype, **{f'forward_{k}': v for k, v in latency_stats.items()}}

def benchmark_fp16_h2d_forward_postprocess(model, sample_chw, device):
    model.eval()
    cpu_x = sample_chw.unsqueeze(0).contiguous().half()
    latencies = []
    with torch.inference_mode():
        for _ in range(10):
            x = cpu_x.to(device, non_blocking=False)
            out = model(x)
            _ = F.softmax(out, dim=1).argmax(dim=1)
            torch.cuda.synchronize()
        for _ in range(PIPELINE_BENCHMARK_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = cpu_x.to(device, non_blocking=False)
            out = model(x)
            _ = F.softmax(out, dim=1).argmax(dim=1)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    stats = summarize_latencies_ms(latencies)
    return {f'h2d_forward_postprocess_{k}': v for k, v in stats.items()}

def evaluate_real_test_fp16(model, dataset, device):
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=EVAL_NUM_WORKERS, pin_memory=True, persistent_workers=False, drop_last=False)
    y_true = []
    y_pred = []
    y_prob = []
    model.eval()
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float16, non_blocking=False)
            logits = model(xb)
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.append(probs.cpu().numpy())
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.concatenate(y_prob, axis=0).astype(np.float64)
    labels = np.arange(len(dataset.classes), dtype=np.int64)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    try:
        auc_macro = float(roc_auc_score(y_true, y_prob, labels=labels, multi_class='ovr', average='macro'))
    except Exception:
        auc_macro = np.nan
    return {'real_test_images': int(len(y_true)), 'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_precision': float(p_macro), 'macro_recall': float(r_macro), 'macro_f1': float(f1_macro), 'mcc': float(matthews_corrcoef(y_true, y_pred)), 'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)), 'roc_auc_macro_ovr': auc_macro}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--real-test', type=str, default=str(DEFAULT_REAL_TEST))
    p.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument('--baseline-source', type=str, default='', help='Optional exact path to jetson_lightweight_academic_benchmark.py')
    p.add_argument('--nano-source', type=str, default='', help='Optional exact path to nano_sdrn_v11_detailed_ablation_study_fair.py')
    p.add_argument('--models', nargs='*', default=None, help='Optional subset. Default: all eight publication models.')
    return p.parse_args()

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. This benchmark must run on the Jetson GPU.')
    device = torch.device('cuda:0')
    set_deployment_mode()
    real_test_root = Path(args.real_test).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_source = Path(args.baseline_source).expanduser().resolve() if args.baseline_source else first_existing(BASELINE_SOURCE_CANDIDATES)
    nano_source = Path(args.nano_source).expanduser().resolve() if args.nano_source else first_existing(NANO_SOURCE_CANDIDATES)
    if baseline_source is None:
        raise FileNotFoundError('Could not find jetson_lightweight_academic_benchmark.py.\nUse --baseline-source /exact/path/to/file.py')
    if nano_source is None:
        raise FileNotFoundError('Could not find nano_sdrn_v11_detailed_ablation_study_fair.py.\nUse --nano-source /exact/path/to/file.py')
    baseline_module = load_module(baseline_source, '_fp16_baseline_source')
    nano_module = load_module(nano_source, '_fp16_nano_source')
    dataset = build_real_test(real_test_root)
    num_classes = len(dataset.classes)
    requested_models = MODEL_ORDER if not args.models else list(dict.fromkeys(args.models))
    unknown = [m for m in requested_models if m not in MODEL_ORDER]
    if unknown:
        raise ValueError('Unknown --models entries: ' + ', '.join(unknown))
    missing_ckpts = [(m, CHECKPOINTS[m]) for m in requested_models if not CHECKPOINTS[m].is_file()]
    if missing_ckpts:
        msg = '\n'.join((f'  {m}: {p}' for m, p in missing_ckpts))
        raise FileNotFoundError('Required seed-42 checkpoint(s) not found:\n' + msg)
    environment = {'timestamp_local': time.strftime('%Y-%m-%d %H:%M:%S'), 'platform': platform.platform(), 'python': sys.version.replace('\n', ' '), 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'sklearn': sklearn.__version__, 'cuda_reported_by_torch': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None, 'device': str(device), 'gpu_name': torch.cuda.get_device_name(0), 'precision': 'FP16', 'fp16_mode': 'explicit model.half(); no autocast', 'allow_tf32': False, 'representative_seed': REPRESENTATIVE_SEED, 'speed_batch_size': 1, 'warmup_runs': PYTORCH_WARMUP_RUNS, 'min_timed_runs': PYTORCH_MIN_TIMED_RUNS, 'max_timed_runs': PYTORCH_MAX_TIMED_RUNS, 'min_benchmark_seconds': PYTORCH_MIN_BENCHMARK_SECONDS, 'pipeline_runs': PIPELINE_BENCHMARK_RUNS, 'real_test_root': str(real_test_root), 'real_test_images': len(dataset), 'class_names': dataset.classes, 'class_to_idx': dataset.class_to_idx, 'eval_transform': 'Resize(256) -> CenterCrop(224) -> ToTensor() -> ImageNet Normalize', 'baseline_source': str(baseline_source), 'nano_source': str(nano_source), 'models': requested_models}
    (output_root / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print('=' * 118)
    print('JETSON ORIN NANO — PYTORCH FP16 GPU BENCHMARK — FAIR B1 COUNTERPART TO EXISTING PYTORCH FP32')
    print('=' * 118)
    print(f'GPU                 : {torch.cuda.get_device_name(0)}')
    print(f'PyTorch             : {torch.__version__}')
    print(f'CUDA                : {torch.version.cuda}')
    print(f'Precision           : explicit FP16 (model.half), no autocast')
    print(f'TF32                : disabled')
    print(f'Models              : {len(requested_models)}')
    print(f'real_test           : {real_test_root} | N={len(dataset)}')
    print(f'Speed batch         : B1')
    print(f'Warm-up             : {PYTORCH_WARMUP_RUNS}')
    print(f'Timing policy       : >= {PYTORCH_MIN_TIMED_RUNS} runs AND >= {PYTORCH_MIN_BENCHMARK_SECONDS:.1f}s')
    print(f'Baseline source     : {baseline_source}')
    print(f'NANO source         : {nano_source}')
    print('=' * 118)
    sample_chw, sample_label = dataset[0]
    sample_path = dataset.samples[0][0]
    rows = []
    failures = []
    for idx, model_name in enumerate(requested_models, start=1):
        print('\n' + '#' * 118)
        print(f'[{idx}/{len(requested_models)}] {model_name}')
        print('#' * 118)
        model_out = output_root / safe_name(model_name)
        model_out.mkdir(parents=True, exist_ok=True)
        ckpt = CHECKPOINTS[model_name].resolve()
        try:
            gc.collect()
            torch.cuda.empty_cache()
            model = construct_model(model_name, num_classes, baseline_module, nano_module)
            state = load_state_dict(ckpt)
            model.load_state_dict(state, strict=True)
            model = convert_to_explicit_fp16(model, device)
            checkpoint_info = {'Model': model_name, 'Checkpoint Path': str(ckpt), 'Checkpoint SHA256': sha256_file(ckpt), 'Checkpoint Size (MiB)': ckpt.stat().st_size / 1024 ** 2, 'Representative Seed': REPRESENTATIVE_SEED, 'Sample Path': str(sample_path), 'Sample Label': int(sample_label)}
            print(f'  checkpoint : {ckpt}')
            print(f"  sha256     : {checkpoint_info['Checkpoint SHA256']}")
            print(f'  sample     : {sample_path}')
            forward = benchmark_pytorch_fp16_forward(model, sample_chw, device, model_out)
            pipeline = benchmark_fp16_h2d_forward_postprocess(model, sample_chw, device)
            classification = evaluate_real_test_fp16(model, dataset, device)
            row = {**checkpoint_info, **forward, **pipeline, **classification}
            rows.append(row)
            pd.DataFrame([row]).to_csv(model_out / 'pytorch_fp16_summary.csv', index=False)
            (model_out / 'pytorch_fp16_summary.json').write_text(json.dumps({k: None if isinstance(v, float) and (not math.isfinite(v)) else v for k, v in row.items()}, indent=2), encoding='utf-8')
            print(f"  FP16 forward : {row['forward_mean_ms']:.4f} ms | P95={row['forward_p95_ms']:.4f} ms | FPS={row['forward_fps_from_mean']:.2f}")
            print(f"  real_test    : Acc={100.0 * row['accuracy']:.2f}% | Macro-F1={100.0 * row['macro_f1']:.2f}%")
            del model, state
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            failures.append({'Model': model_name, 'Error Type': type(exc).__name__, 'Error': str(exc), 'Checkpoint': str(ckpt)})
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            gc.collect()
            torch.cuda.empty_cache()
        if rows:
            pd.DataFrame(rows).to_csv(output_root / 'pytorch_fp16_8models_partial.csv', index=False)
        if failures:
            pd.DataFrame(failures).to_csv(output_root / 'failures.csv', index=False)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_root / 'pytorch_fp16_8models_summary.csv', index=False)
        df.to_json(output_root / 'pytorch_fp16_8models_summary.json', orient='records', indent=2)
        compact_cols = ['Model', 'Precision', 'Batch Size', 'forward_mean_ms', 'forward_p95_ms', 'forward_p99_ms', 'forward_fps_from_mean', 'h2d_forward_postprocess_mean_ms', 'accuracy', 'macro_f1', 'mcc', 'roc_auc_macro_ovr', 'Checkpoint SHA256']
        df[[c for c in compact_cols if c in df.columns]].to_csv(output_root / 'pytorch_fp16_publication_compact.csv', index=False)
    print('\n' + '=' * 118)
    print('PYTORCH FP16 BENCHMARK COMPLETE')
    print(f'Output directory: {output_root}')
    print(f'Successful models: {len(rows)}/{len(requested_models)}')
    print(f'Failures         : {len(failures)}')
    print('=' * 118)
    if failures:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
