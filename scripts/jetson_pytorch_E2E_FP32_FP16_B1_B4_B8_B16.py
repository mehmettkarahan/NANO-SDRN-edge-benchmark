"""
Jetson Orin Nano PyTorch FP32 / FP16 true end-to-end benchmark, seed 42.

Companion to jetson_pytorch_fp32_fp16_B1_B4_B8_B16.py.
Runs 8 models x 2 precisions x B=1/4/8/16 (64 configurations).
Uses the same reported evaluation stages as the TensorRT benchmark where possible:
  - Same 1,500 real_test images (one evaluation pass per configuration)
  - Same short-side Resize(256)/CenterCrop(224)/ImageNet normalize
  - Image reading + CPU preprocessing INSIDE e2e timed section
  - H2D -> CUDA GPU forward -> D2H -> CPU softmax and argmax
  - 50 model warmups per configuration
  - Pad final batch by repeating last valid sample; exclude pad from accuracy
  - CUDA Event GPU-only time ALSO separately saved, not used as e2e
  - Exact nominal throughput convention: 1000 * B / mean_ms_per_batch

Backend-specific input preparation and timings cannot be identical to TensorRT,
but the inclusion/exclusion boundary of end-to-end timing matches the source.
GPU-forward measurements are separate from E2E measurements.
This script does not train or change checkpoints.

IMPORTANT: place companion jetson_pytorch_fp32_fp16_B1_B4_B8_B16.py in the SAME folder."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import jetson_pytorch_fp32_fp16_B1_B4_B8_B16 as source

def parse_args():
    p = argparse.ArgumentParser(description='Jetson PyTorch FP32/FP16 end-to-end benchmark')
    home = Path.home()
    p.add_argument('--weights-dir', type=Path, default=home / 'Desktop/modeller_8')
    p.add_argument('--real-test', type=Path, default=home / 'Desktop/dataset/real_test')
    p.add_argument('--output-dir', type=Path, default=home / 'Desktop/pytorch_gpu_END_TO_END_B1_B4_B8_B16')
    p.add_argument('--models', nargs='+', default=source.MODELS, choices=source.MODELS)
    p.add_argument('--batches', nargs='+', default=[1, 4, 8, 16], type=int)
    p.add_argument('--precisions', nargs='+', choices=['FP32', 'FP16'], default=['FP32', 'FP16'])
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--check-only', action='store_true')
    return p.parse_args()

def preprocess_image(path: Path) -> np.ndarray:
    """Mirrors source TensorRT preprocess_image, NCHW float32 output."""
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        if w < 1 or h < 1:
            raise ValueError(f'Invalid image dimensions: {path}')
        if w <= h:
            resized = (256, int(256 * h / w))
        else:
            resized = (int(256 * w / h), 256)
        im = im.resize(resized, resample=Image.Resampling.BILINEAR)
        w, h = im.size
        left = int(round((w - 224) / 2.0))
        top = int(round((h - 224) / 2.0))
        im = im.crop((left, top, left + 224, top + 224))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(source.MEAN, dtype=np.float32)[None, None, :]) / np.asarray(source.STD, dtype=np.float32)[None, None, :]
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, :, :, :], dtype=np.float32)

def make_batch(chunk, B):
    xs = [preprocess_image(path) for path, label in chunk]
    n = len(xs)
    if n < 1 or n > B:
        raise ValueError(f'Invalid batch: n={n} B={B}')
    if n < B:
        xs.extend([xs[-1]] * (B - n))
    batch = np.ascontiguousarray(np.concatenate(xs, axis=0), dtype=np.float32)
    if batch.shape != (B, 3, 224, 224):
        raise AssertionError(f'Bad shape: {batch.shape}')
    return (batch, n)

def softmax_numpy(logits: np.ndarray):
    logits = logits.astype(np.float32, copy=False)
    logits = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(logits)
    return ez / ez.sum(axis=1, keepdims=True)

def stats_ms(values):
    a = np.asarray(values, dtype=np.float64)
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99))}

def main():
    args = parse_args()
    args.weights_dir = args.weights_dir.expanduser().resolve()
    args.real_test = args.real_test.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.weights_dir.is_dir() or not args.real_test.is_dir():
        raise FileNotFoundError(f'Check folders: {args.weights_dir} ; {args.real_test}')
    if len(args.batches) != len(set(args.batches)) or min(args.batches) <= 0:
        raise ValueError('Batch sizes must be unique, positive integers')
    if args.warmup < 0:
        raise ValueError('Invalid warmup count')
    checkpoints = {model: source.locate_checkpoint(args.weights_dir, model) for model in args.models}
    dataset = source.build_dataset(args.real_test)
    if len(dataset.classes) != 3:
        raise ValueError(f'Expected 3 classes, found {dataset.classes}')
    paths_and_labels = [(Path(filename), int(label)) for filename, label in dataset.samples]
    print(f'[DATASET] {len(paths_and_labels)} images, classes={dataset.classes}', flush=True)
    for model, path in checkpoints.items():
        print(f'[CHECKPOINT] {model:14s} {path}', flush=True)
    if args.check_only:
        print('OK: paths and seed-42 file names were located. No model inference was run.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available: use the Jetson GPU PyTorch environment')
    source.prepare()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = {'purpose': 'E2E, one full real-test pass per configuration, not original PyTorch repeated GPU-only protocol', 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'seed': source.SEED, 'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'platform': platform.platform(), 'batches': args.batches, 'precision': args.precisions, 'classes': dataset.classes, 'num_real_test': len(dataset), 'warmup': args.warmup, 'timing_boundary': 'time.perf_counter_ns immediately before reading/CPU preprocessing through after GPU H2D/forward/D2H + CPU softmax/argmax', 'gpu_only_timing': 'CUDA events around model(x), separately recorded', 'preprocessing': 'PIL RGB, Resize(256 shorter edge, BILINEAR), CenterCrop(224), float32 /255, ImageNet normalization', 'dtype': 'FP32/FP16 model parameters and input on GPU; preprocessed input float32 on host', 'batch_pad': 'repeat last valid image to B; exclude padded predictions from accuracy', 'result_scope': 'E2E includes on-demand image read and preprocessing, host/device transfer, GPU compute, CPU prediction; excludes checkpoint loading, warmup and CSV writing'}
    (args.output_dir / 'environment.json').write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding='utf-8')
    rows, errors = ([], [])
    device = torch.device('cuda:0')
    for midx, name in enumerate(args.models, 1):
        ckpt = checkpoints[name]
        for precision in args.precisions:
            dtype = torch.float32 if precision == 'FP32' else torch.float16
            print(f'[{midx}/{len(args.models)}] {name} {precision}', flush=True)
            try:
                gc.collect()
                torch.cuda.empty_cache()
                model = source.build_model(name, len(dataset.classes))
                state = source.read_state_dict(ckpt)
                model.load_state_dict(state, strict=True)
                del state
                model = model.to(device=device, dtype=dtype).eval()
                for B in args.batches:
                    log_dir = args.output_dir / name / precision / f'B{B}'
                    log_dir.mkdir(parents=True, exist_ok=True)
                    print(f'   B{B}...', flush=True)
                    try:
                        first, _ = make_batch(paths_and_labels[:B], B)
                        warm_x = torch.from_numpy(first).to(device=device, dtype=dtype, non_blocking=False)
                        with torch.inference_mode():
                            for _ in range(args.warmup):
                                _ = model(warm_x)
                        torch.cuda.synchronize()
                        del warm_x
                        prep_times = []
                        h2d_times = []
                        gpu_times = []
                        d2h_times = []
                        post_times = []
                        e2e_times = []
                        truth = []
                        predictions = []
                        try:
                            with torch.inference_mode():
                                for start in range(0, len(paths_and_labels), B):
                                    chunk = paths_and_labels[start:start + B]
                                    t0 = time.perf_counter_ns()
                                    x_np, valid = make_batch(chunk, B)
                                    x_host = torch.from_numpy(x_np)
                                    t1 = time.perf_counter_ns()
                                    x_dev = x_host.to(device=device, dtype=dtype, non_blocking=False)
                                    torch.cuda.synchronize()
                                    t2 = time.perf_counter_ns()
                                    ev0 = torch.cuda.Event(enable_timing=True)
                                    ev1 = torch.cuda.Event(enable_timing=True)
                                    ev0.record()
                                    logits = model(x_dev)
                                    ev1.record()
                                    ev1.synchronize()
                                    t3 = time.perf_counter_ns()
                                    gpu_ms = float(ev0.elapsed_time(ev1))
                                    host_logits = logits.detach().cpu().numpy()
                                    t4 = time.perf_counter_ns()
                                    probs = softmax_numpy(host_logits[:valid])
                                    pred = np.argmax(probs, axis=1).astype(np.int64)
                                    t5 = time.perf_counter_ns()
                                    prep_times.append((t1 - t0) / 1000000.0)
                                    h2d_times.append((t2 - t1) / 1000000.0)
                                    gpu_times.append(gpu_ms)
                                    d2h_times.append((t4 - t3) / 1000000.0)
                                    post_times.append((t5 - t4) / 1000000.0)
                                    e2e_times.append((t5 - t0) / 1000000.0)
                                    truth.extend((label for _, label in chunk))
                                    predictions.extend(pred.tolist())
                                    del x_np, x_host, x_dev, logits, host_logits, probs
                        finally:
                            pass
                        if len(truth) != len(dataset) or len(predictions) != len(dataset):
                            raise RuntimeError('Prediction count mismatch')
                        acc = float(accuracy_score(truth, predictions) * 100)
                        f1 = float(precision_recall_fscore_support(truth, predictions, average='macro', zero_division=0, labels=list(range(len(dataset.classes))))[2] * 100)
                        e = stats_ms(e2e_times)
                        g = stats_ms(gpu_times)
                        row = {'Model': name, 'Platform': 'Jetson Orin Nano GPU', 'Backend': 'PyTorch', 'Precision': precision, 'Batch': B, 'Seed': 42, 'Real-test images': len(dataset), 'Warmup runs': args.warmup, 'Measured batches': len(e2e_times), 'End-to-End Mean (ms/batch)': e['mean'], 'End-to-End P50 (ms/batch)': e['p50'], 'End-to-End P95 (ms/batch)': e['p95'], 'End-to-End P99 (ms/batch)': e['p99'], 'Amortized End-to-End Mean (ms/image)': e['mean'] / B, 'Nominal End-to-End Throughput (images/s)': B * 1000 / e['mean'], 'GPU Mean (ms/batch)': g['mean'], 'GPU P50 (ms/batch)': g['p50'], 'GPU P95 (ms/batch)': g['p95'], 'GPU P99 (ms/batch)': g['p99'], 'Amortized GPU Mean (ms/image)': g['mean'] / B, 'Nominal GPU Throughput (images/s)': B * 1000 / g['mean'], 'Preprocess Mean (ms/batch)': float(np.mean(prep_times)), 'H2D Mean (ms/batch)': float(np.mean(h2d_times)), 'D2H Mean (ms/batch)': float(np.mean(d2h_times)), 'Postprocess Mean (ms/batch)': float(np.mean(post_times)), 'Accuracy (%)': acc, 'Macro-F1 (%)': f1, 'Checkpoint': str(ckpt), 'Checkpoint SHA256': source.file_sha256(ckpt), 'Method': 'one-pass E2E incl image IO+prep H2D GPU D2H post; CUDA Event forward separately'}
                        rows.append(row)
                        source.dump_csv(log_dir / 'summary.csv', [row])
                        source.dump_csv(log_dir / 'timings_ms.csv', [{'e2e_ms_batch': e2e_times[i], 'gpu_ms_batch': gpu_times[i], 'pre_ms_batch': prep_times[i], 'h2d_ms_batch': h2d_times[i], 'd2h_ms_batch': d2h_times[i], 'post_ms_batch': post_times[i]} for i in range(len(e2e_times))])
                        print(f"     E2E {e['mean']:.3f} ms/batch | {B * 1000 / e['mean']:.1f} img/s | GPU-only {g['mean']:.3f} ms/batch | Acc {acc:.2f}%", flush=True)
                    except Exception as exc:
                        errors.append({'Model': name, 'Precision': precision, 'Batch': B, 'Error': f'{type(exc).__name__}: {exc}'})
                        print(f'   ERROR B{B}: {type(exc).__name__}: {exc}', flush=True)
                    finally:
                        source.dump_csv(args.output_dir / 'PYTORCH_GPU_E2E_SCI_SUMMARY.csv', rows)
                        source.dump_csv(args.output_dir / 'failures.csv', errors)
                del model
            except Exception as exc:
                for B in args.batches:
                    errors.append({'Model': name, 'Precision': precision, 'Batch': B, 'Error': f'{type(exc).__name__}: {exc}'})
                source.dump_csv(args.output_dir / 'failures.csv', errors)
                print(f' MODEL ERROR: {type(exc).__name__}: {exc}', flush=True)
        torch.cuda.empty_cache()
    print(f'COMPLETED: {len(rows)} successful, {len(errors)} failed (expected 64 successful).', flush=True)
    print('CSV:', args.output_dir / 'PYTORCH_GPU_E2E_SCI_SUMMARY.csv', flush=True)
    if not rows:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
