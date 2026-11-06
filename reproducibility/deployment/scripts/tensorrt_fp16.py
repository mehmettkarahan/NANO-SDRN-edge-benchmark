"""
Jetson Orin Nano — SCI REAL_TEST TensorRT FP16 GPU benchmark
================================================================

IMPORTANT ENVIRONMENT
---------------------
This version intentionally does NOT import PyTorch.

Reason:
- TensorRT Python binding on this Jetson is installed for system Python 3.12.
- The existing torch_gpu_env uses Python 3.10.
- Mixing those ABI-specific Python packages is unsafe.

This script therefore uses:
  - system TensorRT Python 10.16.2.10
  - CUDA Runtime directly through ctypes (libcudart)
  - NumPy host buffers
  - TensorRT execute_async_v3()

It benchmarks every selected ONNX file on the SAME real_test images.

Metrics
-------
Classification:
  Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1,
  MCC, Cohen Kappa, ROC-AUC Macro OvR, per-class metrics, confusion matrix.

Deployment:
  TensorRT FP16 GPU latency mean/median/p90/p95/p99/std/min/max,
  throughput FPS, H2D, D2H, preprocessing, end-to-end latency,

Evaluation preprocessing
------------------------
Matches the source torchvision evaluation pipeline:
  transforms.Resize(256)
  transforms.CenterCrop((224,224))
  transforms.ToTensor()
  transforms.Normalize(
      mean=[0.485,0.456,0.406],
      std =[0.229,0.224,0.225]
  )

For torchvision Resize(256), the new long edge uses integer truncation:
  int(256 * long / short)

Expected layout
---------------
~/Desktop/models_9/
    MobileNetV4.onnx
    RepViT.onnx
    SHViT.onnx
    StarNet.onnx
    FastViT.onnx
    GhostNetV2.onnx
    EdgeNeXt.onnx
    NANO-SDRN.onnx

~/Desktop/dataset/real_test/
    Tomato__Early_blight/
    Tomato__Late_blight/
    Tomato__healthy/

Run
---
source ~/trt_gpu_env/bin/activate

python -u ~/Desktop/sci_jetson_gpu_realtest_trt_fp16.py   --models-root ~/Desktop/models_9   --real-test ~/Desktop/dataset/real_test   --output-dir ~/Desktop/jetson_gpu_realtest_sci_trt_fp16   2>&1 | tee ~/Desktop/jetson_gpu_realtest_sci_trt_fp16.log"""
import os
import re
import gc
import json
import time
import math
import shutil
import ctypes
import ctypes.util
import hashlib
import argparse
import platform
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
from PIL import Image
import tensorrt as trt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_auc_score
PREFERRED_MODEL_ORDER = ['MobileNetV4', 'RepViT', 'SHViT', 'StarNet', 'FastViT', 'GhostNetV2', 'EdgeNeXt', 'NANO-SDRN']
CLASS_NAMES = ['Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__healthy']
CLASS_TO_IDX = {'Tomato__Early_blight': 0, 'Tomato__Late_blight': 1, 'Tomato__healthy': 2}
EXPECTED_IMAGES_PER_CLASS = 500
EXPECTED_TOTAL_IMAGES = 1500
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
IMG_SIZE = 224
RESIZE_SHORT_EDGE = 256
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
DEFAULT_WARMUP = 50
DEFAULT_WORKSPACE_GIB = 2.0
DEFAULT_COOLDOWN_S = 8.0
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2

def load_cudart():
    candidates = []
    found = ctypes.util.find_library('cudart')
    if found:
        candidates.append(found)
    candidates += ['libcudart.so', 'libcudart.so.13', 'libcudart.so.13.0', '/usr/local/cuda/lib64/libcudart.so', '/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so', '/usr/local/cuda-13.2/targets/aarch64-linux/lib/libcudart.so', '/usr/local/cuda-13.0/targets/aarch64-linux/lib/libcudart.so']
    last_err = None
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            return (lib, candidate)
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f'CUDA Runtime (libcudart) yüklenemedi. Son hata: {last_err}')
CUDART, CUDART_PATH = load_cudart()
CUDART.cudaGetErrorString.argtypes = [ctypes.c_int]
CUDART.cudaGetErrorString.restype = ctypes.c_char_p
CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
CUDART.cudaMalloc.restype = ctypes.c_int
CUDART.cudaFree.argtypes = [ctypes.c_void_p]
CUDART.cudaFree.restype = ctypes.c_int
CUDART.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
CUDART.cudaStreamCreate.restype = ctypes.c_int
CUDART.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
CUDART.cudaStreamDestroy.restype = ctypes.c_int
CUDART.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
CUDART.cudaStreamSynchronize.restype = ctypes.c_int
CUDART.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
CUDART.cudaMemcpyAsync.restype = ctypes.c_int
CUDART.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
CUDART.cudaEventCreate.restype = ctypes.c_int
CUDART.cudaEventDestroy.argtypes = [ctypes.c_void_p]
CUDART.cudaEventDestroy.restype = ctypes.c_int
CUDART.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
CUDART.cudaEventRecord.restype = ctypes.c_int
CUDART.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
CUDART.cudaEventElapsedTime.restype = ctypes.c_int

def cuda_check(code, where):
    if code != CUDA_SUCCESS:
        msg = CUDART.cudaGetErrorString(code)
        text = msg.decode('utf-8', errors='replace') if msg else 'unknown'
        raise RuntimeError(f'CUDA error at {where}: code={code}, {text}')

def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    cuda_check(CUDART.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))), 'cudaMalloc')
    return ptr

def cuda_event():
    ev = ctypes.c_void_p()
    cuda_check(CUDART.cudaEventCreate(ctypes.byref(ev)), 'cudaEventCreate')
    return ev

def elapsed_event_ms(start_ev, end_ev):
    ms = ctypes.c_float()
    cuda_check(CUDART.cudaEventElapsedTime(ctypes.byref(ms), start_ev, end_ev), 'cudaEventElapsedTime')
    return float(ms.value)

def percentile(values, p):
    if len(values) == 0:
        return float('nan')
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))

def run_cmd(cmd, timeout=3):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception:
        return ''

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def inspect_onnx_artifact(path: Path):
    """Best-effort ONNX structural metadata; never required for inference."""
    out = {'ONNX Inspection Available': False, 'ONNX Initializer Tensors': float('nan'), 'ONNX Initializer Elements': float('nan'), 'ONNX Initializer Elements (M)': float('nan'), 'ONNX Graph Nodes': float('nan'), 'ONNX Opsets': '', 'ONNX Operator Histogram': ''}
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        init_count = len(model.graph.initializer)
        init_elements = 0
        for t in model.graph.initializer:
            n = 1
            for d in t.dims:
                n *= int(d)
            init_elements += n
        op_hist = {}
        for node in model.graph.node:
            op_hist[node.op_type] = op_hist.get(node.op_type, 0) + 1
        opsets = {str(x.domain or 'ai.onnx'): int(x.version) for x in model.opset_import}
        out.update({'ONNX Inspection Available': True, 'ONNX Initializer Tensors': int(init_count), 'ONNX Initializer Elements': int(init_elements), 'ONNX Initializer Elements (M)': float(init_elements / 1000000.0), 'ONNX Graph Nodes': int(len(model.graph.node)), 'ONNX Opsets': json.dumps(opsets, sort_keys=True), 'ONNX Operator Histogram': json.dumps(op_hist, sort_keys=True)})
    except Exception as exc:
        out['ONNX Inspection Error'] = f'{type(exc).__name__}: {exc}'
    return out

def detect_device():
    parts = []
    for candidate in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        try:
            parts.append(Path(candidate).read_text(errors='ignore').replace('\x00', '').strip())
        except Exception:
            pass
    return ' | '.join((x for x in parts if x)) or platform.node()

def read_temperature_c():
    vals = []
    root = Path('/sys/class/thermal')
    if root.exists():
        for z in root.glob('thermal_zone*'):
            try:
                v = float((z / 'temp').read_text().strip())
                if v > 1000:
                    v /= 1000.0
                if -20 <= v <= 150:
                    vals.append(v)
            except Exception:
                pass
    return max(vals) if vals else float('nan')

def discover_real_test(real_test_root):
    real_test_root = Path(real_test_root)
    if not real_test_root.is_dir():
        raise FileNotFoundError(f'real_test bulunamadı: {real_test_root}')
    actual_dirs = sorted((p.name for p in real_test_root.iterdir() if p.is_dir()))
    missing = [c for c in CLASS_NAMES if c not in actual_dirs]
    if missing:
        raise RuntimeError('Eksik real_test sınıf klasörü:\n' + '\n'.join((f'  - {x}' for x in missing)) + f'\nBulunan: {actual_dirs}')
    samples = []
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = real_test_root / class_name
        paths = sorted((p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
        counts[class_name] = len(paths)
        if len(paths) != EXPECTED_IMAGES_PER_CLASS:
            raise RuntimeError(f'{class_name}: {len(paths)} görüntü; beklenen={EXPECTED_IMAGES_PER_CLASS}')
        class_idx = CLASS_TO_IDX[class_name]
        for p in paths:
            samples.append((p.resolve(), class_idx, class_name))
    if len(samples) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(f'Toplam real_test={len(samples)}, beklenen={EXPECTED_TOTAL_IMAGES}')
    return (samples, counts)

def resize_short_edge_bilinear(image, size=256):
    """
    Match torchvision.transforms.Resize(256) for PIL images.

    torchvision computes:
        new_long = int(size * long / short)
    i.e. truncation, not round().
    """
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f'Invalid image size: {(w, h)}')
    if w <= h:
        new_w = int(size)
        new_h = int(size * h / w)
    else:
        new_h = int(size)
        new_w = int(size * w / h)
    return image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

def center_crop(image, crop_size=224):
    w, h = image.size
    left = int(round((w - crop_size) / 2.0))
    top = int(round((h - crop_size) / 2.0))
    return image.crop((left, top, left + crop_size, top + crop_size))

def preprocess_image(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im = resize_short_edge_bilinear(im, RESIZE_SHORT_EDGE)
        im = center_crop(im, IMG_SIZE)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)

def load_external_model_metadata(csv_path):
    """Optional merge of official complexity fields (e.g. Parameters/GMACs)."""
    if not csv_path:
        return ({}, None)
    p = Path(os.path.expanduser(str(csv_path))).resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Metadata CSV bulunamadı: {p}')
    df = pd.read_csv(p)
    if 'Model' not in df.columns:
        raise RuntimeError("Metadata CSV must contain a 'Model' column.")
    mapping = {}
    for _, row in df.iterrows():
        name = str(row['Model'])
        d = {}
        for col, val in row.items():
            if col == 'Model':
                continue
            if pd.isna(val):
                continue
            if isinstance(val, np.generic):
                val = val.item()
            d[str(col)] = val
        mapping[name] = d
    return (mapping, p)

def attach_external_metadata(row, model_name, metadata_map):
    meta = metadata_map.get(model_name, {})
    for key, val in meta.items():
        out_key = key if key not in row else f'Metadata::{key}'
        row[out_key] = val
    return row

def discover_models(models_root, requested_names=None):
    models_root = Path(models_root)
    if not models_root.is_dir():
        raise FileNotFoundError(f'Model klasörü bulunamadı: {models_root}')
    all_onnx = sorted((p.resolve() for p in models_root.rglob('*.onnx')))
    if not all_onnx:
        raise RuntimeError(f'Hiç ONNX modeli bulunamadı: {models_root}')
    by_name = {}
    duplicates = {}
    for p in all_onnx:
        name = p.stem
        if name in by_name:
            duplicates.setdefault(name, [by_name[name]]).append(p)
        else:
            by_name[name] = p
    if duplicates:
        lines = [f'{k}: ' + ' | '.join(map(str, v)) for k, v in duplicates.items()]
        raise RuntimeError("Aynı stem'e sahip birden fazla ONNX var; isimleri benzersiz yapın:\n" + '\n'.join(lines))
    if requested_names:
        missing = [n for n in requested_names if n not in by_name]
        if missing:
            raise RuntimeError('İstenen modeller bulunamadı: ' + ', '.join(missing))
        names = list(dict.fromkeys(requested_names))
    else:
        preferred = [n for n in PREFERRED_MODEL_ORDER if n in by_name]
        extras = sorted((n for n in by_name if n not in preferred))
        names = preferred + extras
    return {name: by_name[name] for name in names}
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]

def validate_batch_sizes(values):
    vals = []
    for x in values:
        x = int(x)
        if x < 1:
            raise ValueError(f'Batch size must be >=1, got {x}')
        if x not in vals:
            vals.append(x)
    if not vals:
        raise ValueError('At least one batch size is required.')
    return sorted(vals)

def _onnx_graph_compute_fingerprint(model):
    """Fingerprint compute graph nodes + initializers, intentionally excluding IO shape metadata."""
    h = hashlib.sha256()
    for init in model.graph.initializer:
        h.update(init.SerializeToString())
    for node in model.graph.node:
        h.update(node.SerializeToString())
    return h.hexdigest()

def prepare_batch_capable_onnx(source_path: Path, prepared_dir: Path):
    """Create a dynamic-batch copy if legacy ONNX input batch is fixed to 1.

    The computational graph and initializer fingerprint must remain identical.
    TensorRT engine preflight later executes every requested batch size, so any
    model whose internal graph still hard-codes batch=1 will fail before timing.
    """
    try:
        import onnx
    except Exception as exc:
        raise RuntimeError(f"The GPU true-batch benchmark needs the 'onnx' Python package to inspect/prepare legacy fixed-batch-1 artifacts. Install it in this TensorRT environment with 'python -m pip install onnx' or provide ONNX files already exported with a dynamic batch axis. Import error: {exc}")
    source_path = Path(source_path).resolve()
    model = onnx.load(str(source_path))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError(f'{source_path.name}: expected one input and one output; found {len(model.graph.input)} input(s), {len(model.graph.output)} output(s).')
    shp = model.graph.input[0].type.tensor_type.shape
    if len(shp.dim) != 4:
        raise RuntimeError(f'{source_path.name}: expected rank-4 NCHW input.')
    bdim = shp.dim[0]
    source_dynamic = bool(bdim.dim_param) or not bdim.HasField('dim_value') or (bdim.HasField('dim_value') and int(bdim.dim_value) <= 0)
    source_batch_value = int(bdim.dim_value) if bdim.HasField('dim_value') else None
    source_fp = _onnx_graph_compute_fingerprint(model)
    if source_dynamic:
        prepared_path = prepared_dir / f'{source_path.stem}.dynamic_batch.onnx'
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, prepared_path)
        preparation = 'source_already_dynamic_copied_byte_identically'
        prepared_fp = source_fp
    elif source_batch_value == 1:
        prepared_path = prepared_dir / f'{source_path.stem}.dynamic_batch.onnx'
        inp_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        inp_dim.ClearField('dim_value')
        inp_dim.dim_param = 'batch'
        out_shape = model.graph.output[0].type.tensor_type.shape
        if len(out_shape.dim) >= 1:
            out_dim = out_shape.dim[0]
            out_dim.ClearField('dim_value')
            out_dim.dim_param = 'batch'
        prepared_dir.mkdir(parents=True, exist_ok=True)
        onnx.checker.check_model(model)
        onnx.save(model, str(prepared_path))
        prepared_model = onnx.load(str(prepared_path))
        prepared_fp = _onnx_graph_compute_fingerprint(prepared_model)
        if prepared_fp != source_fp:
            raise RuntimeError(f'{source_path.name}: compute-graph fingerprint changed while preparing dynamic batch.')
        preparation = 'batch_metadata_patched_compute_graph_fingerprint_identical'
    else:
        raise RuntimeError(f'{source_path.name}: source batch dimension is fixed to {source_batch_value}; automatic preparation only supports legacy fixed-batch-1 artifacts.')
    return (prepared_path, {'Source ONNX Path': str(source_path), 'Source ONNX SHA256': sha256_file(source_path), 'Prepared ONNX Path': str(prepared_path), 'Prepared ONNX SHA256': sha256_file(prepared_path), 'Batch Preparation': preparation, 'Compute Graph Fingerprint': source_fp, 'Compute Graph Fingerprint Equal': bool(source_fp == prepared_fp)})

def prepare_clean_output_dir(output_dir: Path, clean_output: bool):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if clean_output:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError(f'Output directory is not empty: {output_dir}\nFor a publication-clean run, use a new directory or pass --clean-output.')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def build_true_batch(samples_chunk, batch_size):
    xs = []
    prep_each_ms = []
    for path, _, _ in samples_chunk:
        t0 = time.perf_counter_ns()
        x = preprocess_image(path)
        t1 = time.perf_counter_ns()
        xs.append(x)
        prep_each_ms.append((t1 - t0) / 1000000.0)
    if not xs:
        raise RuntimeError('Empty batch encountered.')
    valid_n = len(xs)
    pad_n = batch_size - valid_n
    if pad_n < 0:
        raise RuntimeError('Internal batch construction error.')
    if pad_n:
        xs.extend([xs[-1]] * pad_n)
    x_batch = np.concatenate(xs, axis=0)
    if x_batch.shape != (batch_size, 3, IMG_SIZE, IMG_SIZE):
        raise RuntimeError(f'Constructed input has shape {x_batch.shape}; expected ({batch_size},3,{IMG_SIZE},{IMG_SIZE})')
    return (np.ascontiguousarray(x_batch, dtype=np.float32), valid_n, pad_n, prep_each_ms)

def build_engine_fp16(model_name, onnx_path, engine_path, workspace_gib, batch_sizes):
    batch_sizes = validate_batch_sizes(batch_sizes)
    build_start = time.perf_counter()
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f'{model_name} ONNX parse failed:\n' + '\n'.join(errors))
    if network.num_inputs != 1:
        raise RuntimeError(f'{model_name}: expected one input, found {network.num_inputs}')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(float(workspace_gib) * 1024 ** 3))
    config.set_flag(trt.BuilderFlag.FP16)
    input_tensor = network.get_input(0)
    input_shape = tuple((int(d) for d in input_tensor.shape))
    if len(input_shape) != 4:
        raise RuntimeError(f'{model_name}: expected rank-4 input, got {input_shape}')
    min_b = min(batch_sizes)
    max_b = max(batch_sizes)
    opt_b = batch_sizes[len(batch_sizes) // 2]
    if input_shape[0] >= 0:
        if input_shape[0] != min_b or len(batch_sizes) > 1:
            raise RuntimeError(f'{model_name}: TensorRT parser still sees static batch shape {input_shape}. Use a true dynamic-batch ONNX re-export.')
    else:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, min=(min_b, 3, IMG_SIZE, IMG_SIZE), opt=(opt_b, 3, IMG_SIZE, IMG_SIZE), max=(max_b, 3, IMG_SIZE, IMG_SIZE))
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT FP16 engine build failed: {model_name}')
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return (time.perf_counter() - build_start, {'profile_min_batch': min_b, 'profile_opt_batch': opt_b, 'profile_max_batch': max_b, 'parsed_input_shape': str(input_shape)})

def ensure_engine(model_name, onnx_path, engine_dir, workspace_gib, rebuild, batch_sizes):
    batch_sizes = validate_batch_sizes(batch_sizes)
    profile_key = f'B{min(batch_sizes)}-{batch_sizes[len(batch_sizes) // 2]}-{max(batch_sizes)}'
    engine_path = Path(engine_dir) / f'{model_name}.fp16.{profile_key}.engine'
    meta_path = Path(str(engine_path) + '.meta.json')
    onnx_hash = sha256_file(onnx_path)
    expected_profile = {'min': min(batch_sizes), 'opt': batch_sizes[len(batch_sizes) // 2], 'max': max(batch_sizes)}
    use_existing = False
    if engine_path.exists() and meta_path.exists() and (not rebuild):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            use_existing = meta.get('onnx_sha256') == onnx_hash and meta.get('tensorrt_version') == trt.__version__ and (meta.get('precision') == 'FP16') and (meta.get('batch_profile') == expected_profile)
        except Exception:
            use_existing = False
    build_seconds = 0.0
    build_info = {'profile_min_batch': expected_profile['min'], 'profile_opt_batch': expected_profile['opt'], 'profile_max_batch': expected_profile['max'], 'parsed_input_shape': None}
    if not use_existing:
        print(f'Building dynamic FP16 engine: {model_name} | profile={expected_profile}')
        build_seconds, build_info = build_engine_fp16(model_name, onnx_path, engine_path, workspace_gib, batch_sizes)
        meta = {'model': model_name, 'onnx_path': str(onnx_path), 'onnx_sha256': onnx_hash, 'tensorrt_version': trt.__version__, 'precision': 'FP16', 'workspace_gib': workspace_gib, 'batch_profile': expected_profile, 'build_seconds': build_seconds, 'timestamp_local': datetime.now().isoformat(timespec='seconds'), **build_info}
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    else:
        print(f'Using verified dynamic engine: {model_name} | profile={expected_profile}')
        try:
            saved = json.loads(meta_path.read_text(encoding='utf-8'))
            build_info.update({'parsed_input_shape': saved.get('parsed_input_shape')})
        except Exception:
            pass
    return (engine_path, build_seconds, build_info)

class TensorRTRunner:

    def __init__(self, engine_path, batch_size):
        self.batch_size = int(batch_size)
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Engine deserialize failed: {engine_path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context creation failed.')
        self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise RuntimeError(f'Expected one input/output, got inputs={self.input_names}, outputs={self.output_names}')
        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        engine_shape = tuple((int(x) for x in self.engine.get_tensor_shape(self.input_name)))
        requested = (self.batch_size, 3, IMG_SIZE, IMG_SIZE)
        if any((d < 0 for d in engine_shape)):
            ok = self.context.set_input_shape(self.input_name, requested)
            if not ok:
                raise RuntimeError(f'Failed to set dynamic input shape {requested}')
        elif engine_shape != requested:
            raise RuntimeError(f'Static engine shape {engine_shape} does not match requested {requested}')
        self.input_shape = tuple((int(x) for x in self.context.get_tensor_shape(self.input_name)))
        self.output_shape = tuple((int(x) for x in self.context.get_tensor_shape(self.output_name)))
        if any((d < 0 for d in self.input_shape)) or any((d < 0 for d in self.output_shape)):
            raise RuntimeError(f'Unresolved shapes input={self.input_shape}, output={self.output_shape}')
        if self.input_shape[0] != self.batch_size or self.output_shape[0] != self.batch_size:
            raise RuntimeError(f'TRUE batch validation failed: input={self.input_shape}, output={self.output_shape}, requested B={self.batch_size}')
        self.input_dtype_trt = self.engine.get_tensor_dtype(self.input_name)
        self.output_dtype_trt = self.engine.get_tensor_dtype(self.output_name)
        self.input_dtype_np = np.dtype(trt.nptype(self.input_dtype_trt))
        self.output_dtype_np = np.dtype(trt.nptype(self.output_dtype_trt))
        self.input_nbytes = int(np.prod(self.input_shape) * self.input_dtype_np.itemsize)
        self.output_nbytes = int(np.prod(self.output_shape) * self.output_dtype_np.itemsize)
        self.d_input = cuda_malloc(self.input_nbytes)
        self.d_output = cuda_malloc(self.output_nbytes)
        self.stream = ctypes.c_void_p()
        cuda_check(CUDART.cudaStreamCreate(ctypes.byref(self.stream)), 'cudaStreamCreate')
        self.h2d_start, self.h2d_end = (cuda_event(), cuda_event())
        self.gpu_start, self.gpu_end = (cuda_event(), cuda_event())
        self.d2h_start, self.d2h_end = (cuda_event(), cuda_event())
        if not self.context.set_tensor_address(self.input_name, int(self.d_input.value)):
            raise RuntimeError('set_tensor_address(input) failed')
        if not self.context.set_tensor_address(self.output_name, int(self.d_output.value)):
            raise RuntimeError('set_tensor_address(output) failed')
        self.host_output = np.empty(self.output_shape, dtype=self.output_dtype_np)
        self._closed = False

    def run(self, x_np):
        x = np.ascontiguousarray(x_np.astype(self.input_dtype_np, copy=False))
        if x.shape != self.input_shape:
            raise RuntimeError(f'Input shape mismatch: {x.shape} != {self.input_shape}')
        if x.nbytes != self.input_nbytes:
            raise RuntimeError(f'Input byte-size mismatch: {x.nbytes} != {self.input_nbytes}')
        cuda_check(CUDART.cudaEventRecord(self.h2d_start, self.stream), 'cudaEventRecord(h2d_start)')
        cuda_check(CUDART.cudaMemcpyAsync(self.d_input, ctypes.c_void_p(int(x.ctypes.data)), ctypes.c_size_t(self.input_nbytes), CUDA_MEMCPY_HOST_TO_DEVICE, self.stream), 'cudaMemcpyAsync(H2D)')
        cuda_check(CUDART.cudaEventRecord(self.h2d_end, self.stream), 'cudaEventRecord(h2d_end)')
        cuda_check(CUDART.cudaEventRecord(self.gpu_start, self.stream), 'cudaEventRecord(gpu_start)')
        ok = self.context.execute_async_v3(stream_handle=int(self.stream.value))
        if not ok:
            raise RuntimeError('TensorRT execute_async_v3 returned False.')
        cuda_check(CUDART.cudaEventRecord(self.gpu_end, self.stream), 'cudaEventRecord(gpu_end)')
        cuda_check(CUDART.cudaEventRecord(self.d2h_start, self.stream), 'cudaEventRecord(d2h_start)')
        cuda_check(CUDART.cudaMemcpyAsync(ctypes.c_void_p(int(self.host_output.ctypes.data)), self.d_output, ctypes.c_size_t(self.output_nbytes), CUDA_MEMCPY_DEVICE_TO_HOST, self.stream), 'cudaMemcpyAsync(D2H)')
        cuda_check(CUDART.cudaEventRecord(self.d2h_end, self.stream), 'cudaEventRecord(d2h_end)')
        cuda_check(CUDART.cudaStreamSynchronize(self.stream), 'cudaStreamSynchronize')
        return (self.host_output.copy(), elapsed_event_ms(self.h2d_start, self.h2d_end), elapsed_event_ms(self.gpu_start, self.gpu_end), elapsed_event_ms(self.d2h_start, self.d2h_end))

    def close(self):
        if self._closed:
            return
        self._closed = True
        for ev in (self.h2d_start, self.h2d_end, self.gpu_start, self.gpu_end, self.d2h_start, self.d2h_end):
            if ev:
                CUDART.cudaEventDestroy(ev)
        if self.stream:
            CUDART.cudaStreamDestroy(self.stream)
        if self.d_input:
            CUDART.cudaFree(self.d_input)
        if self.d_output:
            CUDART.cudaFree(self.d_output)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def softmax(logits):
    x = np.asarray(logits, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def calculate_metrics(y_true, y_pred, y_prob):
    """Comprehensive multiclass metrics for publication reporting."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    labels = np.arange(len(CLASS_NAMES), dtype=np.int64)
    if y_prob.ndim != 2 or y_prob.shape[1] != len(CLASS_NAMES):
        raise ValueError(f'Probability matrix must be [N,{len(CLASS_NAMES)}], got {y_prob.shape}')
    prob = np.clip(y_prob, 1e-12, 1.0)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='micro', zero_division=0)
    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = int(len(y_true))
    correct_n = int(np.sum(y_true == y_pred))
    acc = float(correct_n / n) if n else float('nan')
    if n:
        z = 1.959963984540054
        denom = 1.0 + z * z / n
        center = (acc + z * z / (2.0 * n)) / denom
        half = z * math.sqrt(acc * (1.0 - acc) / n + z * z / (4.0 * n * n)) / denom
        acc_ci_lo = max(0.0, center - half)
        acc_ci_hi = min(1.0, center + half)
    else:
        acc_ci_lo = acc_ci_hi = float('nan')
    try:
        auc_macro = float(roc_auc_score(y_true, prob, labels=labels, multi_class='ovr', average='macro'))
    except Exception:
        auc_macro = float('nan')
    try:
        auc_weighted = float(roc_auc_score(y_true, prob, labels=labels, multi_class='ovr', average='weighted'))
    except Exception:
        auc_weighted = float('nan')
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[y_true]
    nll = float(-np.mean(np.log(prob[np.arange(n), y_true]))) if n else float('nan')
    brier = float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))) if n else float('nan')
    confidence = np.max(prob, axis=1) if n else np.asarray([], dtype=np.float64)
    correct_mask = y_true == y_pred
    top2 = np.argsort(prob, axis=1)[:, -min(2, len(CLASS_NAMES)):]
    top2_acc = float(np.mean([yt in row for yt, row in zip(y_true, top2)])) if n else float('nan')
    ece = 0.0
    if n:
        edges = np.linspace(0.0, 1.0, 16)
        for bi in range(15):
            lo, hi = (edges[bi], edges[bi + 1])
            mask = (confidence >= lo) & (confidence < hi if bi < 14 else confidence <= hi)
            if np.any(mask):
                bin_acc = float(np.mean(correct_mask[mask]))
                bin_conf = float(np.mean(confidence[mask]))
                ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)
    else:
        ece = float('nan')
    overall = {'Accuracy': acc, 'Accuracy CI95 Lower': float(acc_ci_lo), 'Accuracy CI95 Upper': float(acc_ci_hi), 'Error Rate': float(1.0 - acc) if np.isfinite(acc) else float('nan'), 'Correct Predictions': correct_n, 'Incorrect Predictions': int(n - correct_n), 'Balanced Accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'Macro Precision': float(p_macro), 'Macro Recall': float(r_macro), 'Macro F1': float(f1_macro), 'Weighted Precision': float(p_weighted), 'Weighted Recall': float(r_weighted), 'Weighted F1': float(f1_weighted), 'Micro Precision': float(p_micro), 'Micro Recall': float(r_micro), 'Micro F1': float(f1_micro), 'MCC': float(matthews_corrcoef(y_true, y_pred)), 'Cohen Kappa': float(cohen_kappa_score(y_true, y_pred)), 'ROC-AUC Macro OvR': auc_macro, 'ROC-AUC Weighted OvR': auc_weighted, 'Top-2 Accuracy': top2_acc, 'Negative Log-Likelihood': nll, 'Multiclass Brier Score': brier, 'ECE (15 bins)': float(ece), 'Mean Max Confidence': float(np.mean(confidence)) if n else float('nan'), 'Mean Confidence Correct': float(np.mean(confidence[correct_mask])) if np.any(correct_mask) else float('nan'), 'Mean Confidence Incorrect': float(np.mean(confidence[~correct_mask])) if np.any(~correct_mask) else float('nan')}
    per_class = []
    total = int(cm.sum())
    for i, class_name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        specificity = float(tn / (tn + fp)) if tn + fp else float('nan')
        fpr = float(fp / (fp + tn)) if fp + tn else float('nan')
        fnr = float(fn / (fn + tp)) if fn + tp else float('nan')
        try:
            class_auc = float(roc_auc_score((y_true == i).astype(np.int64), prob[:, i]))
        except Exception:
            class_auc = float('nan')
        per_class.append({'Class Index': i, 'Class Name': class_name, 'Precision': float(p_cls[i]), 'Recall/Sensitivity': float(r_cls[i]), 'Specificity': specificity, 'F1': float(f1_cls[i]), 'FPR': fpr, 'FNR': fnr, 'ROC-AUC OvR': class_auc, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Support': int(support_cls[i])})
    return (overall, per_class, cm)

def benchmark_model(model_name, source_onnx_path, prepared_onnx_path, engine_path, samples, batch_size, warmup, output_dir):
    gc.collect()
    runner = TensorRTRunner(engine_path, batch_size=batch_size)
    first_chunk = samples[:min(batch_size, len(samples))]
    first_x, _, _, _ = build_true_batch(first_chunk, batch_size)
    for _ in range(warmup):
        runner.run(first_x)
    y_true, y_pred, y_prob = ([], [], [])
    preprocess_ms, h2d_ms, gpu_ms, d2h_ms, postprocess_ms, end_to_end_ms = ([], [], [], [], [], [])
    prediction_rows = []
    total_padding = 0
    wall_start = time.perf_counter()
    batch_index = 0
    for start in range(0, len(samples), batch_size):
        batch_index += 1
        chunk = samples[start:start + batch_size]
        e2e_t0 = time.perf_counter_ns()
        prep_t0 = time.perf_counter_ns()
        x, valid_n, pad_n, prep_each = build_true_batch(chunk, batch_size)
        prep_t1 = time.perf_counter_ns()
        total_padding += pad_n
        logits_all, this_h2d_ms, this_gpu_ms, this_d2h_ms = runner.run(x)
        if logits_all.shape[0] != batch_size:
            raise RuntimeError(f'{model_name}: expected output batch {batch_size}, got {logits_all.shape}')
        logits = logits_all[:valid_n]
        post_t0 = time.perf_counter_ns()
        probs_batch = softmax(logits)
        preds_batch = np.argmax(probs_batch, axis=1).astype(np.int64)
        post_t1 = time.perf_counter_ns()
        e2e_t1 = time.perf_counter_ns()
        this_prep_ms = (prep_t1 - prep_t0) / 1000000.0
        this_post_ms = (post_t1 - post_t0) / 1000000.0
        this_e2e_ms = (e2e_t1 - e2e_t0) / 1000000.0
        preprocess_ms.append(this_prep_ms)
        h2d_ms.append(this_h2d_ms)
        gpu_ms.append(this_gpu_ms)
        d2h_ms.append(this_d2h_ms)
        postprocess_ms.append(this_post_ms)
        end_to_end_ms.append(this_e2e_ms)
        for j, ((path, true_idx, true_name), probs, pred_idx) in enumerate(zip(chunk, probs_batch, preds_batch)):
            pred_idx = int(pred_idx)
            global_idx = start + j + 1
            y_true.append(int(true_idx))
            y_pred.append(pred_idx)
            y_prob.append(probs.tolist())
            prediction_rows.append({'Image Index': global_idx, 'Batch Index': batch_index, 'Requested Batch Size': int(batch_size), 'Valid Images in Batch': int(valid_n), 'Padded Images in Batch': int(pad_n), 'Relative Path': str(path.relative_to(path.parents[1])), 'True Index': int(true_idx), 'True Class': true_name, 'Pred Index': pred_idx, 'Pred Class': CLASS_NAMES[pred_idx], 'Correct': int(pred_idx == true_idx), f'Prob_{CLASS_NAMES[0]}': float(probs[0]), f'Prob_{CLASS_NAMES[1]}': float(probs[1]), f'Prob_{CLASS_NAMES[2]}': float(probs[2]), 'Image Preprocess (ms)': float(prep_each[j]), 'H2D (ms/batch)': float(this_h2d_ms), 'TRT GPU (ms/batch)': float(this_gpu_ms), 'D2H (ms/batch)': float(this_d2h_ms), 'Amortized TRT GPU (ms/image)': float(this_gpu_ms / batch_size), 'End-to-End (ms/batch)': float(this_e2e_ms), 'Amortized End-to-End (ms/image)': float(this_e2e_ms / batch_size)})
        done = min(start + batch_size, len(samples))
        if done % 250 == 0 or done == len(samples):
            print(f'    {done}/{len(samples)} images complete | batch={batch_size}')
    wall_total_s = time.perf_counter() - wall_start
    overall, per_class, cm = calculate_metrics(y_true, y_pred, y_prob)
    prep_arr = np.asarray(preprocess_ms, dtype=np.float64)
    h2d_arr = np.asarray(h2d_ms, dtype=np.float64)
    gpu_arr = np.asarray(gpu_ms, dtype=np.float64)
    d2h_arr = np.asarray(d2h_ms, dtype=np.float64)
    post_arr = np.asarray(postprocess_ms, dtype=np.float64)
    e2e_arr = np.asarray(end_to_end_ms, dtype=np.float64)
    gpu_mean_batch = float(np.mean(gpu_arr))
    e2e_mean_batch = float(np.mean(e2e_arr))
    transfer_compute_batch = h2d_arr + gpu_arr + d2h_arr
    transfer_compute_mean_batch = float(np.mean(transfer_compute_batch))
    summary = {'Platform': 'Jetson Orin Nano GPU / TensorRT FP16', 'Model': model_name, 'Batch Size': int(batch_size), 'Precision': 'FP16', 'TensorRT Version': trt.__version__, 'CUDA Runtime Library': CUDART_PATH, 'Source ONNX Path': str(source_onnx_path), 'Source ONNX SHA256': sha256_file(source_onnx_path), 'Prepared ONNX Path': str(prepared_onnx_path), 'Prepared ONNX SHA256': sha256_file(prepared_onnx_path), 'Engine Path': str(engine_path), 'Engine Size (MiB)': Path(engine_path).stat().st_size / 1024 ** 2, 'Input Shape': str(runner.input_shape), 'Output Shape': str(runner.output_shape), 'Input TRT Dtype': str(runner.input_dtype_trt), 'Output TRT Dtype': str(runner.output_dtype_trt), 'real_test Images': int(len(samples)), 'Timed Batches': int(len(gpu_arr)), 'Padded Images': int(total_padding), 'Warmup Batches': int(warmup), **overall, 'TRT GPU Mean (ms/batch)': gpu_mean_batch, 'TRT GPU Median (ms/batch)': float(np.median(gpu_arr)), 'TRT GPU P90 (ms/batch)': percentile(gpu_ms, 90), 'TRT GPU P95 (ms/batch)': percentile(gpu_ms, 95), 'TRT GPU P99 (ms/batch)': percentile(gpu_ms, 99), 'TRT GPU Std (ms/batch)': float(np.std(gpu_arr, ddof=1)) if len(gpu_arr) > 1 else 0.0, 'TRT GPU Min (ms/batch)': float(np.min(gpu_arr)), 'TRT GPU Max (ms/batch)': float(np.max(gpu_arr)), 'Amortized TRT GPU Mean (ms/image)': gpu_mean_batch / batch_size, 'Amortized TRT GPU P95 (ms/image)': percentile(gpu_ms, 95) / batch_size, 'TRT Nominal Throughput (images/s)': batch_size * 1000.0 / gpu_mean_batch, 'TRT Effective Dataset Throughput (images/s)': len(samples) * 1000.0 / float(np.sum(gpu_arr)), 'H2D Mean (ms/batch)': float(np.mean(h2d_arr)), 'D2H Mean (ms/batch)': float(np.mean(d2h_arr)), 'H2D+GPU+D2H Mean (ms/batch)': transfer_compute_mean_batch, 'H2D+GPU+D2H Throughput (images/s)': batch_size * 1000.0 / transfer_compute_mean_batch, 'Preprocess Mean (ms/batch)': float(np.mean(prep_arr)), 'Postprocess Mean (ms/batch)': float(np.mean(post_arr)), 'End-to-End Mean (ms/batch)': e2e_mean_batch, 'End-to-End P95 (ms/batch)': percentile(end_to_end_ms, 95), 'Amortized End-to-End Mean (ms/image)': e2e_mean_batch / batch_size, 'Nominal End-to-End Throughput (images/s)': batch_size * 1000.0 / e2e_mean_batch, 'Effective Dataset End-to-End Throughput (images/s)': len(samples) / wall_total_s, 'Total real_test Wall Time (s)': wall_total_s}
    summary.update(inspect_onnx_artifact(source_onnx_path))
    safe_model = re.sub('[^A-Za-z0-9._-]+', '_', model_name)
    tag = f'{safe_model}_B{batch_size}'
    pd.DataFrame(prediction_rows).to_csv(output_dir / f'predictions_{tag}.csv', index=False)
    class_df = pd.DataFrame(per_class)
    class_df.insert(0, 'Batch Size', batch_size)
    class_df.insert(0, 'Model', model_name)
    class_df.to_csv(output_dir / f'classwise_{tag}.csv', index=False)
    pd.DataFrame(cm, index=[f'True_{c}' for c in CLASS_NAMES], columns=[f'Pred_{c}' for c in CLASS_NAMES]).to_csv(output_dir / f'confusion_matrix_{tag}.csv')
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm_norm, index=[f'True_{c}' for c in CLASS_NAMES], columns=[f'Pred_{c}' for c in CLASS_NAMES]).to_csv(output_dir / f'confusion_matrix_normalized_{tag}.csv')
    runner.close()
    del runner
    gc.collect()
    return (summary, np.asarray(y_pred, dtype=np.int64))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models-root', type=str, default='~/Desktop/models_9')
    p.add_argument('--models', nargs='*', default=None, help='Optional ONNX stem list. Omit to benchmark every .onnx under --models-root.')
    p.add_argument('--metadata-csv', type=str, default='', help='Optional CSV with Model column; official Parameters/GMACs/etc. are merged into summaries.')
    p.add_argument('--real-test', type=str, default='~/Desktop/dataset/real_test')
    p.add_argument('--output-dir', type=str, default='~/Desktop/jetson_gpu_realtest_sci_trt_fp16_B1_B4_B8_B16')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=DEFAULT_BATCH_SIZES, help='TRUE tensor batch sizes. Default: 1 4 8 16')
    p.add_argument('--warmup', type=int, default=DEFAULT_WARMUP, help='Warm-up BATCHES per model/batch configuration.')
    p.add_argument('--workspace-gib', type=float, default=DEFAULT_WORKSPACE_GIB)
    p.add_argument('--cooldown-seconds', type=float, default=DEFAULT_COOLDOWN_S)
    p.add_argument('--rebuild-engines', action='store_true')
    p.add_argument('--clean-output', action='store_true', help='Delete an existing non-empty output directory before starting. Otherwise non-empty output aborts.')
    return p.parse_args()

def main():
    args = parse_args()
    batch_sizes = validate_batch_sizes(args.batch_sizes)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    models_root = Path(os.path.expanduser(args.models_root)).resolve()
    real_test_root = Path(os.path.expanduser(args.real_test)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    prepare_clean_output_dir(output_dir, args.clean_output)
    engine_dir = output_dir / 'engines_fp16_dynamic'
    engine_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / 'prepared_dynamic_onnx'
    source_model_paths = discover_models(models_root, args.models)
    model_names = list(source_model_paths.keys())
    external_metadata, metadata_path = load_external_model_metadata(args.metadata_csv)
    samples, class_counts = discover_real_test(real_test_root)
    prepared_model_paths = {}
    prep_infos = {}
    print('\nTRUE-BATCH ONNX PREPARATION')
    for i, name in enumerate(model_names, 1):
        print(f'  [{i}/{len(model_names)}] {name}')
        prepared, info = prepare_batch_capable_onnx(source_model_paths[name], prepared_dir)
        prepared_model_paths[name] = prepared
        prep_infos[name] = info
        print(f"      {info['Batch Preparation']} | graph fingerprint equal={info['Compute Graph Fingerprint Equal']}")
    environment = {'timestamp_local': datetime.now().isoformat(timespec='seconds'), 'device': detect_device(), 'hostname': platform.node(), 'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(), 'tensorrt': trt.__version__, 'cuda_runtime_library': CUDART_PATH, 'precision': 'FP16', 'batch_sizes': batch_sizes, 'batching': 'TRUE tensor batching; one dynamic TensorRT profile per model; no sequential pseudo-batching', 'warmup_batches': int(args.warmup), 'model_count': len(model_names), 'metadata_csv': str(metadata_path) if metadata_path else None, 'workspace_gib': args.workspace_gib, 'nvpmodel_query': run_cmd(['/usr/sbin/nvpmodel', '-q'], timeout=5), 'jetson_clocks_show': run_cmd(['/usr/bin/jetson_clocks', '--show'], timeout=5), 'models_root': str(models_root), 'real_test_root': str(real_test_root), 'real_test_total': len(samples), 'real_test_counts': class_counts, 'class_to_idx': CLASS_TO_IDX, 'final_partial_batch_policy': 'repeat last valid image to requested tensor batch; padded outputs discarded', 'eval_transform': {'resize_short_edge': 256, 'resize_semantics': 'torchvision: int(size*long/short)', 'interpolation': 'PIL bilinear', 'center_crop': [224, 224], 'rgb': True, 'scale': 'uint8 / 255.0', 'normalize_mean': IMAGENET_MEAN.reshape(-1).tolist(), 'normalize_std': IMAGENET_STD.reshape(-1).tolist()}}
    print('=' * 118)
    print('JETSON ORIN NANO — CLEAN SCI REAL_TEST TENSORRT FP16 GPU — TRUE BATCH 1,4,8,16')
    print('=' * 118)
    print(f"Device              : {environment['device']}")
    print(f"Python              : {environment['python']}")
    print(f"TensorRT            : {environment['tensorrt']}")
    print(f"CUDA Runtime        : {environment['cuda_runtime_library']}")
    print(f'TRUE batch tests    : {batch_sizes}')
    print(f'Models              : {len(model_names)}')
    print(f'real_test images    : {len(samples)} | {class_counts}')
    print('Eval preprocessing  : Resize(256 truncation) -> CenterCrop(224) -> ImageNet Normalize')
    print('=' * 118)
    manifest_rows = []
    for name in model_names:
        srcp = source_model_paths[name]
        prepp = prepared_model_paths[name]
        mr = {'Model': name}
        mr.update(prep_infos[name])
        mr['Source ONNX Size (MiB)'] = srcp.stat().st_size / 1024 ** 2
        mr['Prepared ONNX Size (MiB)'] = prepp.stat().st_size / 1024 ** 2
        mr.update(inspect_onnx_artifact(srcp))
        attach_external_metadata(mr, name, external_metadata)
        manifest_rows.append(mr)
    pd.DataFrame(manifest_rows).to_csv(output_dir / 'model_manifest.csv', index=False)
    (output_dir / 'environment.json').write_text(json.dumps(environment, indent=2, allow_nan=True), encoding='utf-8')
    print('\n' + '#' * 118)
    print('TENSORRT FP16 DYNAMIC ENGINE PREPARATION')
    print('#' * 118)
    engine_paths, build_rows = ({}, [])
    for model_name in model_names:
        engine_path, build_seconds, build_info = ensure_engine(model_name=model_name, onnx_path=prepared_model_paths[model_name], engine_dir=engine_dir, workspace_gib=args.workspace_gib, rebuild=args.rebuild_engines, batch_sizes=batch_sizes)
        engine_paths[model_name] = engine_path
        build_rows.append({'Model': model_name, 'Source ONNX SHA256': sha256_file(source_model_paths[model_name]), 'Prepared ONNX SHA256': sha256_file(prepared_model_paths[model_name]), 'Engine Path': str(engine_path), 'Engine Size (MiB)': engine_path.stat().st_size / 1024 ** 2, 'Build Time (s)': build_seconds, 'TensorRT Version': trt.__version__, 'Precision': 'FP16', **build_info})
        for b in batch_sizes:
            runner = TensorRTRunner(engine_path, batch_size=b)
            dummy = np.zeros((b, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            y, _, _, _ = runner.run(dummy)
            if y.shape[0] != b:
                runner.close()
                raise RuntimeError(f'{model_name}: TensorRT preflight batch={b} returned {y.shape}')
            runner.close()
        print(f'  {model_name:<28} engine={engine_path.stat().st_size / 1024 ** 2:.3f} MiB | verified B={batch_sizes}')
    pd.DataFrame(build_rows).to_csv(output_dir / 'engine_build_summary.csv', index=False)
    time.sleep(max(10.0, float(args.cooldown_seconds)))
    rows, failures = ([], [])
    prediction_reference = {}
    total_runs = len(batch_sizes) * len(model_names)
    run_no = 0
    for batch_size in batch_sizes:
        print('\n' + '#' * 118)
        print(f'TRUE BATCH CONFIGURATION: B{batch_size}')
        print('#' * 118)
        for model_name in model_names:
            run_no += 1
            print('\n' + '-' * 118)
            print(f'[{run_no}/{total_runs}] {model_name} | TRT FP16 | B{batch_size} | real_test')
            print('-' * 118)
            try:
                row, preds = benchmark_model(model_name=model_name, source_onnx_path=source_model_paths[model_name], prepared_onnx_path=prepared_model_paths[model_name], engine_path=engine_paths[model_name], samples=samples, batch_size=batch_size, warmup=args.warmup, output_dir=output_dir)
                attach_external_metadata(row, model_name, external_metadata)
                ref = prediction_reference.get(model_name)
                if ref is None:
                    prediction_reference[model_name] = preds.copy()
                    disagreement = 0
                else:
                    disagreement = int(np.sum(ref != preds))
                row['Prediction Disagreement vs B1'] = disagreement
                row['Predictions Identical vs B1'] = bool(disagreement == 0)
                rows.append(row)
                print(f"Accuracy={row['Accuracy'] * 100:.2f}% | F1={row['Macro F1']:.6f} | MCC={row['MCC']:.6f} | AUC={row['ROC-AUC Macro OvR']:.6f}")
                print(f"TRT GPU batch mean={row['TRT GPU Mean (ms/batch)']:.4f} ms | amortized={row['Amortized TRT GPU Mean (ms/image)']:.4f} ms/image | throughput={row['TRT Nominal Throughput (images/s)']:.2f} images/s")
                print(f'Prediction disagreement vs B1: {disagreement}')
            except Exception as exc:
                msg = f'{type(exc).__name__}: {exc}'
                print('ERROR:', msg)
                failures.append({'Model': model_name, 'Batch Size': batch_size, 'error': msg})
            pd.DataFrame(rows).to_csv(output_dir / 'jetson_gpu_realtest_sci_trt_fp16_summary_partial.csv', index=False)
            time.sleep(max(0.0, float(args.cooldown_seconds)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_dir / 'jetson_gpu_realtest_sci_trt_fp16_summary.csv', index=False)
    pd.DataFrame(failures).to_csv(output_dir / 'failures.csv', index=False)
    payload = {'environment': environment, 'model_manifest': manifest_rows, 'engine_builds': build_rows, 'results': rows, 'failures': failures}
    (output_dir / 'jetson_gpu_realtest_sci_trt_fp16_summary.json').write_text(json.dumps(payload, indent=2, allow_nan=True), encoding='utf-8')
    print('\n' + '=' * 118)
    print('JETSON GPU CLEAN TRUE-BATCH BENCHMARK COMPLETED')
    print(f'Output   : {output_dir}')
    print(f'Results  : {len(rows)} | Failures: {len(failures)}')
    print('=' * 118)
    if failures:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
