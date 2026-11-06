"""PyTorch Jetson GPU benchmark (8 models, FP32/FP16, batch sizes 1/4/8/16).

Reuses the seed-42 architecture and evaluation preprocessing from the supplied
training/benchmark scripts. No training, no conversion of checkpoint files.

Critical timing distinction: this script evaluates the real_test images, rather
than repeatedly timing one representative tensor as in the earlier PyTorch B1
script. GPU CUDA Event forward timings exclude preprocessing and H2D transfers.
Its results are therefore not identical in protocol to the old PyTorch B1 CSV."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
try:
    import timm
except ImportError:
    timm = None
MODEL_SPECS = {'MobileNetV4': 'mobilenetv4_conv_small', 'RepViT': 'repvit_m0_9', 'SHViT': 'shvit_s1', 'StarNet': 'starnet_s1', 'FastViT': 'fastvit_t8', 'GhostNetV2': 'ghostnetv2_100', 'EdgeNeXt': 'edgenext_xx_small'}
MODELS = list(MODEL_SPECS) + ['NANO-SDRN']
SEED = 42
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class SEBlock(nn.Module):

    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class HybridDomainNormV11(nn.Module):
    """
    Full-V11 IBN-style domain normalization.

    64-channel stage:
        IN 50% + BN 50%

    128-channel stage:
        IN 25% + BN 75%
    """

    def __init__(self, channels, instance_ratio=0.5):
        super().__init__()
        in_channels = int(round(channels * float(instance_ratio)))
        in_channels = max(1, min(channels - 1, in_channels))
        bn_channels = channels - in_channels
        self.in_channels = in_channels
        self.inorm = nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=False, eps=1e-05)
        self.bnorm = nn.BatchNorm2d(bn_channels)

    def forward(self, x):
        x_in, x_bn = torch.split(x, [self.in_channels, x.shape[1] - self.in_channels], dim=1)
        x_in = self.inorm(x_in)
        x_bn = self.bnorm(x_bn)
        return torch.cat([x_in, x_bn], dim=1)

class DSConvV11Block(nn.Module):
    """Depthwise-separable Conv -> BN -> Hardswish -> PWConv -> BN -> Hardswish."""

    def __init__(self, in_channels, out_channels, kernel_size=3, use_se=True, se_ratio=8):
        super().__init__()
        if kernel_size not in (3, 5):
            raise ValueError('kernel_size must be 3 or 5')
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Hardswish(inplace=True)
        self.se = SEBlock(out_channels, ratio=se_ratio) if use_se else nn.Identity()

    def forward(self, x):
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return self.se(x)

class SelectiveStandardConvV11Block(nn.Module):
    """Full 3x3 convolution retained only at the important stage transition."""

    def __init__(self, in_channels, out_channels, use_se=True, se_ratio=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish(inplace=True)
        self.se = SEBlock(out_channels, ratio=se_ratio) if use_se else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.se(x)

class SpatialRefineV11(nn.Module):
    """
    V11 lesion-refinement module:
        DW5x5 -> BN -> Hardswish -> optional SE
    """

    def __init__(self, channels=128, use_se=True):
        super().__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False), nn.BatchNorm2d(channels), nn.Hardswish(inplace=True))
        self.se = SEBlock(channels, ratio=8) if use_se else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        return self.se(x)

class NanoSDRNV11AblationCNN(nn.Module):
    """
    Configurable V11 implementation used for the complete ablation study.

    Full V11 hierarchy
    ------------------
    224 -> 112:
        Conv 3->32, stride 2

    112 -> 56:
        DSConv 32->32
        MaxPool

    56 -> 28:
        DSConv 32->64 + SE
        DSConv 64->64 k5 + SE
        Hybrid IN+BN @64
        MaxPool

    28x28 texture stage:
        StandardConv 64->128 + SE
        DSConv 128->128 + SE
        Hybrid IN+BN @128
        DW5 lesion refinement + SE
        save T: 28x28x128

    14x14 semantic stage:
        MaxPool
        StandardConv 128->288 + SE
        DSConv 288->288 + SE
        save S: 14x14x288

    7x7 global stage:
        MaxPool
        DSConv 288->384 + SE
        DSConv 384->384 + SE
        save G: 7x7x384

    Full V11 pooled fusion:
        (GAP+GMP)(T) = 256
        (GAP+GMP)(S) = 576
        (GAP+GMP)(G) = 768
        concat = 1600
        classifier 1600 -> 160 -> classes
    """

    def __init__(self, num_classes, dropout=0.2, use_hybrid_norm=True, use_tri_scale=True, use_spatial_refine=True, use_gmp=True, use_se=True):
        super().__init__()
        self.use_tri_scale = bool(use_tri_scale)
        self.use_gmp = bool(use_gmp)
        self.stem = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.Hardswish(inplace=True))
        self.stage1 = nn.Sequential(DSConvV11Block(32, 32, kernel_size=3, use_se=False), nn.MaxPool2d(2))
        norm64 = HybridDomainNormV11(64, instance_ratio=0.5) if use_hybrid_norm else nn.BatchNorm2d(64)
        self.stage2 = nn.Sequential(DSConvV11Block(32, 64, kernel_size=3, use_se=use_se, se_ratio=8), DSConvV11Block(64, 64, kernel_size=5, use_se=use_se, se_ratio=8), norm64, nn.MaxPool2d(2))
        norm128 = HybridDomainNormV11(128, instance_ratio=0.25) if use_hybrid_norm else nn.BatchNorm2d(128)
        spatial_refine = SpatialRefineV11(128, use_se=use_se) if use_spatial_refine else nn.Identity()
        self.texture_stage = nn.Sequential(SelectiveStandardConvV11Block(64, 128, use_se=use_se, se_ratio=8), DSConvV11Block(128, 128, kernel_size=3, use_se=use_se, se_ratio=8), norm128, spatial_refine)
        self.texture_to_semantic_pool = nn.MaxPool2d(2)
        self.semantic_stage = nn.Sequential(SelectiveStandardConvV11Block(128, 288, use_se=use_se, se_ratio=4), DSConvV11Block(288, 288, kernel_size=3, use_se=use_se, se_ratio=4))
        self.semantic_to_global_pool = nn.MaxPool2d(2)
        self.global_stage = nn.Sequential(DSConvV11Block(288, 384, kernel_size=3, use_se=use_se, se_ratio=4), DSConvV11Block(384, 384, kernel_size=3, use_se=use_se, se_ratio=4))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if self.use_tri_scale:
            pooled_channels = 128 + 288 + 384
        else:
            pooled_channels = 384
        classifier_in = pooled_channels * (2 if self.use_gmp else 1)
        self.classifier = nn.Sequential(nn.Linear(classifier_in, 160), nn.Hardswish(inplace=True), nn.Dropout(float(dropout)), nn.Linear(160, num_classes))

    def _pool_vector(self, x):
        avg_vec = self.avg_pool(x).flatten(1)
        if not self.use_gmp:
            return avg_vec
        max_vec = self.max_pool(x).flatten(1)
        return torch.cat([avg_vec, max_vec], dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        texture = self.texture_stage(x)
        semantic = self.texture_to_semantic_pool(texture)
        semantic = self.semantic_stage(semantic)
        global_feat = self.semantic_to_global_pool(semantic)
        global_feat = self.global_stage(global_feat)
        if self.use_tri_scale:
            fused = torch.cat([self._pool_vector(texture), self._pool_vector(semantic), self._pool_vector(global_feat)], dim=1)
        else:
            fused = self._pool_vector(global_feat)
        return self.classifier(fused)

def build_model(name: str, num_classes: int) -> nn.Module:
    if name == 'NANO-SDRN':
        return NanoSDRNV11AblationCNN(num_classes=num_classes, dropout=0.2, use_hybrid_norm=True, use_tri_scale=False, use_spatial_refine=True, use_gmp=True, use_se=True)
    if timm is None:
        raise RuntimeError("timm eksik. Mevcut Jetson PyTorch ortaminda 'pip install timm' gerekli.")
    return timm.create_model(MODEL_SPECS[name], pretrained=False, num_classes=num_classes)

def args_parser():
    home = Path.home()
    parser = argparse.ArgumentParser(description='Jetson GPU: PyTorch FP32/FP16, 8 model, gercek B1/B4/B8/B16')
    parser.add_argument('--weights-dir', type=Path, default=home / 'Desktop/modeller_8')
    parser.add_argument('--real-test', type=Path, default=home / 'Desktop/dataset/real_test')
    parser.add_argument('--output-dir', type=Path, default=home / 'Desktop/pytorch_gpu_B1_B4_B8_B16')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--precisions', nargs='+', choices=['FP32', 'FP16'], default=['FP32', 'FP16'])
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--min-seconds', type=float, default=5.0, help='Her konfigürasyonda en az bu sure boyunca tam test gecisi tekrarlanir. 0=tek gecis.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--check-only', action='store_true', help='Yalnizca agirlik ve veri yollarini kontrol et')
    return parser.parse_args()

def signature(name: str) -> list[str]:
    if name == 'NANO-SDRN':
        return ['nanosdrn', 'nanosdrv11notriscale', 'nanosdr']
    return [re.sub('[^a-z0-9]', '', name.lower())]

def normalize(text: str) -> str:
    return re.sub('[^a-z0-9]', '', text.lower())

def locate_checkpoint(root: Path, name: str) -> Path:
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.pth', '.pt', '.ckpt'}]
    aliases = signature(name)
    candidates = [p for p in files if any((normalize(p.stem).startswith(k + 'seed42bestrealvalidation') for k in aliases))]
    if not candidates:
        candidates = [p for p in files if any((k in normalize(p.stem) or k in normalize(p.parent.name) for k in aliases)) and 'seed42' in normalize(str(p.relative_to(root)))]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError('Bir model icin birden fazla seed-42 agirligi bulundu. Fazlaliklari ayirin:\n' + '\n'.join((str(p) for p in candidates)))
    raise FileNotFoundError(f'{name}: seed-42 agirlik dosyasi bulunamadi: {root}. Dosya adini veya alt klasorleri kontrol edin.')

def read_state_dict(path: Path) -> dict:
    try:
        container = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        container = torch.load(path, map_location='cpu')
    if isinstance(container, dict) and container and all((torch.is_tensor(v) for v in container.values())):
        return container
    if isinstance(container, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            obj = container.get(key)
            if isinstance(obj, dict) and obj and all((torch.is_tensor(v) for v in obj.values())):
                return obj
    raise ValueError(f'Checkpoint icinde state_dict bulunamadi: {path}')

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def prepare():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision('highest')
    except Exception:
        pass

def build_dataset(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f'real_test klasoru bulunamadi: {root}')
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    data = datasets.ImageFolder(str(root), transform=tf)
    if len(data) < 1:
        raise RuntimeError(f'real_test bos: {root}')
    return data

def padded_batch(x, target_size: int):
    """Match TensorRT: repeat the last valid image; exclude copies from accuracy."""
    actual = x.shape[0]
    if actual == target_size:
        return (x, actual)
    return (torch.cat((x, x[-1:].expand(target_size - actual, -1, -1, -1)), dim=0), actual)

def pct(data, p):
    return float(np.percentile(np.asarray(data, dtype=np.float64), p))

def dump_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def benchmark_one(model, data, name, precision, batch, args, out, ckpt):
    dtype = torch.float32 if precision == 'FP32' else torch.float16
    model.eval()
    model = model.to(device='cuda:0', dtype=dtype)
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_floating_point() and tensor.dtype != dtype:
            raise RuntimeError(f'{name} floating tensor dtype mismatch: {tensor.dtype}')
    loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False, persistent_workers=args.workers > 0)
    first_x, _ = next(iter(loader))
    first_x, _ = padded_batch(first_x, batch)
    x_cuda = first_x.to(device='cuda:0', dtype=dtype, non_blocking=False)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x_cuda)
        torch.cuda.synchronize()
    batch_times_ms = []
    y_true, y_pred = ([], [])
    repeats = 0
    start_clock = time.perf_counter()
    try:
        with torch.inference_mode():
            while True:
                repeats += 1
                for xb, yb in loader:
                    xb, valid_n = padded_batch(xb, batch)
                    xb = xb.to(device='cuda:0', dtype=dtype, non_blocking=False)
                    torch.cuda.synchronize()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    logits = model(xb)
                    end.record()
                    end.synchronize()
                    ms = float(begin.elapsed_time(end))
                    if not math.isfinite(ms) or ms <= 0:
                        raise RuntimeError(f'Gecersiz CUDA event latency: {ms}')
                    batch_times_ms.append(ms)
                    if repeats == 1:
                        yp = logits[:valid_n].float().argmax(dim=1).cpu().tolist()
                        y_pred.extend(yp)
                        y_true.extend(yb.tolist())
                if args.min_seconds <= 0 or time.perf_counter() - start_clock >= args.min_seconds:
                    break
    finally:
        pass
    if len(y_true) != len(data):
        raise RuntimeError(f'Classification sample count mismatch: {len(y_true)} vs {len(data)}')
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(data.classes))), average='macro', zero_division=0)
    tm = float(np.mean(batch_times_ms))
    row = {'Model': name, 'Backend': 'PyTorch', 'Precision': precision, 'Batch': batch, 'Seed': SEED, 'Real-test images': len(data), 'Dataset passes': repeats, 'Timed batches': len(batch_times_ms), 'GPU mean batch (ms)': tm, 'GPU P50 batch (ms)': pct(batch_times_ms, 50), 'GPU P95 batch (ms)': pct(batch_times_ms, 95), 'GPU P99 batch (ms)': pct(batch_times_ms, 99), 'GPU amortized mean (ms/image)': tm / batch, 'GPU nominal throughput (img/s)': 1000 * batch / tm, 'Accuracy (%)': 100 * float(accuracy_score(y_true, y_pred)), 'Macro-F1 (%)': 100 * float(f1), 'Checkpoint': str(ckpt), 'Checkpoint SHA256': file_sha256(ckpt), 'Method': 'CUDA Event GPU forward only; preprocessing/H2D excluded; last batch padded'}
    dump_csv(out / 'gpu_forward_latency_samples.csv', [{'batch_gpu_ms': round(v, 7)} for v in batch_times_ms])
    (out / 'summary.json').write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False, default=str) if all((not isinstance(v, float) or math.isfinite(v) for v in row.values())) else json.dumps({k: None if isinstance(v, float) and (not math.isfinite(v)) else v for k, v in row.items()}, ensure_ascii=False, indent=2), encoding='utf-8')
    dump_csv(out / 'summary.csv', [row])
    return row

def main():
    a = args_parser()
    a.weights_dir = a.weights_dir.expanduser().resolve()
    a.real_test = a.real_test.expanduser().resolve()
    a.output_dir = a.output_dir.expanduser().resolve()
    if len(set(a.batches)) != len(a.batches) or any((b < 1 for b in a.batches)):
        raise ValueError('Batch boyutlari pozitif ve benzersiz olmali')
    if a.warmup < 0 or a.min_seconds < 0 or a.workers < 0:
        raise ValueError('Warmup, sure ve worker negatif olamaz')
    if not a.weights_dir.is_dir():
        raise FileNotFoundError(f'Masaustundeki model klasoru bulunamadi: {a.weights_dir}')
    weights = {m: locate_checkpoint(a.weights_dir, m) for m in a.models}
    for model, fn in weights.items():
        print(f'[CHECKPOINT] {model:14s}: {fn}', flush=True)
    data = build_dataset(a.real_test)
    print(f'[DATASET] {len(data)} goruntu; siniflar={data.classes}', flush=True)
    if len(data.classes) != 3:
        raise ValueError('Model checkpointleri uc sinif icin egitilmisti. Dataset 3 sinif icermeli.')
    if a.check_only:
        print('Yollar hazir. --check-only kaldirilarak benchmark baslatilabilir.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU gorunmuyor. Jetson uzerinde CUDA etkin PyTorch ortamini acin.')
    prepare()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    env = {'script': Path(__file__).name, 'date_local': time.strftime('%Y-%m-%dT%H:%M:%S'), 'platform': platform.platform(), 'python': sys.version, 'pytorch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': getattr(timm, '__version__', None), 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'gpu_available': torch.cuda.is_available(), 'precision': a.precisions, 'batch_sizes': a.batches, 'class_to_idx': data.class_to_idx, 'real_test_images': len(data), 'seed': SEED, 'warmup': a.warmup, 'min_timed_wall_seconds': a.min_seconds, 'weights_dir': str(a.weights_dir), 'real_test_dir': str(a.real_test), 'image_transform': 'Resize(256), CenterCrop(224), ToTensor(), ImageNet normalize', 'precision_policy': 'model.float/model.half (no autocast), TF32 disabled', 'timing': 'CUDA Event GPU forward only; no preprocessing, H2D, disk I/O, or end-to-end', 'padding': 'repeat last image to full batch; padded outputs excluded from classification'}
    (a.output_dir / 'environment.json').write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8')
    summary, errors = ([], [])
    for n, name in enumerate(a.models, 1):
        for precision in a.precisions:
            print(f'\n[{n}/{len(a.models)}] {name} {precision}', flush=True)
            try:
                gc.collect()
                torch.cuda.empty_cache()
                model = build_model(name, num_classes=len(data.classes))
                state = read_state_dict(weights[name])
                model.load_state_dict(state, strict=True)
                del state
                model.eval()
                for batch in a.batches:
                    out = a.output_dir / name / precision / f'B{batch}'
                    out.mkdir(parents=True, exist_ok=True)
                    print(f'  B{batch} ...', flush=True)
                    try:
                        row = benchmark_one(model, data, name, precision, batch, a, out, weights[name])
                        summary.append(row)
                        print(f"    GPU {row['GPU mean batch (ms)']:.4f} ms/batch | {row['GPU nominal throughput (img/s)']:.2f} img/s | Acc {row['Accuracy (%)']:.2f}%", flush=True)
                    except (RuntimeError, MemoryError) as exc:
                        if 'out of memory' in str(exc).lower():
                            torch.cuda.empty_cache()
                        errors.append({'Model': name, 'Precision': precision, 'Batch': batch, 'error': str(exc)})
                        print(f'    HATA {type(exc).__name__}: {exc}', flush=True)
                    finally:
                        dump_csv(a.output_dir / 'PYTORCH_GPU_SCI_SUMMARY.csv', summary)
                        dump_csv(a.output_dir / 'failures.csv', errors)
                del model
                torch.cuda.empty_cache()
            except Exception as exc:
                msg = f'{type(exc).__name__}: {exc}'
                print(f'  MODEL YUKLEME HATASI: {msg}', flush=True)
                errors.extend(({'Model': name, 'Precision': precision, 'Batch': b, 'error': msg} for b in a.batches))
                dump_csv(a.output_dir / 'failures.csv', errors)
    print(f'\nTAMAMLANDI: {len(summary)} basarili, {len(errors)} basarisiz konfig. ')
    print('Ana CSV:', a.output_dir / 'PYTORCH_GPU_SCI_SUMMARY.csv')
    if errors:
        print('Hatalar:', a.output_dir / 'failures.csv')
    if not summary:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
