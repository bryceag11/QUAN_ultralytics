# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Advanced benchmarking for YOLO models with layer-by-layer profiling.

Features:
- PyTorch model benchmarking with zero tensors or custom data
- Dataset validation with mAP metrics reporting
- Layer-by-layer profiling for detailed timing analysis (always prints full table)
- No export dependency

Usage:
    from advanced_benchmark import benchmark_pytorch, profile_layers

    # Basic benchmark
    benchmark_pytorch(model="yolo11n.pt", imgsz=640)

    # Profile individual layers on a batch of images from a dataset YAML
    profile_layers(
        model="yolo11n.pt",
        imgsz=640,
        batch_size=8,
        data="path/to/your_dataset.yaml"
    )
"""

import os
import time
import glob
import yaml
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.cfg import TASK2METRIC
from ultralytics.utils import TQDM
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import select_device


def benchmark_pytorch(
    model="yolo11n.pt",
    data=None,
    imgsz=640,
    half=False,
    int8=False,
    device="cuda",
    verbose=True,
    batch_size=1,
    warmup_iterations=25,
    benchmark_iterations=100,
):
    imgsz = check_imgsz(imgsz)
    device = select_device(device, verbose=False)

    # Load model
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    model_name = Path(model.ckpt_path).stem if hasattr(model, 'ckpt_path') and model.ckpt_path else model.model_name
    model = model.to(device)

    if verbose:
        print(f"Benchmarking {model_name} on {device} device")
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
    use_dataset = False
    if data:
        if verbose:
            print(f"Using dataset: {data}")
        if str(data).endswith(('.yaml', '.yml')):
            use_dataset = True
        elif os.path.isfile(data) and str(data).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                img = Image.open(data).convert('RGB').resize((imgsz, imgsz))
                input_tensor = np.array(img)
                if verbose:
                    print(f"Using image: {data} with shape {input_tensor.shape}")
            except Exception as e:
                print(f"Failed to load image: {e}, falling back to zeros")
                input_tensor = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        else:
            if verbose:
                print(f"Data path {data} is not a supported file type, falling back to zeros")
            input_tensor = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    else:
        input_tensor = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        if verbose:
            print("No data provided, using zero tensors")

    # Warm-up runs
    if verbose:
        print(f"Performing {warmup_iterations} warm-up iterations...")
    for _ in range(warmup_iterations):
        if not use_dataset:
            model.predict(input_tensor, imgsz=imgsz, verbose=False)

    # Benchmark time for dataset if specified
    if use_dataset:
        if verbose:
            print(f"Benchmarking with dataset {data}...")
        start_time = time.time()
        results = model.val(
            data=data,
            batch=batch_size,
            imgsz=imgsz,
            plots=False,
            device=device,
            half=half,
            int8=int8,
            verbose=verbose
        )
        validation_time = time.time() - start_time

        metrics_dict = results.results_dict
        key = TASK2METRIC[model.task]
        main_metric = metrics_dict[key]
        map50 = metrics_dict.get("metrics/mAP50(B)", None)
        map50_95 = metrics_dict.get("metrics/mAP50-95(B)", None)
        inference_time = results.speed.get('inference', 0)
        total_time_per_image = sum(results.speed.values())
        fps = round(1000 / total_time_per_image, 2) if total_time_per_image > 0 else 0

        print("\n" + "=" * 80)
        print(f"PyTorch Benchmark Results for {model_name}")
        print("=" * 80)
        print(f"Dataset     : {Path(data).stem}")
        print(f"Image Size  : {imgsz}")
        print(f"Batch Size  : {batch_size}")
        print(f"Device      : {device}")
        print(f"Speed       : {inference_time} ms/image (inference)")
        print(f"Total Time  : {total_time_per_image} ms/image")
        print(f"Throughput  : {fps} FPS")
        if map50 is not None:
            print(f"mAP@0.5     : {map50:.4f}")
        if map50_95 is not None:
            print(f"mAP@0.5:0.95: {map50_95:.4f}")
        print("=" * 80)

        return {
            "model": model_name,
            "dataset": Path(data).stem,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "device": device,
            "inference_time_ms": inference_time,
            "total_time_ms": total_time_per_image,
            "FPS": fps,
            "accuracy": main_metric,
            "mAP50": map50,
            "mAP50-95": map50_95
        }

    # Benchmark with zero tensors
    if verbose:
        print(f"Benchmarking with {benchmark_iterations} iterations...")
    run_times = []
    for _ in TQDM(range(benchmark_iterations), desc=f"Benchmarking {model_name}"):
        res = model.predict(input_tensor, imgsz=imgsz, verbose=False)
        run_times.append(res[0].speed["inference"])
    run_times = iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)
    mean_time = float(np.mean(run_times))
    std_time = float(np.std(run_times))
    fps = round(1000 / mean_time, 2)

    print("\n" + "=" * 80)
    print(f"PyTorch Benchmark Results for {model_name}")
    print("=" * 80)
    print(f"Dataset     : {'zeros' if data is None else Path(data).stem}")
    print(f"Image Size  : {imgsz}")
    print(f"Device      : {device}")
    print(f"Speed       : {mean_time:.2f} ± {std_time:.2f} ms/image")
    print(f"Throughput  : {fps} FPS")
    print("=" * 80)

    return {
        "model": model_name,
        "dataset": 'zeros' if data is None else Path(data).stem,
        "imgsz": imgsz,
        "device": device,
        "ms_per_image": round(mean_time, 2),
        "ms_std": round(std_time, 2),
        "FPS": fps
    }


def profile_layers(
    model="runs/detect/mar_aism_exps/train77/weights/best.pt",
    data=None, 
    imgsz=640,
    device="cuda",
    batch_size=1,
    iterations=10
):
    """
    Profile each layer of a YOLO model for performance analysis using PyTorch hooks
    """

    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    model = model.to(device)
    model.model.eval()

    img_size = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)

    # Input batch 
    def load_from_yaml(yaml_path):
        cfg = yaml.safe_load(Path(yaml_path).read_text())
        root = Path(cfg.get('path', Path(yaml_path).parent))
        if not root.is_absolute():
            root = (Path(yaml_path).parent / root).resolve()
        imgs = []
        for split in ('train', 'val'):
            val = cfg.get(split)
            if not val:
                continue
            p = Path(val)
            if not p.is_absolute():
                p = (root / p).resolve()
            imgs += [str(f) for f in p.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
        return sorted(imgs)[:batch_size]

    if data and data.lower().endswith(('.yaml', '.yml')):
        files = load_from_yaml(data)
        if files:
            arrs = []
            for fp in files:
                im = Image.open(fp).convert("RGB").resize(img_size)
                arrs.append(np.array(im).transpose(2, 0, 1))  # C,H,W
            batch = np.stack(arrs, axis=0)                   # B,C,H,W
            im = torch.from_numpy(batch).to(device).float() / 255.0
            print(f"Loaded {len(files)} images from {data}")
        else:
            print(f"⚠️ No images found in {data}, falling back to zeros")
            im = torch.zeros(batch_size, 3, *img_size, device=device)
    elif data and os.path.isfile(data):
        base = Image.open(data).convert("RGB").resize(img_size)
        arr  = np.array(base).transpose(2, 0, 1)
        im   = torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0
        im   = im.repeat(batch_size, 1, 1, 1)
        print(f"Profiling on single image {data} ×{batch_size}")
    else:
        print(f"No valid data input, using zero tensor of shape ({batch_size},3,{imgsz},{imgsz})")
        im = torch.zeros(batch_size, 3, *img_size, device=device)

    # Profiling hooks 
    def time_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    layers = list(model.model.model)
    hooks, elapsed = [], {}

    for idx, layer in enumerate(layers):
        def pre_hook(m, inp, idx=idx):
            m._t0 = time_sync()
        hooks.append(layer.register_forward_pre_hook(pre_hook))

        def post_hook(m, inp, out, idx=idx):
            dt = (time_sync() - m._t0) * 1000
            name = f"{idx:03d} {m.__class__.__name__}"
            elapsed.setdefault(name, []).append(dt)
            del m._t0
        hooks.append(layer.register_forward_hook(post_hook))

    # Warm-up & timed runs
    with torch.no_grad():
        _ = model.predict(source=im, verbose=False)
    for _ in range(iterations):
        with torch.no_grad():
            _ = model.predict(source=im, verbose=False)

    for h in hooks:
        h.remove()

    # Table aggregation
    total_time = 0.0
    results = []
    for name, times in elapsed.items():
        avg = sum(times) / len(times)
        total_time += avg
        results.append({"layer": name, "time_ms": avg})

    # Summary
    print(f"\n*** Layer-by-layer summary (batch={batch_size}) ***")
    print(f"  Total time: {total_time:.2f} ms   —→ {1000/total_time:.2f} FPS\n")

    # Full detailed table
    print(f"{'Layer':<20}{'Time (ms)':>10}")
    print("-" * 30)
    for r in results:
        print(f"{r['layer']:<20}{r['time_ms']:>10.2f}")
    print("-" * 30)

    return {
        "model": model.model_name,
        "batch_size": batch_size,
        "layers": results,
        "total_time_ms": total_time,
        "fps": 1000 / total_time
    }


def iterative_sigma_clipping(data, sigma=2, max_iters=3):
    """Applies iterative sigma clipping to data to remove outliers."""
    data = np.array(data)
    for _ in range(max_iters):
        mu, std = data.mean(), data.std()
        clipped = data[(data > mu - sigma * std) & (data < mu + sigma * std)]
        if len(clipped) == len(data):
            break
        data = clipped
    return data
# Example usage when script is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced benchmarking for YOLO models")
    parser.add_argument('--model', type=str, default='runs/detect/train212/weights/best.pt', help='Model path or name')
    parser.add_argument('--data', type=str, default='C:/Users/bag100/ultralytics/ultralytics/cfg/datasets/aism2.yaml', help='Dataset path (yaml or image)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=25, help='Number of warmup iterations')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16)')
    parser.add_argument('--profile-layers', action='store_true', help='Profile individual layers')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if args.profile_layers:
        profile_layers(
            model=args.model,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            batch_size=args.batch_size,
            iterations=args.iterations // 10
        )
    else:
        benchmark_pytorch(
            model=args.model,
            data=args.data,
            imgsz=args.imgsz,
            half=args.half,
            device=args.device,
            verbose=args.verbose,
            batch_size=args.batch_size,
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations
        )
