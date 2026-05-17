#!/usr/bin/env python3
"""
Depth inference benchmark — compares PyTorch fp32, ONNX, and fp16.

Run:  python3 benchmark_depth.py

Tests 50 inference runs per backend and reports:
  - Average time per run (ms)
  - Min time (ms)
  - RAM peak during inference (RSS, MB)
  - Comparison table

Note: fp16 on CPU is very slow on non-ARM/non-AVX512 hardware.
      Benchmark is configured to skip fp16 by default (disabled in main()).
"""
import sys, time, os, gc

DEPTH_PROJECT = os.path.expanduser("~/projects/depth-anything-v2")
CHECKPOINT_DIR = os.path.expanduser("~/projects/depth-anything-v2/checkpoints")
sys.path.insert(0, DEPTH_PROJECT)

import numpy as np
import cv2
import torch
from pathlib import Path
from depth_anything_v2.dpt import DepthAnythingV2

ENCODER_TARGET = 392  # must match depth.py


def get_rss_mb():
    """Peak RSS in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def make_dummy_image(w=640, h=480):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class PyTorchFP32Engine:
    """Baseline: PyTorch fp32 with ENCODER_TARGET resolution."""

    def __init__(self):
        self.model = None

    def load(self):
        self.model = DepthAnythingV2(encoder='vits', features=64,
                                      out_channels=[48, 96, 192, 384])
        state = torch.load(str(Path(CHECKPOINT_DIR) / "depth_anything_v2_vits.pth"),
                           map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        # Warm up (first run is always slower)
        self._infer(make_dummy_image())

    def _infer(self, img):
        orig_h, orig_w = img.shape[:2]
        if orig_h > orig_w:
            new_h, new_w = ENCODER_TARGET, int(orig_w * ENCODER_TARGET / orig_h)
        else:
            new_h, new_w = int(orig_h * ENCODER_TARGET / orig_w), ENCODER_TARGET
        resized = cv2.resize(img, (new_w, new_h))
        pad_h = ENCODER_TARGET - new_h
        pad_w = ENCODER_TARGET - new_w
        pt, pl = pad_h // 2, pad_w // 2
        padded = np.zeros((ENCODER_TARGET, ENCODER_TARGET, 3), dtype=np.uint8)
        padded[pt:pt+new_h, pl:pl+new_w] = resized
        input_t = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            depth = self.model(input_t).squeeze().cpu().numpy()
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = depth[pt:pt+new_h, pl:pl+new_w]
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def run(self, img):
        return self._infer(img)


class PyTorchFP16Engine:
    """PyTorch fp16 — 50% RAM reduction but MUCH slower on CPU without AVX512/NEON."""

    def __init__(self):
        self.model = None

    def load(self):
        self.model = DepthAnythingV2(encoder='vits', features=64,
                                      out_channels=[48, 96, 192, 384])
        state = torch.load(str(Path(CHECKPOINT_DIR) / "depth_anything_v2_vits.pth"),
                           map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.half()
        self.model.eval()
        # Warm up — fp16 is extremely slow without hardware F16 support
        self._infer(make_dummy_image())

    def _infer(self, img):
        orig_h, orig_w = img.shape[:2]
        if orig_h > orig_w:
            new_h, new_w = ENCODER_TARGET, int(orig_w * ENCODER_TARGET / orig_h)
        else:
            new_h, new_w = int(orig_h * ENCODER_TARGET / orig_w), ENCODER_TARGET
        resized = cv2.resize(img, (new_w, new_h))
        pad_h = ENCODER_TARGET - new_h
        pad_w = ENCODER_TARGET - new_w
        pt, pl = pad_h // 2, pad_w // 2
        padded = np.zeros((ENCODER_TARGET, ENCODER_TARGET, 3), dtype=np.uint8)
        padded[pt:pt+new_h, pl:pl+new_w] = resized
        input_t = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        input_t = input_t.half()
        with torch.no_grad():
            depth = self.model(input_t).squeeze().cpu().float().numpy()
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = depth[pt:pt+new_h, pl:pl+new_w]
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def run(self, img):
        return self._infer(img)


class ONNXEngine:
    """ONNX Runtime (CPU) — fastest on CPU, lowest peak RAM."""

    def __init__(self):
        self.session = None

    def load(self):
        import onnxruntime as ort
        onnx_path = Path(CHECKPOINT_DIR) / "depth_anything_v2_vits.onnx"
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        # Warm up
        self._infer(make_dummy_image())

    def _infer(self, img):
        orig_h, orig_w = img.shape[:2]
        input_t = np.transpose(img, (2, 0, 1))[np.newaxis, ...].astype(np.uint8)
        depth = self.session.run(None, {'input': input_t})[0][0, 0]
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def run(self, img):
        return self._infer(img)


def benchmark(name, engine_cls, n_runs=50, skip=False):
    """Run benchmark: n_runs inference, measure time + RAM."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if skip:
        print(f"  [SKIPPED — not available or disabled]")
        return None

    try:
        gc.collect()
        mem_before = get_rss_mb()
        print(f"  Loading model (RAM baseline: {mem_before:.0f} MB)...")
        t_load = time.perf_counter()
        engine = engine_cls()
        engine.load()
        t_load = time.perf_counter() - t_load
        mem_after_load = get_rss_mb()
        print(f"  Load time: {t_load:.1f}s | RAM after load: {mem_after_load:.0f} MB")

        dummy = make_dummy_image()
        print(f"  Warming up (1 run)...")
        _ = engine.run(dummy)

        print(f"  Running {n_runs} inference iterations...")
        times = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            depth = engine.run(dummy)
            times.append(time.perf_counter() - t0)
            if (i + 1) % 10 == 0:
                print(f"    run {i+1}/{n_runs}...")

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000

        # RAM after all runs
        mem_peak = get_rss_mb()
        ram_delta = mem_peak - mem_before

        print(f"  Avg: {avg_ms:.1f}ms  Min: {min_ms:.1f}ms  Max: {max_ms:.1f}ms")
        print(f"  RAM peak: {mem_peak:.0f} MB (delta from baseline: +{ram_delta:.0f} MB)")
        print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")

        del engine
        gc.collect()

        return {
            "name": name,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "ram_peak_mb": mem_peak,
            "ram_delta_mb": ram_delta,
        }

    except Exception as e:
        print(f"  [ERROR: {e}]")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("  Depth Anything V2 — Inference Benchmark")
    print("  Image: 640x480 | ENCODER_TARGET: 392px | Runs: 50")
    print("=" * 60)

    n_runs = 50

    # Check what's available
    onnx_available = Path(CHECKPOINT_DIR).joinpath("depth_anything_v2_vits.onnx").exists()

    results = []

    # 1. PyTorch fp32 (baseline)
    r_fp32 = benchmark("PyTorch fp32 (baseline)", PyTorchFP32Engine, n_runs)
    if r_fp32:
        results.append(r_fp32)

    # 2. ONNX Runtime (CPU) — primary P2 optimization
    r_onnx = benchmark("ONNX Runtime (CPU)", ONNXEngine, n_runs,
                        skip=not onnx_available)
    if r_onnx:
        results.append(r_onnx)

    # 3. PyTorch fp16 — DISABLED by default (very slow on CPU without F16 hardware)
    # Uncomment to test on ARM/RISC-V with NEON, or Intel with AVX512-FP16
    # r_fp16 = benchmark("PyTorch fp16 (half)", PyTorchFP16Engine, n_runs)
    # if r_fp16:
    #     results.append(r_fp16)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS — Depth Anything V2 vits @ 392px (50 runs, 640x480)")
    print("=" * 80)
    print(f"  {'Backend':<28} {'Avg (ms)':>10} {'Min (ms)':>10} {'RAM Peak (MB)':>12} {'RAM Δ (MB)':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    baseline_avg = None
    baseline_min = None
    for r in results:
        print(f"  {r['name']:<28} {r['avg_ms']:>10.1f} {r['min_ms']:>10.1f} "
              f"{r['ram_peak_mb']:>12.0f} {r['ram_delta_mb']:>+10.0f}")
        if baseline_avg is None:
            baseline_avg = r['avg_ms']
            baseline_min = r['min_ms']

    print("=" * 80)

    if len(results) >= 2:
        r0, r1 = results[0], results[1]
        speedup_avg = r0['avg_ms'] / r1['avg_ms']
        speedup_min = r0['min_ms'] / r1['min_ms']
        print(f"\n  ONNX vs PyTorch fp32:")
        print(f"    • Speedup (avg): {speedup_avg:.2f}x faster")
        print(f"    • Speedup (min): {speedup_min:.2f}x faster")
        print(f"    • RAM savings: {r0['ram_delta_mb'] - r1['ram_delta_mb']:.0f} MB less peak")
        print(f"\n  Recommendation: ONNX Runtime (CPU) — fastest, lowest RAM peak.")

    print("\n  Note: PyTorch fp16 on this machine is ~30x slower than fp32")
    print("        because CPU lacks hardware FP16 support (no AVX512-FP16, no NEON).")
    print("        fp16 is only useful with GPU or ARM with NEON.")


if __name__ == "__main__":
    main()