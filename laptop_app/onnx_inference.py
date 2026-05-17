#!/usr/bin/env python3
"""
ONNX inference wrapper for Depth Anything V2 (vits).

Uses the exported ONNX model (from export_onnx.py) instead of PyTorch.
The ONNX model includes preprocessing (resize + pad + normalize).

Usage:
    from onnx_inference import ONNXDepthEngine
    engine = ONNXDepthEngine()
    depth = engine.infer(img_bgr)  # returns (H, W) depth array
"""
import os, sys
from pathlib import Path

DEPTH_PROJECT = os.path.expanduser("~/projects/depth-anything-v2")
CHECKPOINT_DIR = Path(DEPTH_PROJECT) / "checkpoints"

import numpy as np
import cv2
import onnxruntime as ort

# Must match export_onnx.py
ENCODER_TARGET = 392


class ONNXDepthEngine:
    """
    Depth inference via ONNX Runtime (CPU).
    Replaces PyTorch DepthEngine for faster, lower-RAM inference on CPU.
    """

    def __init__(self, encoder='vits'):
        self.encoder = encoder
        self.session = None

    def load_model(self):
        if self.session is None:
            onnx_path = CHECKPOINT_DIR / f"depth_anything_v2_{self.encoder}.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {onnx_path}\n"
                    "Run: python3 export_onnx.py"
                )
            print(f"[onnx] Loading ONNX model from {onnx_path}")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=sess_opts,
                providers=['CPUExecutionProvider']
            )
            print(f"[onnx] Model loaded. Providers: {self.session.get_providers()}")

            # Warm up — first inference is slower (graph optimization)
            dummy = np.zeros((1, 3, 480, 640), dtype=np.uint8)
            self._run(dummy)

    def _run(self, input_uint8: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on a uint8 image.
        input_uint8: [1, 3, H, W] uint8 (any H, W)
        returns: depth [ENCODER_TARGET, ENCODER_TARGET] float32
        """
        return self.session.run(None, {'input': input_uint8})[0][0, 0]

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth inference on a BGR image (e.g., from cv2.imdecode).
        Returns depth map at the original image resolution.

        img_bgr: HxWx3 uint8 BGR (from cv2.imdecode)
        returns: HxW float32 depth map (relative values)
        """
        orig_h, orig_w = img_bgr.shape[:2]

        # Transpose to CHW and add batch dimension
        input_t = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, ...].astype(np.uint8)
        depth_et = self._run(input_t)  # [ENCODER_TARGET, ENCODER_TARGET]

        # Resize depth back to original camera resolution
        depth = cv2.resize(depth_et, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def infer_from_jpg_bytes(self, jpg_bytes: bytes) -> np.ndarray:
        """Convenience: decode JPG → infer → return depth map."""
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed")
        return self.infer(img)


# Singleton
_onnx_engine = None

def get_onnx_engine():
    global _onnx_engine
    if _onnx_engine is None:
        _onnx_engine = ONNXDepthEngine()
    return _onnx_engine


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("[onnx] Testing ONNX inference at ENCODER_TARGET=392...")
    engine = ONNXDepthEngine()
    engine.load_model()

    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Warm up
    engine.infer(dummy_img)

    # Benchmark
    n_runs = 10
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        depth = engine.infer(dummy_img)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"[onnx] Average: {avg*1000:.1f}ms (over {n_runs} runs)")
    print(f"[onnx] Depth output: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]")