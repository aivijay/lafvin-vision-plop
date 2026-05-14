#!/usr/bin/env python3
"""
Depth Anything V2 wrapper for laptop-side analysis.
Uses ~/projects/depth-anything-v2/
"""
import os, sys, time, json, base64
import numpy as np
import cv2
from pathlib import Path

DEPTH_PROJECT = os.path.expanduser("~/projects/depth-anything-v2")
sys.path.insert(0, DEPTH_PROJECT)

import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Smooth 4-zone scale-only calibration (no discontinuity)
DEFAULT_CALIB = {
    "thresholds": [2.5, 4.0, 4.3],
    "scales": [0.408, 0.590, 1.000, 0.672]
}
CALIB_PATH = Path(os.path.expanduser("~/.lafvin_depth_calib.json"))


def load_calibration():
    if CALIB_PATH.exists():
        return json.loads(CALIB_PATH.read_text())
    return DEFAULT_CALIB


def save_calibration(calib):
    CALIB_PATH.write_text(json.dumps(calib, indent=2))


def apply_calibration(raw_depth, calib):
    t1, t2, t3 = calib["thresholds"]
    s0, s1, s2, s3 = calib["scales"]
    calibrated = np.zeros_like(raw_depth, dtype=np.float32)
    m0 = raw_depth < t1
    m1 = (raw_depth >= t1) & (raw_depth < t2)
    m2 = (raw_depth >= t2) & (raw_depth < t3)
    m3 = raw_depth >= t3
    calibrated[m0] = raw_depth[m0] * s0
    calibrated[m1] = raw_depth[m1] * s1
    calibrated[m2] = raw_depth[m2] * s2
    calibrated[m3] = raw_depth[m3] * s3
    return np.maximum(calibrated, 0)


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

class DepthEngine:
    def __init__(self, encoder='vits', dataset='hypersim', max_depth=20.0):
        self.model = None
        self.device = "cpu"
        self.encoder = encoder
        self.dataset = dataset
        self.max_depth = max_depth
        self.config = MODEL_CONFIGS[encoder]
        self._warmup_done = False
        self._inference_lock = False
        self._cached_colorized_jpg = b''

    def load_model(self):
        if self.model is None:
            print("[depth] Loading Depth Anything V2 ({}/{})...".format(self.dataset, self.encoder))
            # Use non-metric checkpoint (metric one produces all-zero output on CPU)
            ckpt_path = Path(DEPTH_PROJECT) / 'checkpoints' / f'depth_anything_v2_{self.encoder}.pth'
            self.model = DepthAnythingV2(**self.config)
            self.model.load_state_dict(torch.load(str(ckpt_path), map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("[depth] Model loaded from", ckpt_path)

    def warmup(self):
        if not self._warmup_done:
            dummy = np.zeros((518, 518, 3), dtype=np.uint8)
            self._inference(dummy)
            self._warmup_done = True
            print("[depth] Warmup done.")

    def _inference(self, img_bgr) -> np.ndarray:
        orig_h, orig_w = img_bgr.shape[:2]

        # ── Subsample to ~320px on longer side for faster inference ──
        # Depth Anything was trained on 518px; going to 320px is ~62% the pixels
        # but still produces good enough depth for navigation decisions.
        # We keep the original dimensions separate and resize back at the end.
        target_long = 320
        if max(orig_h, orig_w) > target_long:
            if orig_h > orig_w:
                sub_h, sub_w = target_long, int(orig_w * target_long / orig_h)
            else:
                sub_h, sub_w = int(orig_h * target_long / orig_w), target_long
            img_sub = cv2.resize(img_bgr, (sub_w, sub_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_sub = img_bgr
            sub_h, sub_w = orig_h, orig_w

        # Letterbox resize to 518x518 preserving aspect ratio
        target = 518
        if sub_h > sub_w:
            new_h, new_w = target, int(sub_w * target / sub_h)
        else:
            new_h, new_w = int(sub_h * target / sub_w), target
        resized = cv2.resize(img_sub, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to square 518x518
        input_ = np.zeros((target, target, 3), dtype=np.uint8)
        y_off = (target - new_h) // 2
        x_off = (target - new_w) // 2
        input_[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        # Inference
        input_t = torch.from_numpy(input_).permute(2, 0, 1).float() / 255.0
        input_t = input_t.unsqueeze(0)
        with torch.no_grad():
            depth = self.model(input_t)
        depth = depth.squeeze().cpu().numpy()
        # Crop back to cropped region (remove padding)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = depth[y_off:y_off+new_h, x_off:x_off+new_w]
        # Resize back to original camera resolution
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def analyze(self, jpg_bytes: bytes) -> dict:
        """Full analysis: depth + nav planning."""
        self.load_model()
        self.warmup()

        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "decode_failed"}

        h, w = img.shape[:2]

        # Raw → calibrated
        raw_depth = self._inference(img)
        calib = load_calibration()
        depth_cal = apply_calibration(raw_depth, calib)

        # Nav: center strip, bottom half (floor area)
        cx1, cx2 = w // 3, 2 * w // 3
        center = depth_cal[:, cx1:cx2]
        floor = center[h // 2:, :]

        # Filter valid floor pixels
        valid = floor[floor > 0.05]
        if len(valid) == 0:
            return {
                "depth_m": 0.0, "clear_path": True, "obstacle_detected": False,
                "suggested_action": "forward", "obstacle_pct": 0.0,
                "distance_to_obstacle_cm": 500.0,
                "calibration": calib, "img_h": h, "img_w": w
            }

        median_depth = float(np.median(valid))
        mean_depth = float(np.mean(valid))

        # Obstacle: floor pixels < 0.5m (near = obstacle)
        obstacle_mask = floor < 0.5
        obstacle_pct = float(np.sum(obstacle_mask) / floor.size)

        # Column-wise analysis for turn decision
        col_means = floor.mean(axis=0)  # 1D array per column
        threshold_depth = 0.8  # meters
        near_cols = np.sum(col_means < threshold_depth)
        total_cols = len(col_means)
        near_pct = near_cols / total_cols if total_cols > 0 else 0

        # Decide action
        if obstacle_pct > 0.20 or median_depth < 0.5:
            action = "stop"
            clear = False
        elif obstacle_pct > 0.10 or near_pct > 0.15:
            # Narrow passage — turn toward clearer side
            left_half = col_means[:len(col_means)//2]
            right_half = col_means[len(col_means)//2:]
            left_clear = np.sum(left_half > threshold_depth)
            right_clear = np.sum(right_half > threshold_depth)
            action = "turn_left" if left_clear > right_clear else "turn_right"
            clear = False
        else:
            action = "forward"
            clear = True

        # Build depth color map (for display)
        depth_color = self._colorize_depth(raw_depth, w, h)

        # Store colorized JPEG in cache for /depth/colorized.jpg endpoint (fast read, no inference)
        self._colorize_depth_to_jpg(raw_depth, w, h)

        return {
            "depth_m": round(median_depth, 3),
            "mean_depth_m": round(mean_depth, 3),
            "clear_path": clear,
            "obstacle_detected": not clear,
            "obstacle_pct": round(obstacle_pct, 3),
            "suggested_action": action,
            "distance_to_obstacle_cm": round(median_depth * 100, 1),
            "near_pct": round(near_pct, 3),
            "calibration": calib,
            "img_h": h,
            "img_w": w,
        }

    def _colorize_depth(self, raw_depth, w, h) -> list:
        """Colorize depth to rainbow RGBA list for frontend canvas rendering."""
        # raw_depth min/max for normalization
        valid = raw_depth[raw_depth > 0.05]
        min_d = float(np.min(valid)) if len(valid) > 0 else 0.0
        max_d = float(np.max(valid)) if len(valid) > 0 else 5.0
        if max_d <= min_d:
            max_d = min_d + 5.0

        # 8-stop rainbow
        stops = [
            (255, 0, 0), (255, 128, 0), (255, 255, 0),
            (0, 255, 0), (0, 255, 255), (0, 0, 255),
            (128, 0, 255), (0, 0, 0)
        ]

        def interp(t):
            t = max(0, min(1, t))
            scaled = t * (len(stops) - 1)
            i = int(scaled)
            f = scaled - i
            i = min(i, len(stops) - 2)
            r = int(stops[i][0] * (1-f) + stops[i+1][0] * f)
            g = int(stops[i][1] * (1-f) + stops[i+1][1] * f)
            b = int(stops[i][2] * (1-f) + stops[i+1][2] * f)
            return (r, g, b)

        # Sample every 4th row for speed, full width
        step = 4
        pixels = []
        for y in range(0, h, step):
            row = []
            for x in range(w):
                v = raw_depth[y, x]
                t = (v - min_d) / (max_d - min_d) if max_d > min_d else 1.0
                if v <= 0.05 or v > 20:
                    t = 1.0
                row.append(interp(t))
            pixels.append(row)
        return pixels

    def get_colorized_depth_jpg(self, jpg_bytes: bytes) -> bytes:
        """Return cached colorized JPEG. Inference runs once in background_depth_loop."""
        return self._cached_colorized_jpg or b''

    def _colorize_depth_to_jpg(self, raw_depth, w, h) -> bytes:
        """Colorize raw depth array to JPEG, cache result. No inference."""
        valid = raw_depth[raw_depth > 0.05]
        min_d = float(np.min(valid)) if len(valid) > 0 else 0.0
        max_d = float(np.max(valid)) if len(valid) > 0 else 5.0
        if max_d <= min_d:
            max_d = min_d + 5.0

        palette = self._build_palette()

        t = np.clip((raw_depth - min_d) / (max_d - min_d + 1e-8), 0, 1)
        t_flat = (t * 255).astype(np.uint8).ravel()
        flat = palette[t_flat]
        colorized = flat.reshape(h, w, 3)

        ret, buf = cv2.imencode('.jpg', colorized, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self._cached_colorized_jpg = bytes(buf) if ret else b''
        return self._cached_colorized_jpg

    def _build_palette(self):
        """Build the 256-entry BGR color palette once and cache it."""
        if hasattr(self, '_palette_cache') and self._palette_cache is not None:
            return self._palette_cache
        # 7-stop color table (BGR order for OpenCV): BLACK, BLUE, CYAN, GREEN, YELLOW, ORANGE, RED
        color_table = np.array([
            [0,   0,   0],    # BLACK — far
            [255, 0,   0],    # BLUE
            [255, 255, 0],    # CYAN
            [0,   255, 0],    # GREEN
            [0,   255, 255],  # YELLOW
            [0,   165, 255],  # ORANGE
            [0,   0,   255],  # RED — near
        ], dtype=np.uint8)
        indices = np.linspace(0, 255, 7).astype(int)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for i in range(7 - 1):
            s, e = indices[i], indices[i+1]
            f = np.arange(e - s + 1) / (e - s)
            palette[s:e+1] = (color_table[i] * (1-f[:, None]) + color_table[i+1] * f[:, None]).astype(np.uint8)
        self._palette_cache = palette
        return palette


_depth_engine = None

def get_depth_engine():
    global _depth_engine
    if _depth_engine is None:
        _depth_engine = DepthEngine()
    return _depth_engine