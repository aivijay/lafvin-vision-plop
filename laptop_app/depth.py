#!/usr/bin/env python3
"""
Depth Anything V2 wrapper for laptop-side analysis.
Uses ~/projects/depth-anything-v2/

Supports three inference backends (priority order):
  1. ONNX Runtime (CPU)     — fastest, ~30MB less RAM than PyTorch fp32
  2. PyTorch fp16 (CPU)    — 50% RAM reduction vs fp32, minimal quality loss
  3. PyTorch fp32 (CPU)    — baseline (original)

Merger note (2026-05-16): v1.0 Buddy changes merged in — floor analysis
uses lower_floor (bottom 1/4 of bottom-half), col_mins instead of col_means,
and history-based distance smoothing.
"""
import os, sys, time, json, base64
import numpy as np
import cv2
from pathlib import Path

DEPTH_PROJECT = os.path.expanduser("~/projects/depth-anything-v2")
sys.path.insert(0, DEPTH_PROJECT)

import torch
from depth_anything_v2.dpt import DepthAnythingV2

# ── Threading ──────────────────────────────────────────────────────────────────
# Constrain PyTorch to available cores — prevents thread oversubscription
_torch_threads = max(1, __import__('os').cpu_count() or 4)
torch.set_num_threads(_torch_threads)
try:
    torch.set_num_interop_threads(_torch_threads)
except AttributeError:
    pass  # torch < 2.0 may not have set_num_interop_threads

# ── Constants ──────────────────────────────────────────────────────────────────
# ENCODER_TARGET must be divisible by 14 (ViT patch size).
# 392 = 28×14 — official DA-V2 size, valid for both PyTorch and ONNX paths.
ENCODER_TARGET = 392

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# ── Calibration ──────────────────────────────────────────────────────────────────
CALIB_PATH = Path(os.path.expanduser("~/.lafvin_depth_calib.json"))

def load_calibration():
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            return json.load(f)
    return None

def save_calibration(calib):
    with open(CALIB_PATH, 'w') as f:
        json.dump(calib, f)

def apply_calibration(raw_depth, calib):
    if calib is None:
        return raw_depth
    scale = calib.get('scale', 1.0)
    offset = calib.get('offset', 0.0)
    return raw_depth * scale + offset

# ── ONNX Backend ───────────────────────────────────────────────────────────────
_CHECKPOINT_DIR = Path(DEPTH_PROJECT) / "checkpoints"

class _ONNXEngine:
    """Lightweight ONNX Runtime wrapper. Separate from the main DepthEngine."""

    def __init__(self, encoder='vits'):
        self.encoder = encoder
        self.session = None

    def load(self):
        if self.session is None:
            onnx_path = _CHECKPOINT_DIR / f"depth_anything_v2_{self.encoder}.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {onnx_path}\n"
                    "Run: python3 export_onnx.py"
                )
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=opts,
                providers=['CPUExecutionProvider']
            )
            # Warm up
            dummy = np.zeros((1, 3, 480, 640), dtype=np.uint8)
            self.session.run(None, {'input': dummy})

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference. Input: HxWx3 uint8 BGR. Output: HxW float32.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        input_t = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, ...].astype(np.uint8)
        # ONNX outputs [1, 1, ENCODER_TARGET, ENCODER_TARGET]
        depth = self.session.run(None, {'input': input_t})[0][0, 0]  # [ET, ET]

        # Letterbox crop: same as PyTorch path — remove padding before resize
        if orig_h > orig_w:
            new_h, new_w = ENCODER_TARGET, int(orig_w * ENCODER_TARGET / orig_h)
        else:
            new_h, new_w = int(orig_h * ENCODER_TARGET / orig_w), ENCODER_TARGET
        pad_h = ENCODER_TARGET - new_h
        pt = pad_h // 2
        depth = depth[pt:pt+new_h, :]  # crop padded rows

        # Resize cropped depth back to original resolution
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth


# ── Main DepthEngine ───────────────────────────────────────────────────────────
class DepthEngine:
    """
    Depth Anything V2 — multi-backend inference.
    Auto-selects ONNX → PyTorch fp32 → PyTorch fp16.
    Calibration is cached after first call (avoids redundant JSON reads).
    """

    def __init__(self, encoder='vits', dataset='hypersim', use_fp16=False):
        self.encoder = encoder
        self.dataset = dataset
        self.use_fp16 = use_fp16
        self.model = None
        self.session = None
        self._onnx = None
        self._calib = None
        self._cached_colorized_jpg = b''
        self._depth_dist_history = []  # max 3 recent distance_to_obstacle_cm readings
        self._build_colorize_lut()
        self.config = MODEL_CONFIGS[encoder]

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_model(self):
        if self.model is not None:
            return

        # Try ONNX first
        onnx_path = _CHECKPOINT_DIR / f"depth_anything_v2_{self.encoder}.onnx"
        if onnx_path.exists():
            try:
                self._onnx = _ONNXEngine(self.encoder)
                self._onnx.load()
                self.session = self._onnx
                print("[depth] ONNX Runtime loaded OK")
                return
            except Exception as e:
                print(f"[depth] ONNX failed ({e}), falling back to PyTorch")

        # PyTorch fp32 baseline
        self.model = DepthAnythingV2(**self.config)
        ckpt = _CHECKPOINT_DIR / f"depth_anything_v2_{self.encoder}.pth"
        state = torch.load(str(ckpt), map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()
            print("[depth] PyTorch fp16 loaded")
        else:
            print("[depth] PyTorch fp32 loaded")

    # ── Inference ────────────────────────────────────────────────────────────────

    def _inference_pytorch(self, img_bgr) -> np.ndarray:
        """
        PyTorch inference path: letterbox resize to ENCODER_TARGET + inference.
        Works with fp32 (float32) and fp16 (float16) models.
        """
        orig_h, orig_w = img_bgr.shape[:2]

        # Letterbox resize: longer side → ENCODER_TARGET
        if orig_h > orig_w:
            new_h, new_w = ENCODER_TARGET, int(orig_w * ENCODER_TARGET / orig_h)
        else:
            new_h, new_w = int(orig_h * ENCODER_TARGET / orig_w), ENCODER_TARGET

        resized = cv2.resize(img_bgr, (new_w, new_h))

        # Pad to ENCODER_TARGET × ENCODER_TARGET (center pad)
        pad_h = ENCODER_TARGET - new_h
        pad_w = ENCODER_TARGET - new_w
        pt, pl = pad_h // 2, pad_w // 2

        padded = np.zeros((ENCODER_TARGET, ENCODER_TARGET, 3), dtype=np.uint8)
        padded[pt:pt+new_h, pl:pl+new_w] = resized

        input_t = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        if self.use_fp16:
            input_t = input_t.half()

        with torch.no_grad():
            depth = self.model(input_t).squeeze().cpu().numpy()

        # Remove padding + resize back
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth = depth[pt:pt+new_h, pl:pl+new_w]
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def _inference(self, img_bgr) -> np.ndarray:
        """Run inference on the selected backend."""
        if self.session is not None:
            return self.session.infer(img_bgr)
        return self._inference_pytorch(img_bgr)

    def warmup(self):
        """Prime the model (first inference is always slow)."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._inference(dummy)

    # ── Analysis ────────────────────────────────────────────────────────────────

    def analyze(self, jpg_bytes: bytes) -> dict:
        """Full analysis: depth + nav planning.

        Merged from Buddy's v1.0:
          - lower_floor (bottom 1/4 of bottom-half) for column nav decisions
          - col_mins instead of col_means (ignores ceiling/wall bleed)
          - History-based distance smoothing (median of last 3 readings)
          - Fallback return when no valid floor pixels
        """
        self.load_model()
        self.warmup()

        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "decode_failed"}

        h, w = img.shape[:2]

        raw_depth = self._inference(img)

        # Calibration cached — only load once (plop optimization)
        if self._calib is None:
            self._calib = load_calibration()
        calib = self._calib
        depth_cal = apply_calibration(raw_depth, calib)

        # Nav: center strip, bottom half (actual floor area)
        # Ceiling/walls in top portion produce spurious <0.5m calibrated values.
        # Use bottom half so only real floor + lower-frame obstacles contribute.
        cx1, cx2 = w // 3, 2 * w // 3
        center = depth_cal[:, cx1:cx2]
        floor = center[h // 2:, :]

        # Filter valid floor pixels
        valid = floor[floor > 0.05]
        if len(valid) == 0:
            # No valid floor — still colorize the full depth map so the dashboard shows it
            self._colorize_depth_to_jpg(raw_depth, w, h)
            self._depth_dist_history.append(400)  # treat as unreliable
            if len(self._depth_dist_history) > 3:
                self._depth_dist_history.pop(0)
            sorted_hist = sorted(self._depth_dist_history)
            fallback_dist = min(
                sorted_hist[len(sorted_hist) // 2] if sorted_hist else 400, 400
            )
            return {
                "depth_m": 0.0, "clear_path": True, "obstacle_detected": False,
                "suggested_action": "forward", "obstacle_pct": 0.0,
                "distance_to_obstacle_cm": fallback_dist,
                "calibration": calib, "img_h": h, "img_w": w
            }

        median_depth = float(np.median(valid))
        mean_depth = float(np.mean(valid))

        # Obstacle: fraction of floor pixels where calibrated depth is < 0.5m.
        # Using bottom-half center strip avoids ceiling/wall reflections that
        # pollute the full frame. A real 15-30cm obstacle at close range should
        # register as 20-40%+ of this region.
        obstacle_mask = floor < 0.5
        obstacle_pct = float(np.sum(obstacle_mask) / floor.size)

        # Column-wise analysis for turn decision
        # Only look at the lower PORTION of the floor region (bottom half of
        # the bottom-half) to avoid ceiling/wall near-readings at y=h//2.
        # The ceiling is in every column but occupies a thin band at the top
        # of our region; the lower band is dominated by floor + real obstacles.
        # (Buddy v1.0: bottom 1/4 of bottom-half, col_mins instead of col_means)
        lower_floor = floor[floor.shape[0] // 2:, :]  # bottom half of bottom-half
        col_mins = lower_floor.min(axis=0)  # nearest object per column
        threshold_depth = 0.8  # meters
        near_cols = np.sum(col_mins < threshold_depth)
        total_cols = len(col_mins)
        near_pct = near_cols / total_cols if total_cols > 0 else 0

        # Decide action
        if obstacle_pct > 0.20 or median_depth < 0.5:
            action = "stop"
            clear = False
        elif obstacle_pct > 0.10 or near_pct > 0.15:
            # Narrow passage — turn toward clearer side
            left_half = col_mins[:len(col_mins)//2]
            right_half = col_mins[len(col_mins)//2:]
            left_clear = np.sum(left_half > threshold_depth)
            right_clear = np.sum(right_half > threshold_depth)
            action = "turn_left" if left_clear > right_clear else "turn_right"
            clear = False
        else:
            action = "forward"
            clear = True

        # Store colorized JPEG in cache for /depth/colorized.jpg endpoint (fast read, no inference)
        self._colorize_depth_to_jpg(raw_depth, w, h)

        # Smooth distance_to_obstacle_cm: median of last 3 readings.
        # (Buddy v1.0: history-based smoothing)
        current_dist = round(median_depth * 100, 1)
        self._depth_dist_history.append(current_dist)
        if len(self._depth_dist_history) > 3:
            self._depth_dist_history.pop(0)

        # If current reading is absurd (>400cm) or a fallback, use median of history
        if current_dist > 400 or len(valid) == 0:
            if self._depth_dist_history:
                sorted_hist = sorted(self._depth_dist_history)
                smooth_dist = sorted_hist[len(sorted_hist) // 2]
            else:
                smooth_dist = 400  # no history yet, cap at 400cm
        else:
            smooth_dist = current_dist

        # Hard cap: distance_to_obstacle_cm should never be shown as >= 400
        if smooth_dist >= 400:
            smooth_dist = min(smooth_dist, 400)

        return {
            "depth_m": round(median_depth, 3),
            "mean_depth_m": round(mean_depth, 3),
            "clear_path": clear,
            "obstacle_detected": not clear,
            "obstacle_pct": round(obstacle_pct, 3),
            "suggested_action": action,
            "distance_to_obstacle_cm": smooth_dist,
            "near_pct": round(near_pct, 3),
            "calibration": calib,
            "img_h": h,
            "img_w": w,
        }

    # ── Colorization ──────────────────────────────────────────────────────────

    def _build_colorize_lut(self):
        # Spectrum: red=far, black=near
        # Black → Purple → Blue → Green → Yellow → Orange → Red
        stops = [
            (0, 0, 0),         # 0%:   Black   (near)
            (139, 0, 255),     # 17%:  Purple
            (0, 0, 255),       # 29%:  Blue
            (0, 255, 0),       # 43%:  Green
            (255, 255, 0),     # 57%:  Yellow
            (255, 127, 0),     # 71%:  Orange
            (255, 0, 0),       # 100%: Red     (far)
        ]
        n = 256
        self._spectrum_lut = np.zeros((n, 3), dtype=np.uint8)
        segs = len(stops) - 1
        for i in range(n):
            t = i / (n - 1)
            seg = min(int(t * segs), segs - 1)
            local_t = (t - seg / segs) * segs
            c0, c1 = stops[seg], stops[seg + 1]
            self._spectrum_lut[i] = [
                int(c0[0] + (c1[0] - c0[0]) * local_t),
                int(c0[1] + (c1[1] - c0[1]) * local_t),
                int(c0[2] + (c1[2] - c0[2]) * local_t),
            ]

    def _colorize_depth_to_jpg(self, raw_depth, w, h) -> bytes:
        """Colorize raw depth array to JPEG, cache result. No inference."""
        valid = raw_depth[raw_depth > 0.05]
        if len(valid) > 0:
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            if vmax <= vmin:
                vmax = vmin + 5.0
            depth_norm = np.clip((raw_depth - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            depth_norm = np.zeros((h, w), dtype=np.uint8)

        # Apply spectrum LUT: red=far (high), black=near (low)
        # imencode expects BGR — convert from RGB
        colorized_bgr = cv2.cvtColor(self._spectrum_lut[depth_norm], cv2.COLOR_RGB2BGR)
        self._cached_colorized_jpg = cv2.imencode('.jpg', colorized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        return self._cached_colorized_jpg

    def get_colorized_jpg(self) -> bytes:
        return self._cached_colorized_jpg or b''


_depth_engine = None

def get_depth_engine():
    global _depth_engine
    if _depth_engine is None:
        _depth_engine = DepthEngine()
    return _depth_engine