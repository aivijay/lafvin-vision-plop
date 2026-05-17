#!/usr/bin/env python3
"""
ONNX export for Depth Anything V2 (vits).

Exports the full model (ViT encoder + DPT head) to ONNX.
The ONNX model accepts raw uint8 [0-255] images of any size.

ENCODER_TARGET must be divisible by 14 (ViT patch size).
We use 392 (28×14) — faster than 518 with the same quality for nav use.

Run once:
    python3 export_onnx.py

Produces: ~/projects/depth-anything-v2/checkpoints/depth_anything_v2_vits.onnx
"""
import os, sys
from pathlib import Path

DEPTH_PROJECT = os.path.expanduser("~/projects/depth-anything-v2")
CHECKPOINT_DIR = Path(DEPTH_PROJECT) / "checkpoints"

sys.path.insert(0, DEPTH_PROJECT)
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

# ENCODER_TARGET must be divisible by 14 (ViT patch size).
# 392 (28×14) is fast (~40% fewer FLOPs than 518) while being valid.
ENCODER_TARGET = 392


def load_model(encoder='vits'):
    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)
    ckpt_path = CHECKPOINT_DIR / f"depth_anything_v2_{encoder}.pth"
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class DepthONNXExportable(nn.Module):
    """
    ONNX-exportable model: resize→pad→normalize preprocessing + ViT + DPT head.
    Input:  [1, 3, H, W] uint8 [0-255]  (any H, W — dynamic axes)
    Output: [1, 1, H, W] float32         (depth map at same H/W as input after letterboxing)
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Letterbox resize + center-pad to ENCODER_TARGET square + normalize.
        All ops are ONNX-traceable.

        x: [B, 3, H, W] float [0, 255]
        returns: [B, 3, ENCODER_TARGET, ENCODER_TARGET] float32
        """
        B, C, H, W = x.shape

        # Letterbox: longer side → ENCODER_TARGET
        if H > W:
            new_h, new_w = ENCODER_TARGET, int(W * ENCODER_TARGET / H + 0.5)
        else:
            new_h, new_w = int(H * ENCODER_TARGET / W + 0.5), ENCODER_TARGET

        # Resize — traces cleanly (matches cv2.INTER_LINEAR)
        resized = F.interpolate(x, size=(new_h, new_w),
                               mode='bilinear', align_corners=False)

        # Center-pad to ENCODER_TARGET × ENCODER_TARGET
        pad_h = ENCODER_TARGET - new_h
        pad_w = ENCODER_TARGET - new_w
        pt = pad_h // 2
        pl = pad_w // 2
        padded = F.pad(resized, (pl, pad_w - pl, pt, pad_h - pt), value=0.0)

        # Normalize
        normalized = (padded / 255.0 - self.mean) / self.std
        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] uint8 [0, 255]"""
        x_norm = self.preprocess(x)
        depth = self.base_model(x_norm)
        return depth.unsqueeze(1)


def export(model, output_path: Path, encoder='vits'):
    print(f"[onnx_export] Building exportable wrapper for {encoder}...")
    wrapper = DepthONNXExportable(model)
    dummy = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)

    print(f"[onnx_export] ENCODER_TARGET = {ENCODER_TARGET} (divisible by 14: {ENCODER_TARGET % 14 == 0})")
    print(f"[onnx_export] dummy input: {tuple(dummy.shape)}")
    print(f"[onnx_export] Exporting... (5-15 min on CPU)")

    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=['input'],
            output_names=['depth'],
            dynamic_axes={
                'input': {0: 'batch', 2: 'height', 3: 'width'},
                'depth': {0: 'batch', 2: 'height', 3: 'width'},
            },
            opset_version=18,
            verbose=False,
        )
        size_mb = output_path.stat().st_size / 1e6
        print(f"[onnx_export] SUCCESS — {size_mb:.1f} MB → {output_path}")
        return True
    except Exception as e:
        print(f"[onnx_export] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    encoder = 'vits'
    ckpt_path = CHECKPOINT_DIR / f"depth_anything_v2_{encoder}.pth"

    if not ckpt_path.exists():
        print(f"[onnx_export] ERROR: {ckpt_path} not found.")
        sys.exit(1)

    print(f"[onnx_export] Loading {encoder} from {ckpt_path}...")
    model = load_model(encoder)

    output_path = CHECKPOINT_DIR / f"depth_anything_v2_{encoder}.onnx"
    if output_path.exists():
        sz = output_path.stat().st_size / 1e6
        print(f"[onnx_export] Already exists: {output_path} ({sz:.1f} MB)")
    else:
        success = export(model, output_path, encoder)
        if not success:
            sys.exit(1)

    print("[onnx_export] Done. Run tests/benchmark_depth.py to compare.")