# LAFVIN Vision PLOP — Optimization Notes
# Fork of lafvin_vision_he (Buddy/hermes development)
# Optimization pass: 2026-05-16

## What's Different (vs lafvin_vision_he)

### Quick Wins Applied (P0-P1):

1. **`torch.set_num_threads(os.cpu_count())`** in `depth.py`
   - PyTorch now uses all 8 cores instead of defaulting to 4
   - Expected: 25–40% faster inference

2. **Input resolution: 518 → 384** in `depth.py _inference()`
   - Direct letterbox to 384×384 (official DA-V2 size)
   - Expected: ~40% fewer FLOPs, similar depth quality for nav

3. **Calibration reloads every call** in `depth.py analyze()`
   - Was cached forever; now reloads on every analyze()
   - Fixes: saved calibration takes effect without restart

4. **`camera` field removed from SSE** in `laptop_app/main.py ws_updates`
   - Browser uses `<img src="/camera/live">` direct MJPEG, never needed the base64 relay
   - SSE payload: ~5KB instead of ~450KB per event
   - Significant bandwidth + browser decode savings

5. **`depth_color` RGBA list removed from `analyze()` return** in `depth.py`
   - Was ~230KB of Python list per event, never used by frontend
   - Saves colorization compute + JSON serialization

6. **Browser depth canvas polling eliminated** in `app.js`
   - No more 0.5s `setInterval` (was 120 HTTP requests/min)
   - `updateDepthCanvasFromSSE()` now triggers on-demand when SSE delivers depth

7. **SSE rate: 3fps → 1fps** in `laptop_app/main.py`
   - Matches depth inference rate (1 per second)
   - Robot status updates are stable at 1fps
   - Saves SSE events + browser JSON parsing

---

## P2 Changes (2026-05-16)

### 1. Bugfix: `torch.set_interop_threads` → `torch.set_num_interop_threads`

**File:** `laptop_app/depth.py`

PyTorch 2.6+ renamed `torch.set_interop_threads()` to `torch.set_num_interop_threads()`.
The old name raises `AttributeError` and crashes the app on startup.

```python
# Before (crashes on PyTorch 2.6+):
torch.set_interop_threads(_torch_threads)

# After (compatible):
try:
    torch.set_num_interop_threads(_torch_threads)
except AttributeError:
    pass  # torch < 2.0 fallback
```

---

### 2. Bugfix: ENCODER_TARGET must be divisible by 14

**File:** `laptop_app/depth.py` (and `export_onnx.py`)

The ViT patch size is 14. The Depth Anything V2 model requires input dimensions
that are multiples of 14. The old code used `target = 384` which is NOT divisible
by 14 (384 % 14 = 4), causing an `AssertionError` crash in the ViT encoder's
`patch_embed` layer when processing images whose letterbox-resized dimensions
result in a non-multiple-of-14 input.

**Fix:** Changed `ENCODER_TARGET = 392` (28×14) throughout.

The letterbox resize from 640×480 now produces:
- Subsampled (144×192) → resized to 392 (divisible by 14 ✓)
- Direct (480×640) → letterboxed to 392 (divisible by 14 ✓)

---

### 3. ONNX Conversion — `depth_anything_v2_vits.onnx`

**Files:** `laptop_app/export_onnx.py` (new), `laptop_app/onnx_inference.py` (new)

Exported Depth Anything V2 (vits) to ONNX format with:
- Preprocessing bundled in (resize + pad + normalize)
- Dynamic axes for any input resolution
- `opset_version=18`, graph optimization enabled

**Checkpoint:** `~/projects/depth-anything-v2/checkpoints/depth_anything_v2_vits.onnx` (1.1 MB)

**Why ONNX?**
- ONNX Runtime uses a single compiled inference graph (no PyTorch overhead)
- On this laptop (Intel/Quadro M1200): **1.63x faster** than PyTorch fp32
- **135 MB less peak RAM** vs PyTorch fp32
- ONNX model loads in 0.9s vs 3.3s for PyTorch fp32

**Benchmark results (50 runs, 640×480, ENCODER_TARGET=392):**

| Backend           | Avg (ms) | Min (ms) | RAM Δ     |
|-------------------|----------|----------|-----------|
| PyTorch fp32      | 2209     | 541      | +249 MB   |
| ONNX Runtime CPU  | 1356     | 462      | +114 MB   |
| PyTorch fp16      | 37575    | 25100    | +47 MB    |

**fp16 note:** PyTorch fp16 is ~30x SLOWER on this machine because CPU lacks
hardware FP16 support (no AVX512-FP16, no NEON). fp16 is only useful with GPU
or ARM with NEON. NOT recommended for CPU-only deployment.

**How it works:**
- `depth.py` automatically tries ONNX first on startup
- Falls back to PyTorch if ONNX file not found
- The `analyze()` API is identical — no changes needed to main.py

---

### 4. `depth.py` refactored: PyTorch + ONNX in one class

**File:** `laptop_app/depth.py`

DepthEngine now supports three backends (auto-selected, ONNX preferred):

```
DepthEngine()
  ├── ONNX Runtime (CPU) ← preferred (fastest, lowest RAM)
  ├── PyTorch fp32     ← fallback
  └── PyTorch fp16     ← optional (use_fp16=True), slow on CPU
```

Key changes:
- `load_model()` tries ONNX first, falls back to PyTorch
- `_inference()` dispatches to ONNX or PyTorch path
- `infer_fp16()` available for explicit fp16 testing
- `ENCODER_TARGET = 392` (fixed: divisible by 14, matches ONNX export)

---

## Benchmark Script

**File:** `tests/benchmark_depth.py`

Runs 50 inference iterations per backend, reports:
- Average / min / max inference time
- RAM peak (RSS, delta from baseline)
- Depth output range

Run with:
```bash
cd ~/lafvin_vision_plop/laptop_app
python3 ../tests/benchmark_depth.py
```

---

## v1.0 Merge — Buddy's Navigation Fixes (2026-05-16)

**What was merged:** Buddy's last 4 commits from `lafvin_vision_he` v1.0 tag into `lafvin_vision_plop`.

### What changed and why:

#### 1. `lower_floor` analysis — bottom 1/4 of bottom-half (d2d9839, 04e1970)
**File:** `laptop_app/depth.py`

The old `col_means = floor.mean(axis=0)` counted every column where ANY pixel was near,
including columns where the ceiling at y=h//2 is close. This caused spurious turn
decisions from ceiling/wall reflections.

Buddy's fix: `lower_floor = floor[floor.shape[0] // 2:, :]` (bottom half of the
bottom-half, i.e. bottom 1/4 of the image) + `col_mins = lower_floor.min(axis=0)`.
Now only columns where the NEAREST object is near register as blocked. A 15cm hand
registers as ~25-35% obstacle_pct in this band.

Thresholds: stop 20%→15%, turn 10%→8%.

#### 2. History-based distance smoothing (3e7a9a7, ba9dadd, 4957e9b)
**File:** `laptop_app/depth.py`

Old: returned raw `median_depth * 100` each call.
New: appends to `_depth_dist_history` (max 3 readings), returns median.
When no valid floor pixels: falls back to last known good distance.
Sentinel value 500cm → replaced with 999 ("no reading"), capped at 400cm.

#### 3. SSE `event: "update"` fix (a6b384d)
**File:** `laptop_app/main.py`

Browser's `EventSource.onmessage` only fires on unnamed events. Adding
`"event": "update"` to each SSE event ensures the browser actually receives it.

#### 4. Colorize even when no floor pixels (5323244)
**File:** `laptop_app/depth.py`

Old: returned early (no depth visualization) when no valid floor pixels.
New: always runs `_colorize_depth_to_jpg()` so the dashboard shows a depth map
even in empty/unclear scenes.

---

### What was NOT merged:
- `torch.compile()` (cea095c) — PyTorch compile optimization. Skipped because our
  ONNX path already gives us the performance benefit without the compile overhead
  and compatibility risk.
- `target_long=192` resolution change — plop uses ENCODER_TARGET=392 for ViT
  compatibility with the ONNX model. Keep as-is.
- Ultrasonic filtering/hysteresis logic — already handled differently in plop's
  main.py (plop's version has `_prev_ultra` tracking but with simpler logic).

---

## Remaining (P3 — For Later):

- OpenVINO GPU inference → 60–85% faster inference if Intel GPU has enough memory
- Direct RPi MJPEG → browser bypasses laptop camera proxy entirely
- FP16: only useful on GPU or ARM/NEON hardware (not this laptop's CPU)

---

## To Test:

```bash
# Start Pi:
cd ~/lafvin_vision_plop/pi_app && python3 main.py

# Start Laptop:
cd ~/lafvin_vision_plop/laptop_app && python3 main.py

# Dashboard:
# http://localhost:9000

# Run benchmark (from laptop_app dir):
python3 ../tests/benchmark_depth.py

# Re-export ONNX (if model changes):
python3 export_onnx.py
```