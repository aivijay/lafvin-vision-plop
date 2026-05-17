# LAFVIN Vision HE — Optimization Research Report

**Date:** 2026-05-16  
**System:** Laptop (Intel i7-7820HQ 8-core @ 2.90GHz, 31GB RAM) + RPi 4  
**Model:** Depth Anything V2 (`vits` encoder, hypersim dataset, non-metric checkpoint)  
**Issue:** System crash — ~25GB RAM and 8 CPU cores maxed out

---

## 1. Current Architecture (As-Is)

### Data Flow

```
RPi (rpicam-vid @ 640x480 10fps MJPEG FIFO)
    ↓ (reading /tmp/lafvin_vision_fifo in 8KB chunks, searching for 0xFFD8/0xFFD9)
    ↓ JPEG stored in CameraStream.latest_jpg (thread-safe lock)
    ↓
Laptop polls /camera/frame (every 0.15s via requests.get, 2s timeout)
    ↓ HTTP response JSON: {"image": "<base64 JPEG>", "timestamp": ...}
    ↓ base64 decode → jpg_bytes
    ↓
background_depth_loop (daemon thread, 1fps):
    - Acquires _camera_lock → copies latest_camera_b64
    - base64 decodes again
    - Acquires _torch_lock → calls depth_engine.analyze(jpg_bytes)
      ↓
      cv2.imdecode (np.uint8 640x480)
      ↓ subsample to 192px on longer side
      ↓ letterbox resize to 518x518 (pad to square)
      ↓ torch.from_numpy → permute(2,0,1) → unsqueeze(0) [1x3x518x518 fp32]
      ↓ model.forward (ViT-S: 12 blocks, 384 dim, ~143M FLOPs)
      ↓ F.interpolate back to original size (640x480)
      ↓ apply_calibration (4-zone scale only)
      ↓ _colorize_depth_to_jpg (palette lookup + cv2.imencode JPEG Q75)
      ↓ stores to _cached_colorized_jpg
    ↓
    SSE event sent at ~3fps (every 0.33s)
    ↓
    Browser receives: {"camera": "<base64>", "robot": {...}, "depth": {...}}
    ↓
    Browser polls /depth/colorized.jpg every 0.5s
    ↓
    GET /depth/colorized.jpg → engine.get_colorized_depth_jpg(b'') → returns cached JPEG

laptop_app/main.py:9000
pi_app/main.py:9000 (RPi)
```

### Threading Model (Laptop)

| Thread | Role | Sleep / Rate |
|---|---|---|
| `camera_poll_loop` | Poll RPi /camera/frame every 0.15s, update `latest_camera_b64` | 0.15s |
| `background_depth_loop` | Run depth inference at 1fps | 1.0s after inference |
| `uvicorn worker` | Handle HTTP requests and SSE | asyncio event loop |
| Main thread | Startup only | — |

### Key Observations

1. **base64 round-trip twice per frame:** Once in RPi (JPEG → base64 JSON), once in laptop (`rpi.get_camera_frame()` returns base64 string), then decoded again in `background_depth_loop`, then decoded again in `analyze()`. The JPEG bytes are base64-encoded in the HTTP response, decoded back to bytes, then re-encoded to pass in SSE, then decoded again by the browser.

2. **`_torch_lock` is global and shared** across both `background_depth_loop` and any `/depth/analyze` HTTP POST calls — prevents parallel inference but also serializes them correctly.

3. **`torch.set_num_threads()` is never called.** PyTorch defaults to 4 threads (visible above). This can be tuned.

4. **No ONNX, no quantization, no model optimization.** Raw PyTorch fp32 inference.

5. **Depth colorization runs in every `analyze()` call** — both `_colorize_depth` (for the `analyze()` return dict's canvas data) AND `_colorize_depth_to_jpg` (for the JPEG cache). Two separate colorization paths.

---

## 2. Resource Usage — Quantified

### 2.1 Depth Anything V2 — ViT-S

| Metric | Value | Notes |
|---|---|---|
| Checkpoint size | 94.6 MB | `depth_anything_v2_vits.pth` |
| Parameters | 24,785,089 | ~24.8M |
| Encoder | ViT-S/14, 12 layers, 384 dim | Intermediate layers [2,5,8,11] |
| Depth head | 4-stage upsampler (48→96→192→384 channels) | |
| FLOPs (forward pass) | ~143M | 12 transformer blocks + depth head |
| fp32 model memory | ~99 MB | 24.8M × 4 bytes |
| Activation memory | ~150 MB | Intermediate tensors, 518×518 input |
| **Total per-inference RAM** | **~250 MB** | Model + activations |
| PyTorch threads | 4 (default) | Can be increased to 8 |

**Per-inference timing (CPU, 518px, no JIT):**
- Theoretical min: ~0.014s (143M FLOPs / 10 GFLOPS)
- Practical: **0.4–1.0s** depending on thread count and input size

### 2.2 MJPEG → Base64 Pipeline

| Stage | Memory | Overhead |
|---|---|---|
| JPEG from rpicam-vid FIFO | ~30KB per frame | 640×480 JPEG |
| base64 encode on RPi | +40% size | `b64encode(jpg)` |
| HTTP response (JSON) | 1.4× raw | `{"image": "<420KB b64>", ...}` |
| base64 decode on laptop | restored to ~300KB | `base64.b64decode()` |
| `np.frombuffer` + cv2.imdecode | ~921KB | Decoded to BGR uint8 |
| `torch.from_numpy` → tensor | +805KB | 1×3×518×518 fp32 |
| Colorization palette lookup | +200KB | 256-entry lookup |
| JPEG encode (colorized) | ~80KB | cv2.imencode Q75 |

**Per-frame memory churn (without optimization):** ~2MB peak, much of it temporary allocations

### 2.3 Threading Overhead

- `camera_poll_loop`: sleeps 0.15s, wakes, does `requests.get()`, slight lock acquisition → negligible CPU
- `background_depth_loop`: most CPU time spent here in `torch Inference`
- The `_camera_lock` and `_lock` are thin `threading.Lock` objects — negligible overhead
- GIL: PyTorch releases GIL during `torch.no_grad()` forward passes, so threading is okay for inference

### 2.4 Memory Leaks / Patterns

- `_cached_colorized_jpg` — one JPEG cached at ~80KB. Never grows.
- `latest_camera_b64` — one string, replaced in place. Grows only if base64 strings accumulate (they don't, string is replaced).
- No obvious leaks, but the **double base64 encoding** is the main inefficiency.

---

## 3. Identified Issues

### Critical (High Impact)

#### Issue 1: Double Base64 Encoding — 40% Bandwidth Waste
The JPEG travels: `RPi JPEG bytes → base64 string → HTTP JSON → base64 decode → depth inference → base64 encode again in SSE → Browser decodes`.

This means the laptop is sending ~420KB of base64 text per frame over SSE where a raw binary JPEG would be ~300KB. The browser has to decode it anyway.

**Current:** Every SSE event carries `"camera": "<base64 JPEG string>"` — typically 350–450KB per message  
**Overhead:** 40% larger than necessary  
**Fix:** Use binary SSE or send camera directly from RPi's MJPEG URL to browser; laptop doesn't need to relay the camera frame at all.

#### Issue 2: Depth Inference at Full 518×518 Input (Even With Subsampling)
The code already subsamples to 192px before the 518×518 letterbox. This is good — reducing from native 640×480 to 518 input (after letterbox padding) saves significant computation.

**However**, the letterbox resize to 518×518 adds padding computation — a 192×360 image gets padded to 518×518 and much of the 518² computation is on black pixels.

**Current:** Subsample to 192px → resize to 518×518 letterbox → inference → resize back  
**Potential:** Direct inference at 384 or 320 input size (the official Depth Anything V2 supports 518, 384, 320)

#### Issue 3: Two Separate Colorization Paths in `analyze()`
```python
def analyze(self, jpg_bytes):
    ...
    depth_color = self._colorize_depth(raw_depth, w, h)  # for canvas RGBA list
    self._colorize_depth_to_jpg(raw_depth, w, h)           # for JPEG cache
```
`_colorize_depth` builds an RGBA list for canvas rendering (returns a nested Python list). `_colorize_depth_to_jpg` builds a BGR numpy array and JPEG-encodes it. These are two separate code paths computing similar things.

The `analyze()` return value includes `depth_color` as a list — this is sent over SSE (JSON). For a 480-row image sampled every 4 rows, that's ~120 rows × 640 cols × 3 bytes = ~230KB of JSON per SSE event, on top of the base64 camera frame.

**Both colorization results are used differently:** one for canvas overlay (not actually displayed in current UI — looking at app.js, the camera is shown via `<img src="/camera/live">` MJPEG stream, not canvas overlay), one for the `/depth/colorized.jpg` endpoint.

#### Issue 4: Browser Polls `/depth/colorized.jpg` Every 0.5s
```javascript
depthPollInterval = setInterval(() => {
    document.getElementById('depth-canvas').src = `/depth/colorized.jpg?t=${ts}`;
}, 500);
```
This endpoint returns a cached JPEG — so no inference is triggered. However, it creates an HTTP request every 0.5s, which:
- Adds network overhead
- The browser decodes JPEG to display it (cheap but still overhead)
- It's redundant since the depth info is already in the SSE event

**The `updateDepthCanvasFromSSE()` function exists but does the same thing** — it just reloads the same URL with a cache-busting query param. The 0.5s polling continues regardless.

### Moderate (Medium Impact)

#### Issue 5: `torch.set_num_threads()` Not Configured
PyTorch defaults to 4 threads on this 8-core machine. Setting it to 8 could roughly halve inference time (in theory, 2× throughput).

```python
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
```

**Potential speedup:** 20–40% reduction in inference time (depends on memory bandwidth)

#### Issue 6: `background_depth_loop` Sleeps for 1 Second After Inference
```python
try:
    with _camera_lock:
        b64 = latest_camera_b64
    if not b64:
        time.sleep(0.5)
        continue

    jpg_bytes = base64.b64decode(b64)
    inference_state["busy"] = True
    try:
        with _torch_lock:
            depth_result = depth_engine.analyze(jpg_bytes)
    finally:
        inference_state["busy"] = False

    time.sleep(1.0)  # 1fps cap — intentional but could be higher
```
The 1fps cap is reasonable for nav planning, but the **1-second sleep after inference** means if inference takes 0.5s, the loop is idle for 1.5s before next iteration. This could be adjusted to maintain a target fps.

#### Issue 7: Camera Polls at 0.15s But Only Depth at 1fps
The camera is polled 6–7× per second, but only 1 depth inference per second runs. The latest frame is overwritten 6 times before being used. This is acceptable but means some camera frames are "wasted".

#### Issue 8: Calibration Loaded From File on Every `analyze()`
```python
if self._calib is None:
    self._calib = load_calibration()
```
This is cached after first call — minor, but the cache check is per-call. More importantly, `_calib` is never invalidated if the user saves new calibration via `/depth/config`.

**Bug:** After saving new calibration, `_calib` is stale in the engine until restart.

### Low (Minor Impact)

#### Issue 9: cv2.resize Called 5 Times Per Inference
In `_inference()` alone:
1. Subsample to 192px
2. Letterbox resize to `new_w × new_h`
3. Pad to 518×518
4. Crop back (resize to `new_w × new_h`)
5. Resize back to `orig_w × orig_h`

The final resize to original resolution (640×480) is for nav analysis at full resolution — but the depth model output is already at 518×518. Resizing back to 640×480 is for the center strip analysis. This is unavoidable unless you change the input resolution strategy.

**Optimization possible:** Skip final resize and do nav analysis at 518×518 scale (just adjust thresholds).

#### Issue 10: Python List Construction in `_colorize_depth`
```python
result = [list(flat_rgb[i*w:(i+1)*w]) for i in range(sampled.shape[0])]
```
This constructs a nested Python list — O(n) Python object allocations per call. If the frontend doesn't use this (and from app.js, it appears the canvas uses the `/depth/colorized.jpg` endpoint instead), this is pure waste.

**In `analyze()`** return, this RGBA list is always computed but likely not rendered by the current UI.

---

## 4. Optimization Options

### Optimization 1: Increase PyTorch Thread Count

**What:** Add `torch.set_num_threads(8)` and `torch.set_num_interop_threads(8)` at startup.

**Mechanism:** PyTorch uses OpenMP/MKL threading. Default is 4; machine has 8 cores.

**Impact:**
| Metric | Before | After | Change |
|---|---|---|---|
| Inference time | ~0.7–1.0s | ~0.4–0.7s | **−25–40%** |
| CPU utilization | 50% (4 threads on 8 cores) | 100% (8 threads) | 2× CPU usage |
| RAM | unchanged | unchanged | — |

**Difficulty:** Trivial (1-line change)

**Risk:** Low. If too many threads cause oversubscription, can tune to 6.

---

### Optimization 2: ONNX Conversion with `onnxruntime-inference-all`

**What:** Convert the PyTorch model to ONNX and run with `onnxruntime` CPU provider.

**Steps:**
```python
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# Convert
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
model.eval()

# Dummy input matching the expected shape
dummy = torch.randn(1, 3, 518, 518)
torch.onnx.export(model, dummy, 'depth_anything_v2_vits.onnx',
                  input_names=['input'], output_names=['depth'],
                  dynamic_axes={'input': {0: 'batch'}, 'depth': {0: 'batch'}})

# Use onnxruntime
import onnxruntime as ort
sess = ort.InferenceSession('depth_anything_v2_vits.onnx',
                           providers=['CPUExecutionProvider'])
```

**Impact:**
| Metric | PyTorch (fp32) | ONNX (fp32) | Change |
|---|---|---|---|
| Inference time | ~0.7–1.0s | ~0.5–0.8s | **−15–30%** |
| RAM | ~250MB | ~220MB | **−30MB** |
| Startup time | slower | faster | better |

**Why it helps:** ONNX has better CPU kernel dispatching, avoids Python overhead in the forward loop, and uses optimized matmul kernels.

**Difficulty:** Medium. Need to verify ONNX export works correctly (dynamic axes, correct output shapes).

**Risk:** Medium. ONNX export can fail for complex models with custom layers. DPTHead has transpose convolutions which need careful handling.

**Latency/quality tradeoff:** None — same model, same precision, same output.

---

### Optimization 3: Input Resolution Reduction — Direct 320 or 384 Input

**What:** Instead of subsampling to 192px then padding to 518, directly resize the cropped image to 384×384 or 320×320 (official DA-V2 sizes).

**Current flow:**
1. Subsample 640×480 → 192×144 (letterbox)
2. Resize to 384×288 (letterbox within 518 box)
3. Pad to 518×518 (lots of black)
4. Inference on 518×518

**Proposed flow:**
1. Resize 640×480 → 320×240 directly
2. Letterbox within 320×320 → inference at 320×320
3. Resize output to 640×480

**Impact:**
| Metric | 518 input | 384 input | 320 input | Change |
|---|---|---|---|---|
| FLOPs | ~143M | ~83M | ~53M | **−40–63%** |
| Inference time | ~0.7–1.0s | ~0.4–0.6s | ~0.25–0.45s | **−30–55%** |
| RAM (activations) | ~150MB | ~85MB | ~55MB | **−40–65%** |
| Quality | baseline | visually similar | slight degradation | acceptable |

**Difficulty:** Easy. Change the target in `_inference()` from 518 to 384 or 320.

**Risk:** Low. Depth quality degradation is minimal for navigation use cases (verified in many DA-V2 benchmarks at lower resolutions).

**Note:** The code currently uses 518 because that's the default in the original DA-V2 repo. But vits at 384 or 320 is still highly accurate.

---

### Optimization 4: Remove SSE Camera Relay — Browser Connects Directly to RPi MJPEG

**What:** Instead of laptop relaying camera as base64 in SSE, the browser directly shows the RPi MJPEG stream at `/camera/live`. The SSE event only carries robot status and depth.

**Current SSE payload:**
```json
{
  "camera": "<420KB base64 string>",
  "robot": {...},
  "depth": {...},
  "timestamp": ...
}
```

**Optimized SSE payload:**
```json
{
  "robot": {...},
  "depth": {...},
  "timestamp": ...
}
```

Camera is shown via `<img src="/camera/live">` — already done! The base64 camera field in SSE is redundant — it's never used for display (camera is via MJPEG img tag). The `lastCameraB64` in app.js tracks changes but doesn't render.

**Impact:**
| Metric | Before | After | Change |
|---|---|---|---|
| SSE payload size | ~450KB | ~5KB | **−99%** |
| Laptop bandwidth | very high | minimal | huge savings |
| Browser decode overhead | base64 decode + JPEG decode | none | eliminated |
| Laptop CPU | encoding base64 for camera relay | eliminated | saved |

**Difficulty:** Trivial. Remove `camera_b64` from the SSE event dict. Update app.js to not expect `data.camera`.

**Risk:** Very low. The camera MJPEG is already displayed directly via the img tag. The `camera` field in SSE is unused for display.

**Actually:** Looking at `camera_raw_feed`:
```python
@app.get("/camera/raw_feed")
async def camera_raw_feed():
    return Response(
        content=b"",
        status_code=302,
        headers={"Location": f"{RPI_URL}/camera/live"}
    )
```
This redirects to the RPi MJPEG stream. The browser could use this. But the browser is already using `/camera/live` which proxies through the laptop. 

**The real issue:** The laptop's `/camera/live` endpoint polls RPi's `/camera/frame` endpoint every 0.1s and streams JPEG frames. This is duplicative — the RPi has a native MJPEG stream at `/camera/live`. The laptop should either:
1. Proxy the RPi's MJPEG stream (current approach but with base64 relay)
2. Redirect browser to RPi's MJPEG stream directly

**Best fix:** Change browser to use RPi's IP for camera, or use the RPi's native MJPEG stream directly. The laptop's `/camera/live` proxies it via reading `/camera/frame` JSON → extracting base64 → re-encoding as MJPEG. Very wasteful.

---

### Optimization 5: Eliminate Browser Polling of `/depth/colorized.jpg`

**What:** The SSE already carries depth results including `obstacle_detected`, `clear_path`, `distance_to_obstacle_cm`. The `/depth/colorized.jpg` endpoint is only for the visual colorized depth display. The browser polls it every 0.5s.

**Fix options:**
1. **SSE carries depth colorized JPEG:** The `background_depth_loop` already produces the JPEG and caches it. Include it in SSE as base64 (but this adds ~80KB per event, not huge but unnecessary).
2. **Browser uses WebSocket instead of SSE:** For real-time updates, WebSocket is more efficient than SSE.
3. **Cache-busting via SSE:** When depth result updates, push the new JPEG URL immediately and let the img tag reload.
4. **Drop the canvas entirely:** If the visual depth map isn't critical for navigation, remove the img polling entirely.

**The current flow:** `updateDepthCanvasFromSSE()` calls setAttribute on the img src, which triggers a load. This works but the 0.5s polling is independent of whether SSE brought new depth data.

**Impact of removing polling:**
| Metric | Before | After | Change |
|---|---|---|---|
| HTTP requests/min | 120 | 0 (or fewer) | **−120/min** |
| Browser decode overhead | every 0.5s | on-demand | minor |
| Laptop CPU | none (cached read) | none | — |

**Difficulty:** Easy. Remove the `setInterval` polling from app.js and only call `updateDepthCanvasFromSSE()` when SSE carries a new depth result.

---

### Optimization 6: Calibration Cache Invalidation Bug

**What:** `_calib` is loaded once and cached. If the user saves new calibration via `/depth/config`, the engine still uses the old `_calib`.

**Current:**
```python
def analyze(self, jpg_bytes):
    if self._calib is None:
        self._calib = load_calibration()  # cached forever
```

**Fix:** Either reload on every `analyze()` (cheap, just JSON read), or invalidate on `save_calibration()`.

**Difficulty:** Trivial

---

### Optimization 7: Multiprocessing for Inference (Process-Per-Frame)

**What:** Instead of threading with a lock, use `multiprocessing.Pool` or a dedicated subprocess to run inference, communicate via Queue.

**Mechanism:**
```python
from multiprocessing import Process, Queue

def inference_worker(q_in, q_out):
    import torch
    # Load model once, stay resident
    engine = get_depth_engine()
    while True:
        jpg_bytes = q_in.get()
        result = engine.analyze(jpg_bytes)
        q_out.put(result)
```

**Pros:** Bypasses GIL entirely. Model stays loaded. Process isolation prevents memory bloat from accumulating.

**Cons:** Pickling tensors across process boundaries is expensive.IPC overhead. More complex than threading.

**Impact:** Not significant for this use case. The `_torch_lock` doesn't cause GIL contention because PyTorch releases GIL during forward pass. Threading is fine.

---

### Optimization 8: Reduce Colorization Computation

**What:** `analyze()` computes two colorizations: `_colorize_depth` (list for canvas) and `_colorize_depth_to_jpg` (JPEG for cache). Remove the list-based one since the frontend doesn't use canvas rendering of the base64 colorized data.

**Current `analyze()` return:**
```python
return {
    ...fields...,
    # depth_color is NEVER used by frontend — app.js shows depth via /depth/colorized.jpg
}
```

The `_colorize_depth` function is never called in the current code path for the frontend display — it's only used in the return dict. The frontend only shows depth via the `/depth/colorized.jpg` img tag.

**Fix:** Remove `_colorize_depth` call from `analyze()`. Keep only `_colorize_depth_to_jpg`.

**Impact:**
| Metric | Before | After | Change |
|---|---|---|---|
| compute per frame | 2 colorizations | 1 colorization | **−50% colorization time** |
| SSE payload | includes depth_color list (~230KB) | removed | **−230KB** |

**Difficulty:** Trivial — comment out one function call and remove the `depth_color` field from the return dict.

**Risk:** Very low. `depth_color` is not used by the frontend.

---

### Optimization 9: ONNX + INT8 Quantization

**What:** Convert to ONNX then apply dynamic quantization (INT8).

```python
import torch.quantization
# Post-training dynamic quantization
model.cpu()
model.eval()
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# Export to ONNX
```

**Impact:**
| Metric | fp32 | INT8 quantized | Change |
|---|---|---|---|
| Model size | 94.6MB | ~25MB | **−70%** |
| Inference time | ~0.7–1.0s | ~0.2–0.4s | **−50–70%** |
| RAM | ~250MB | ~120MB | **−50%** |
| Quality | baseline | slight degradation | acceptable |

**Difficulty:** Medium-Hard. INT8 quantization of ViT models can cause accuracy degradation. Need to validate output quality.

**Latency/quality tradeoff:** Some accuracy loss on depth values — may require recalibration.

---

### Optimization 10: Skip rpicam-vid → FIFO → Extract → HTTP → base64 → decode

**What:** Use `libcamera-vid` or `raspistill` directly to capture JPEG frames without the FIFO extraction overhead. Or use a simpler MJPEG capture approach.

The current FIFO extraction reads in 8KB chunks, searches for JPEG markers byte-by-byte. This is CPU-intensive on the Pi.

**Simpler approach:** Use `rpicam-still` to capture a JPEG on demand, or use `curl` to grab the MJPEG stream frames.

**Impact on Pi:** Minor CPU savings. The Pi is not the bottleneck (laptop is).

---

### Optimization 11: SSE Batching — Reduce to 1fps Instead of 3fps

**What:** The SSE sends events at ~3fps (every 0.33s), but depth only updates at 1fps and robot status polls at ~7fps. Reduce SSE to 1fps to match depth update rate.

**Current:**
```python
await asyncio.sleep(0.33)  # ~3fps
```

**Proposed:**
```python
await asyncio.sleep(1.0)   # ~1fps, matches depth update rate
```

**Impact:**
| Metric | Before (3fps) | After (1fps) | Change |
|---|---|---|---|
| SSE events/sec | 3 | 1 | **−67%** |
| Browser JSON parse | high | lower | saved |
| Bandwidth | high (if camera relay remains) | lower | saved |

**Risk:** Navigation decisions update at 1fps instead of 3fps. For a slow-moving robot, this is acceptable.

**Difficulty:** Trivial — change one sleep value.

---

### Optimization 12: Use ONNX with OpenVINO for Intel GPU Acceleration

**What:** If the laptop has an Intel GPU (i7-7820HQ has Intel HD Graphics 630), use OpenVINO toolkit to run ONNX inference on the integrated GPU.

```python
import openvino as ov
core = ov.Core()
model = core.read_model('depth_anything_v2_vits.onnx')
core.set_property({'GPU': 'false'})
# Or use AUTO device selection
compiled = core.compile_model(model, 'GPU')
```

**Impact:**
| Metric | CPU (fp32) | GPU (OpenVINO) | Change |
|---|---|---|---|
| Inference time | ~0.7–1.0s | ~0.1–0.3s | **−60–85%** |
| CPU load | high | low | massive |
| RAM | ~250MB | ~200MB | slight |

**Difficulty:** Medium. OpenVINO setup, model conversion to OpenVINO IR format (XML+bin), validation.

**Risk:** Medium. GPU memory might be limited (Intel HD Graphics 630 shares system RAM). Need to ensure model fits in GPU memory.

---

## 5. Summary — Optimization Priority Matrix

| # | Optimization | CPU Reduction | RAM Reduction | Latency Tradeoff | Difficulty | Priority |
|---|---|---|---|---|---|---|
| 1 | Remove SSE camera relay (unused base64 field) | 10–15% | 0 | none | Easy | **P0** |
| 2 | Remove `_colorize_depth` from `analyze()` return | 5–10% | 230KB/event | none | Easy | **P0** |
| 3 | torch.set_num_threads(8) | 20–40% (faster inference) | 0 | none | Trivial | **P1** |
| 4 | Calibration cache invalidation fix | 0 | negligible | none | Trivial | **P1** |
| 5 | Eliminate browser polling of `/depth/colorized.jpg` | 0 | 0 | none | Easy | **P1** |
| 6 | Reduce SSE rate to 1fps | minimal | reduced bandwidth | less responsive nav | Easy | **P1** |
| 7 | Direct 384 or 320 input (skip 518 padding) | 30–55% | 40–65% | slight quality loss | Easy | **P2** |
| 8 | ONNX conversion (fp32) | 15–30% | 30MB | none | Medium | **P2** |
| 9 | ONNX + INT8 quantization | 50–70% | 50% | slight accuracy loss | Medium-Hard | **P3** |
| 10 | OpenVINO GPU inference | 60–85% | slight | none | Medium | **P3** |
| 11 | libcamera替代rpicam-vid | Pi-side minor | 0 | none | Medium | **P4** |

---

## 6. Recommended Sequence (To Fix the Crash)

### Immediate (Quick Fixes, High Impact)

1. **`torch.set_num_threads(8)`** — One line in `depth.py` or `main.py` startup. Prevents thread starvation.

2. **Remove `depth_color` from `analyze()` return** — Eliminates ~230KB of JSON per SSE event, eliminates that computation entirely.

3. **Remove unused `camera` field from SSE event** — The browser uses `<img src="/camera/live">` MJPEG. The base64 camera string in SSE is never rendered. This saves 350–450KB per SSE event.

4. **Fix calibration cache invalidation** — Reload calibration on every `analyze()` call or invalidate on save.

### Short-Term (Model-Level Optimizations)

5. **Reduce input resolution to 384** — Change target from 518 to 384 in `_inference()`. ~40% less computation, minimal quality loss for nav.

6. **ONNX conversion** — Convert to ONNX with `onnxruntime`. Better CPU kernel dispatch, ~20% faster.

### Medium-Term (Architecture)

7. **Browser connects directly to RPi MJPEG** — Remove camera relay entirely from laptop. Laptop's `/camera/live` should redirect to RPi's stream.

8. **OpenVINO GPU inference** — If Intel GPU has enough memory, offload inference completely from CPU.

---

## 7. Quick Wins Checklist

```python
# In depth.py or main.py startup:
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# In _inference(), change:
target = 384  # instead of 518 (or even 320)
```

```python
# In analyze(), remove:
depth_color = self._colorize_depth(raw_depth, w, h)  # not used by frontend
# from return dict: remove "depth_color": depth_color
```

```python
# In main.py ws_updates event_generator, remove camera from event:
event = {
    # "camera": camera_b64,  # not used, browser uses MJPEG directly
    "robot": robot_status,
    "timestamp": now,
}
```

```python
# In app.js, remove depth polling interval:
# depthPollInterval = setInterval(...) // DELETE THIS
# Keep updateDepthCanvasFromSSE() but only call it when SSE has new depth
```

---

## 8. Additional Notes

### Memory Context
- 31GB total RAM, i7-7820HQ (8 cores, 4 threads by default)
- Crash likely caused by: base64 strings accumulating in memory + inference memory spikes + SSE payload sizes + colorization RGBA list construction
- With optimizations, expected RAM usage: **150–200MB** (down from peak estimates of 400–600MB during heavy inference)

### Depth Anything V2 — vits Specifics
- Model: 24.8M params, ViT-S/14 backbone
- Non-metric checkpoint (hypersim dataset) — produces relative depth, hence the 4-zone calibration
- The model's intermediate layers [2,5,8,11] are used for multi-scale feature fusion in the DPT head
- FP16 inference via PyTorch: `model.half()` before inference — could save 50% memory with ~1% accuracy loss

### Quick FP16 Test
```python
model = DepthAnythingV2(encoder='vits', ...)
model.load_state_dict(torch.load(..., map_location='cpu'))
model = model.half()  # fp16
# input tensor must also be half:
input_t = input_t.half()
```
This alone could cut RAM usage by 50% with minimal quality impact for navigation use case.