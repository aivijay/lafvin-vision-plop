#!/usr/bin/env python3
"""
LAFVIN Vision HE — Laptop App (full web dashboard)
All depth analysis runs HERE on the laptop, not on the Pi.
"""
import os, sys, time, json, base64, io, asyncio, threading
from pathlib import Path

LAFVIN_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LAFVIN_ROOT / "shared"))

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

from depth import get_depth_engine, load_calibration, save_calibration
from robot_client import RobotClient
import torch

app = FastAPI(title="LAFVIN Vision HE", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

RPI_URL = os.environ.get("RPI_URL", "http://192.168.1.54:9000")
rpi = RobotClient(RPI_URL)

# Torch inter-op lock — prevents multiple threads doing torch inference simultaneously
_torch_lock = threading.Lock()

# ─── Camera (proxied from RPi) ────────────────────────────────────────────────

@app.get("/camera/live")
async def camera_live():
    """Proxy MJPEG stream from RPi."""
    import asyncio

    async def stream():
        while True:
            try:
                resp = rpi.get_camera_frame()
                if resp and resp.get("image"):
                    jpg_bytes = base64.b64decode(resp["image"])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
            except Exception as e:
                print(f"[camera_live] error: {e}")
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera/frame")
async def camera_frame():
    """Get latest frame from RPi."""
    return rpi.get_camera_frame() or {"error": "no_frame"}


@app.get("/camera/raw_feed")
async def camera_raw_feed():
    """Redirect to RPi MJPEG stream (browser can access directly)."""
    return Response(
        content=b"",
        status_code=302,
        headers={"Location": f"{RPI_URL}/camera/live"}
    )


# ─── Depth (laptop-only) ───────────────────────────────────────────────────────

@app.post("/depth/analyze")
async def depth_analyze(request: Request):
    """Analyze image on laptop with Depth Anything V2."""
    try:
        body = await request.json()
        image_b64 = body.get("image", "")
        if not image_b64:
            return JSONResponse({"error": "no_image"}, status_code=400)

        jpg_bytes = base64.b64decode(image_b64)
        engine = get_depth_engine()
        result = engine.analyze(jpg_bytes)
        return JSONResponse(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/depth/analyze/colorized")
async def depth_analyze_colorized(request: Request):
    """Return colorized depth JPEG."""
    try:
        body = await request.json()
        image_b64 = body.get("image", "")
        if not image_b64:
            return JSONResponse({"error": "no_image"}, status_code=400)

        jpg_bytes = base64.b64decode(image_b64)
        engine = get_depth_engine()
        color_jpg = engine.get_colorized_depth_jpg(jpg_bytes)
        return Response(
            content=color_jpg,
            media_type="image/jpeg"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/depth/colorized.jpg")
async def depth_colorized_jpg():
    """Return cached colorized depth JPEG (no inference — computed in background_depth_loop)."""
    try:
        from depth import get_depth_engine
        engine = get_depth_engine()
        jpg = engine.get_colorized_depth_jpg(b'')
        if not jpg:
            return Response(content=b"", status_code=204)
        return Response(content=jpg, media_type="image/jpeg")
    except Exception as e:
        print(f"[depth/colorized] error: {e}")
        return Response(content=b"", status_code=500)


@app.get("/depth/config")
async def depth_config():
    return JSONResponse(load_calibration())


@app.post("/depth/config")
async def depth_config_set(request: Request):
    try:
        calib = await request.json()
        save_calibration(calib)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Robot (proxied from RPi) ─────────────────────────────────────────────────

@app.get("/robot/status")
async def robot_status():
    return JSONResponse(rpi.get_status())


@app.post("/robot/command")
async def robot_command(request: Request):
    try:
        body = await request.json()
        action = body.get("action", "stop")
        speed = int(body.get("speed", 50))
        return JSONResponse(rpi.send_command(action, speed))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── SSE stream (laptop orchestrates everything) ─────────────────────────────────

@app.get("/ws/updates")
async def ws_updates(request: Request):
    """
    SSE stream:
    - Fetch camera frame from RPi
    - Get robot status from RPi
    - Serve depth result from last background inference
    - Combine into single event to frontend
    """
    import concurrent.futures

    depth_engine = get_depth_engine()
    latest_camera_b64 = ""
    latest_robot_status = {}
    latest_depth_result = None
    inference_state = {"busy": False, "result": None}
    _camera_event = threading.Event()
    _lock = threading.Lock()
    _camera_lock = threading.Lock()
    _first_frame_ready = threading.Event()

    def camera_poll_loop():
        """Fast loop: poll camera + robot status at ~1s interval. Never blocked by inference."""
        nonlocal latest_camera_b64, latest_robot_status
        while True:
            try:
                frame_data = rpi.get_camera_frame()
                if frame_data and frame_data.get("image"):
                    b64 = frame_data["image"]
                    try:
                        robot_status_data = rpi.get_status()
                    except Exception:
                        robot_status_data = {}
                    with _camera_lock:
                        latest_camera_b64 = b64
                        latest_robot_status = robot_status_data
                        _first_frame_ready.set()
                time.sleep(0.25)
            except Exception as e:
                print(f"[ws camera] error: {e}")
                time.sleep(2)

    def background_depth_loop():
        """Run depth inference continuously. Camera/robot status handled by camera_poll_loop."""
        nonlocal latest_depth_result
        depth_engine = get_depth_engine()
        while True:
            try:
                if inference_state["busy"]:
                    time.sleep(0.2)
                    continue

                with _camera_lock:
                    b64 = latest_camera_b64

                if not b64:
                    time.sleep(0.25)
                    continue

                jpg_bytes = base64.b64decode(b64)
                inference_state["busy"] = True
                try:
                    # Torch inter-op lock prevents multiple threads in torch C++ backend
                    with _torch_lock:
                        depth_result = depth_engine.analyze(jpg_bytes)
                finally:
                    inference_state["busy"] = False

                with _lock:
                    latest_depth_result = depth_result

                time.sleep(0.25)
            except Exception as e:
                print(f"[ws depth] error: {e}")
                time.sleep(1)

    # Start both threads
    camera_thread = threading.Thread(target=camera_poll_loop, daemon=True)
    camera_thread.start()
    depth_thread = threading.Thread(target=background_depth_loop, daemon=True)
    depth_thread.start()

    async def event_generator():
        # Wait for camera thread to get first frame before sending events
        _first_frame_ready.wait(timeout=10)
        while True:
            try:
                now = time.time()

                # Send whatever we have from the background threads
                with _camera_lock:
                    camera_b64 = latest_camera_b64
                    robot_status = latest_robot_status
                with _lock:
                    depth_result = latest_depth_result

                event = {
                    "camera": camera_b64,
                    "robot": robot_status,
                    "timestamp": now,
                }
                if depth_result:
                    event["depth"] = depth_result

                yield {
                    "event": "update",
                    "data": json.dumps(event),
                }

                # SSE at ~2fps — gives depth loop enough time to produce results
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"[ws] error: {e}")
                await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


# ─── Frontend static files ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print(f"[lafvin_vision_he] Starting...")
    print(f"[lafvin_vision_he] RPi URL: {RPI_URL}")
    print(f"[lafvin_vision_he] Listening on http://0.0.0.0:9000")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)