#!/usr/bin/env python3
"""
LAFVIN Vision HE — RPi App (minimal)
Camera MJPEG + robot commands. All vision processing on laptop.
"""
import os, sys, time, json, base64, subprocess, threading
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import uvicorn

# ─── Camera (rpicam-vid) ────────────────────────────────────────────────────────

class CameraStream:
    def __init__(self, resolution=(640, 480), fps=10):
        self.resolution = resolution
        self.fps = fps
        self.fifo = Path("/tmp/lafvin_vision_fifo")
        self.latest_jpg = None
        self.lock = threading.Lock()
        self._running = False
        self._proc = None
        self._thread = None

    def start(self):
        if self._running:
            return
        self.fifo.unlink(missing_ok=True)
        os.mkfifo(self.fifo)
        cmd = [
            "rpicam-vid",
            "--width", str(self.resolution[0]),
            "--height", str(self.resolution[1]),
            "--framerate", str(self.fps),
            "--codec", "mjpeg",
            "--output", str(self.fifo),
            "-t", "0",
            "--nopreview",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._running = True
        self._thread = threading.Thread(target=self._extract, daemon=True)
        self._thread.start()
        print(f"[camera] Started {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")

    def _extract(self):
        with open(self.fifo, "rb") as f:
            data = b""
            while self._running:
                try:
                    chunk = f.read(8192)
                    if not chunk:
                        time.sleep(0.01)
                        continue
                    data += chunk
                    while True:
                        si = data.find(b'\xFF\xD8')
                        ei = data.find(b'\xFF\xD9', si + 2)
                        if si == -1 or ei == -1:
                            break
                        jpg = data[si:ei+2]
                        data = data[ei+2:]
                        with self.lock:
                            self.latest_jpg = jpg
                except Exception as e:
                    print(f"[camera] error: {e}")
                    break

    def get_jpg(self):
        with self.lock:
            return self.latest_jpg

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)


# ─── Robot (from lafvin_robot_vision) ─────────────────────────────────────────

ROBOT_BASE = "/home/vijay/lafvin_robot_vision"
sys.path.insert(0, ROBOT_BASE + "/src")

HARDWARE_OK = False
try:
    from common.hardware import (
        ULTRASONIC_TRIG, ULTRASONIC_ECHO, SPEED_SLOW, SPEED_MEDIUM, SPEED_FAST
    )
    from robot.motors import Motors
    from robot.ultrasonic import Ultrasonic, get_ultrasonic
    from robot.servo_gimbal import ServoGimbal
    HARDWARE_OK = True
except Exception as e:
    print(f"[robot] Hardware not available: {e}")

_motors, _ultrasonic, _gimbal = None, None, None


def get_motors():
    global _motors
    if _motors is None and HARDWARE_OK:
        _motors = Motors()
    return _motors


def get_ultrasonic_sensor():
    global _ultrasonic
    if _ultrasonic is None and HARDWARE_OK:
        _ultrasonic = get_ultrasonic()
    return _ultrasonic


def get_gimbal():
    global _gimbal
    if _gimbal is None and HARDWARE_OK:
        _gimbal = ServoGimbal()
    return _gimbal


def get_robot_status():
    ultrasonic = get_ultrasonic_sensor()
    gimbal = get_gimbal()

    dist_cm = 0.0
    if ultrasonic:
        try:
            d = ultrasonic.read()
            if d > 0:
                dist_cm = d / 10.0  # mm -> cm
        except Exception as e:
            print(f"[ultrasonic] read error: {e}")

    h, v = 90, 90
    if gimbal:
        try:
            h, v = gimbal.h, gimbal.v
        except Exception as e:
            print(f"[gimbal] read error: {e}")

    return {
        "ultrasonic_cm": round(dist_cm, 1),
        "gimbal_h": h,
        "gimbal_v": v,
        "motors_on": HARDWARE_OK,
    }


def send_robot_command(action: str, speed: int = 50):
    motors = get_motors()
    if not motors:
        return {"ok": False, "error": "motors not available"}
    try:
        if action == "forward":   motors.forward(speed)
        elif action == "backward": motors.backward(speed)
        elif action == "left":    motors.spin_left(speed)
        elif action == "right":   motors.spin_right(speed)
        elif action in ("stop", "calm"): motors.stop()
        else: return {"ok": False, "error": f"unknown: {action}"}
        return {"ok": True, "action": action, "speed": speed}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="LAFVIN Vision HE — RPi")
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_cam = None


def get_cam():
    global _cam
    if _cam is None:
        _cam = CameraStream()
        _cam.start()
        time.sleep(1)
    return _cam


@app.get("/camera/live")
async def camera_live():
    """MJPEG stream from rpicam-vid."""
    cam = get_cam()
    def stream():
        while True:
            jpg = cam.get_jpg()
            if jpg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(0.05)
    return Response(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera/frame")
async def camera_frame():
    """Latest frame as base64 JSON."""
    cam = get_cam()
    jpg = cam.get_jpg()
    if jpg is None:
        return {"error": "no_frame"}
    return {"image": base64.b64encode(jpg).decode(), "timestamp": time.time()}


@app.get("/robot/status")
async def robot_status():
    return get_robot_status()


@app.post("/robot/command")
async def robot_command(action: str = "stop", speed: int = 50):
    return send_robot_command(action, speed)


@app.get("/ws/updates")
async def ws_updates():
    """SSE stream: camera frame + robot status."""
    async def gen():
        import asyncio
        cam = get_cam()
        while True:
            jpg = cam.get_jpg()
            status = get_robot_status()
            frame_b64 = base64.b64encode(jpg).decode() if jpg else ""
            data = json.dumps({
                "camera": frame_b64,
                "robot": status,
                "timestamp": time.time(),
            })
            yield {"event": "update", "data": data}
            await asyncio.sleep(0.2)

    return EventSourceResponse(gen())


@app.get("/")
async def root():
    return {"msg": "LAFVIN Vision HE — RPi. Connect laptop app on port 9000."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)