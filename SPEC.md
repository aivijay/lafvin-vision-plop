# LAFVIN Vision HE — Robot Vision Web Dashboard

## Architecture

```
Browser ←→ Laptop (FastAPI :9000)
            ├─ GET /camera/live  ← RPi MJPEG proxy (forwaded)
            ├─ GET /camera/frame ← RPi JPEG captured on request
            ├─ POST /depth/analyze ← Laptop runs Depth Anything V2
            ├─ GET /robot/status  ← RPi robot state
            ├─ POST /robot/command ← RPi motor/gimbal commands
            └─ /ws/updates ← Laptop SSE: camera frame + depth + nav + robot

          Pi (192.168.1.54)
            ├─ rpicam-vid MJPEG stream → /camera/live
            ├─ Servo/motor control
            ├─ Ultrasonic reading
            └─ Port 9000 proxy for camera/robot endpoints

          Laptop (Depth + Nav)
            ├─ Depth Anything V2 (vits/hypersim)
            ├─ 4-zone smooth calibration
            └─ Nav planning from depth
```

## Pi services
- `pi_app/` — minimal FastAPI, camera MJPEG + robot commands only
- Runs on port 9000, forwards requests to laptop except camera/robot

## Laptop services
- `laptop_app/` — full web app with depth + nav
- Runs on port 9000
- `/camera/live` → proxies to RPi
- `/ws/updates` → laptop does depth analysis every 2s

## Tech stack
- Backend: FastAPI + uvicorn
- Frontend: Vanilla JS (no framework)
- Depth: Depth Anything V2 (laptop, not Pi)

## Project structure
```
~/lafvin_vision_he/        # root
  pi_app/                  # runs on RPi 192.168.1.54
    main.py                 # camera MJPEG + robot commands
    static/
      index.html            # minimal Pi camera view
    requirements.txt
    start.sh
  laptop_app/              # runs on laptop (or any machine)
    main.py                 # full web app
    depth.py                # Depth Anything V2 wrapper
    static/
      index.html            # full 4-panel SPA
      style.css
      app.js
    requirements.txt
    start.sh
  shared/
    robot_client.py         # PiRobotClient — talks to RPi
```

## Endpoints (laptop app)

### Camera
- `GET /camera/live` — RPi MJPEG stream (proxied via HTTP)
- `GET /camera/frame` — latest JPEG base64 (fetched from RPi)

### Depth (laptop)
- `POST /depth/analyze` — `{image: b64}` → `{depth_m, clear_path, obstacle_detected, suggested_action, ...}`
- `GET /depth/config`
- `POST /depth/config`

### Robot
- `GET /robot/status` → `{ultrasonic_cm, gimbal_h, gimbal_v, motors_on}` (proxied from RPi)
- `POST /robot/command` → `{action, speed}` (proxied to RPi)

### SSE
- `GET /ws/updates` — streams: `{camera_b64, robot_status, depth_result}`

## Nav planning logic
From depth analysis of center strip bottom half:
- If median floor depth < 0.5m → "stop" (very close)
- If obstacle_pct > 15% → "stop"
- If center clear → "forward"
- Otherwise turn toward clearer side (left/right)

## Startup
```bash
# Pi
cd ~/lafvin_vision_he/pi_app && uvicorn main:app --host 0.0.0.0 --port 9000

# Laptop
cd ~/lafvin_vision_he/laptop_app && uvicorn main:app --host 0.0.0.0 --port 9000
```