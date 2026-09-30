# lafvin-vision-plop

Robot vision web dashboard for the LAFVIN 4WD robot. Streams camera feed from the Pi, runs Depth Anything V2 depth analysis on the laptop, and provides real-time navigation decisions via SSE.

See [SPEC.md](SPEC.md) for full architecture and API details.

## Hardware

- LAFVIN 4WD robot with Raspberry Pi (RPi URL: `http://192.168.1.170:9000`)
- Laptop runs depth inference (not the Pi)

## Quick Start

**Pi** (on the robot):

```bash
cd pi_app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9000
```

**Laptop**:

```bash
cd laptop_app
pip install -r requirements.txt
# Install Depth Anything V2 separately (see docs/OPTIMIZATION_RESEARCH.md)
uvicorn main:app --host 0.0.0.0 --port 9000
```

Then open http://localhost:9000

## Project Structure

```
lafvin-vision-plop/
├── SPEC.md                    # Full architecture and API docs
├── OPTIMIZATION_NOTES.md      # Performance notes
├── docs/
│   └── OPTIMIZATION_RESEARCH.md
├── pi_app/                    # Runs on RPi — camera + robot control
│   ├── main.py
│   ├── static/index.html
│   └── requirements.txt
├── laptop_app/                # Runs on laptop — depth + nav dashboard
│   ├── main.py
│   ├── depth.py               # Depth Anything V2 wrapper
│   ├── onnx_inference.py
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── requirements.txt
├── shared/
│   └── robot_client.py        # PiRobotClient — HTTP client for RPi
└── tests/
    ├── benchmark_depth.py
    └── test_sse_flow.py
```

## Tech Stack

- **Backend**: FastAPI + uvicorn
- **Frontend**: Vanilla JS (no framework)
- **Depth**: Depth Anything V2 (vits/hypersim) on laptop
- **Streaming**: Server-Sent Events (SSE)

## Navigation Logic

Depth analysis of center strip bottom half:

| Condition | Action |
|-----------|--------|
| Median floor depth < 0.5m | `stop` |
| Obstacle % > 15% | `stop` |
| Center clear | `forward` |
| Otherwise | Turn toward clearer side |
