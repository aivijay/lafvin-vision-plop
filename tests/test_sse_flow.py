"""
Test SSE stream delivers camera, robot, and depth data.
Run with: python test_sse_flow.py
"""
import requests
import json
import time

SERVER = "http://localhost:9000"
TIMEOUT = 20

def test_sse_has_camera():
    resp = requests.get(f"{SERVER}/ws/updates", stream=True, timeout=TIMEOUT)
    for line in resp.iter_lines():
        if line.startswith(b"data:"):
            d = json.loads(line[5:])
            camera = d.get("camera", "")
            assert camera, f"Camera empty: {d}"
            assert len(camera) > 1000, f"Camera too short ({len(camera)}): {d}"
            print(f"  camera: {len(camera)} bytes - OK")
            return True
    return False

def test_sse_has_robot():
    resp = requests.get(f"{SERVER}/ws/updates", stream=True, timeout=TIMEOUT)
    for line in resp.iter_lines():
        if line.startswith(b"data:"):
            d = json.loads(line[5:])
            robot = d.get("robot", {})
            assert robot, f"Robot empty: {d}"
            assert "ultrasonic_cm" in robot, f"Missing ultrasonic_cm: {robot}"
            assert "gimbal_h" in robot, f"Missing gimbal_h: {robot}"
            print(f"  robot: {robot} - OK")
            return True
    return False

def test_sse_has_depth():
    start = time.time()
    resp = requests.get(f"{SERVER}/ws/updates", stream=True, timeout=TIMEOUT)
    for line in resp.iter_lines():
        if line.startswith(b"data:"):
            d = json.loads(line[5:])
            depth = d.get("depth")
            if depth:
                elapsed = time.time() - start
                assert "depth_m" in depth, f"Missing depth_m: {depth}"
                assert "clear_path" in depth, f"Missing clear_path: {depth}"
                assert "suggested_action" in depth, f"Missing suggested_action: {depth}"
                print(f"  depth: depth_m={depth['depth_m']} clear={depth['clear_path']} "
                      f"action={depth['suggested_action']} ({elapsed:.1f}s) - OK")
                return True
    return False

def test_depth_colorized_jpg():
    resp = requests.get(f"{SERVER}/depth/colorized.jpg", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(resp.content) > 1000, f"Image too small: {len(resp.content)} bytes"
    print(f"  colorized.jpg: {len(resp.content)} bytes - OK")

def test_robot_status():
    resp = requests.get(f"{SERVER}/robot/status", timeout=5)
    assert resp.status_code == 200
    d = resp.json()
    assert "ultrasonic_cm" in d, f"Missing ultrasonic_cm: {d}"
    print(f"  /robot/status: {d} - OK")

def test_camera_frame():
    resp = requests.get(f"{SERVER}/camera/frame", timeout=5)
    assert resp.status_code == 200
    d = resp.json()
    assert "image" in d, f"Missing image: {d}"
    print(f"  /camera/frame: {len(d['image'])} bytes - OK")

if __name__ == "__main__":
    import sys
    tests = [
        ("camera frame", test_camera_frame),
        ("robot status", test_robot_status),
        ("colorized.jpg", test_depth_colorized_jpg),
        ("SSE camera", test_sse_has_camera),
        ("SSE robot", test_sse_has_robot),
        ("SSE depth", test_sse_has_depth),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            print(f"TEST: {name}...")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
