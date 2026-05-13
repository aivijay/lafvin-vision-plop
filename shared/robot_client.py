#!/usr/bin/env python3
"""
Robot client — proxies requests to the RPi (192.168.1.54).
"""
import requests
import base64
from typing import Optional

RPI_URL = "http://192.168.1.54:9000"
TIMEOUT = 2


class RobotClient:
    def __init__(self, base_url: str = RPI_URL):
        self.base_url = base_url

    def get_camera_frame(self) -> Optional[dict]:
        """Get latest camera frame from RPi."""
        try:
            resp = requests.get(f"{self.base_url}/camera/frame", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            print(f"[robot_client] get_camera_frame error: {e}")
        return None

    def get_camera_jpg(self) -> Optional[bytes]:
        """Get latest JPEG bytes from RPi via /camera/frame endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/camera/frame", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if "image" in data:
                    return base64.b64decode(data["image"])
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            print(f"[robot_client] get_camera_jpg error: {e}")
        return None

    def get_status(self) -> dict:
        """Get robot status from RPi."""
        try:
            resp = requests.get(f"{self.base_url}/robot/status", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            print(f"[robot_client] get_status error: {e}")
        return {"ultrasonic_cm": 0.0, "gimbal_h": 90, "gimbal_v": 90, "motors_on": False}

    def send_command(self, action: str, speed: int = 50) -> dict:
        """Send command to RPi robot."""
        try:
            resp = requests.post(
                f"{self.base_url}/robot/command",
                params={"action": action, "speed": speed},
                timeout=3,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            print(f"[robot_client] send_command error: {e}")
        return {"ok": False, "error": str(e)}

    def get_sse_stream(self):
        """Get the SSE stream URL."""
        return f"{self.base_url}/ws/updates"
