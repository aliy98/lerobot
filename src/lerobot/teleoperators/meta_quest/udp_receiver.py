import json
import socket
from typing import Any, Dict, Optional, Tuple

import numpy as np


R2L_FLIP = np.diag([1.0, 1.0, -1.0, 1.0])


class UDPControllerReceiver:
    """
    Receive Quest controller data over UDP and always return the newest packet.

    The socket is drained on every read so the teleop loop does not fall behind
    and act on stale controller poses when packets arrive faster than the robot
    loop can consume them.
    """

    def __init__(
        self,
        ip: str = "0.0.0.0",
        port: int = 12345,
        buffer_size: int = 4096,
        timeout: Optional[float] = None,
    ) -> None:
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.ip, self.port))
        self._sock.setblocking(False)
        if timeout is not None:
            self._sock.settimeout(timeout)

        self._cached_pose_data: Optional[Dict[str, np.ndarray]] = None
        self._cached_button_data: Optional[Dict[str, Any]] = None

        print(
            f"UDP receiver listening on {self.ip}:{self.port} "
            f"(returning the newest queued packet)"
        )

    def receive_once(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]]]:
        latest_raw = None

        while True:
            try:
                raw, _ = self._sock.recvfrom(self.buffer_size)
                latest_raw = raw
            except BlockingIOError:
                break
            except socket.timeout:
                break
            except OSError:
                break

        if latest_raw is None:
            return self._cached_pose_data, self._cached_button_data

        try:
            text = latest_raw.decode("utf-8", errors="ignore").strip()
            payload = json.loads(text)

            pose_data: Dict[str, np.ndarray] = {}
            for hand in ["l", "r"]:
                matrix_list = payload.get(hand)
                if not isinstance(matrix_list, list) or len(matrix_list) != 4:
                    return self._cached_pose_data, self._cached_button_data

                matrix = np.array(matrix_list, dtype=float)
                if matrix.shape != (4, 4):
                    return self._cached_pose_data, self._cached_button_data

                pose_data[hand] = R2L_FLIP @ matrix @ R2L_FLIP

            button_data = payload.get("buttons", {})
            for trig in ["leftTrig", "rightTrig"]:
                if trig in button_data and isinstance(button_data[trig], (int, float)):
                    button_data[trig] = [float(button_data[trig])]

            self._cached_pose_data = pose_data
            self._cached_button_data = button_data
            return pose_data, button_data

        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            return self._cached_pose_data, self._cached_button_data

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()