#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property

import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive
import torch

from lerobot.robots.robot import Robot
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_ur5 import UR5Config
from .robotiq_gripper import RobotiqGripper

logger = logging.getLogger(__name__)


class UR5(Robot):
    """
    UR5 / UR5e robot for LeRobot.

    send_action auto-selects the control stream from action keys:

      Joint mode (policy / Gello):
        joint_1.pos … joint_6.pos, gripper.pos  →  servoJ

      Cartesian mode (Meta Quest absolute EE):
        eef_x … eef_rz, gripper.pos             →  servoL

    On every joint ↔ cartesian switch, servoStop() is called first.
    """

    config_class = UR5Config
    name = "ur5"

    joint_names = (
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "gripper.pos",
    )
    camera_names = ("front", "wrist")
    camera_shape = (480, 640, 3)

    def __init__(self, config: UR5Config):
        super().__init__(config)
        self.config = config
        self.hardware: HardwareSetup | None = None
        # "joint" | "cartesian" — tracks last stream for safe switching
        self._last_control_mode: str | None = None
        self._last_gripper_bin = None
        

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def _motor_features(self) -> dict[str, type]:
        return {name: float for name in self.joint_names}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self._motor_features,
            **{camera_name: self.camera_shape for camera_name in self.camera_names},
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Dataset / policy interface stays joint-space.
        return self._motor_features

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.hardware is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.hardware = HardwareSetup(
            robot_ip=self.config.robot_ip,
            gripper_port=self.config.gripper_port,
            control_hz=self.config.control_hz,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            camera_fps=self.config.camera_fps,
        )
        self._last_control_mode = None
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Observations / home
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        assert self.hardware is not None

        joint_state = self.hardware.get_robot_state()
        images = self.hardware.get_images()

        obs_dict: RobotObservation = {
            name: float(value) for name, value in zip(self.joint_names, joint_state, strict=True)
        }
        obs_dict["front"] = images[0]
        obs_dict["wrist"] = images[1]
        return obs_dict

    @check_if_not_connected
    def move_robot_home(self) -> None:
        assert self.hardware is not None
        self._ensure_mode(None)  # stop any active servo stream
        self.hardware.move_robot_home()

    @check_if_not_connected
    def set_new_home(self, home_q: list[float]) -> None:
        assert self.hardware is not None
        if len(home_q) != 6:
            raise ValueError(f"home_q must have 6 elements, got {len(home_q)}")
        self.hardware.home_q = home_q

    @check_if_not_connected
    def get_tcp_pose(self) -> list[float]:
        """Absolute TCP pose [x, y, z, rx, ry, rz] (metres + axis-angle)."""
        assert self.hardware is not None
        return self.hardware.get_robot_eef().tolist()

    @check_if_not_connected
    def recover(self) -> bool:
        assert self.hardware is not None
        ok = self.hardware.recover_rtde()
        if ok:
            self._last_control_mode = None  # force clean mode after recovery
        return ok

    # ------------------------------------------------------------------
    # Action: auto joint vs cartesian
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cartesian_action(action: dict) -> bool:
        if "eef_x" in action or "tcp" in action:
            return True
        if "delta_x" in action:
            raise KeyError(
                "Received delta_* keys. MetaQuest must output absolute "
                "eef_x..eef_rz (or 'tcp') before send_action."
            )
        return False

    def _ensure_mode(self, mode: str | None) -> None:
        """
        mode: "joint" | "cartesian" | None
        Stops the previous RTDE servo stream when the mode changes (or on None).
        """
        if self.hardware is None:
            return
        if mode is None or mode != self._last_control_mode:
            try:
                self.hardware.servo_stop()
            except Exception:
                pass
            self._last_control_mode = mode

    def _apply_gripper(self, gripper_goal: float) -> float:
        assert self.hardware is not None
        gripper_bin = 1.0 if gripper_goal >= 0.5 else 0.0
        if gripper_bin == self._last_gripper_bin:
            return gripper_bin
        if gripper_bin >= 0.5:
            self.hardware.gripper.move(self.hardware.grip_max_pos, 255, 10)
        else:
            self.hardware.gripper.move(self.hardware.grip_min_pos, 255, 10)
        self._last_gripper_bin = gripper_bin
        return gripper_bin

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self.hardware is not None

        if self._is_cartesian_action(action):
            return self._send_action_cartesian(action)
        return self._send_action_joint(action)

    def _send_action_joint(self, action: RobotAction) -> RobotAction:
        self._ensure_mode("joint")

        goal_q = [float(action[name]) for name in self.joint_names[:6]]
        gripper_goal = float(action[self.joint_names[6]])

        if self.config.max_relative_target is not None:
            current_q = self.hardware.get_robot_q()
            if isinstance(self.config.max_relative_target, dict):
                limits = np.array(
                    [
                        self.config.max_relative_target.get(name, np.inf)
                        for name in self.joint_names[:6]
                    ]
                )
            else:
                limits = np.full(6, float(self.config.max_relative_target))
            goal_q_array = np.asarray(goal_q, dtype=np.float64)
            goal_q_array = current_q + np.clip(goal_q_array - current_q, -limits, limits)
            goal_q = goal_q_array.tolist()

        self.hardware.move_robot_q(goal_q)  # servoJ
        gripper_bin = self._apply_gripper(gripper_goal)

        return {
            **{name: float(value) for name, value in zip(self.joint_names[:6], goal_q, strict=True)},
            self.joint_names[6]: gripper_bin,
        }

    def _send_action_cartesian(self, action: RobotAction) -> RobotAction:
        self._ensure_mode("cartesian")

        if "tcp" in action:
            pose = [float(x) for x in action["tcp"]]
        else:
            pose = [
                float(action["eef_x"]),
                float(action["eef_y"]),
                float(action["eef_z"]),
                float(action["eef_rx"]),
                float(action["eef_ry"]),
                float(action["eef_rz"]),
            ]
        if len(pose) != 6:
            raise ValueError(f"Cartesian pose must have 6 elements, got {len(pose)}")

        gripper_goal = float(
            action.get(self.joint_names[6], action.get("gripper", action.get("gripper.pos", 1.0)))
        )

        # Optional singularity-aware gains (same idea as URTeleop)
        velocity = 0.15
        acceleration = 0.25
        gain = 150
        lookahead_time = 0.1
        try:
            status = self.hardware.get_singularity_status()
        except Exception:
            status = 0
        if status == 2:
            velocity, acceleration, gain, lookahead_time = 0.03, 0.08, 80, 0.20
        elif status == 1:
            velocity = min(velocity, 0.06)
            acceleration = min(acceleration, 0.12)
            gain = 110
            lookahead_time = 0.15

        self.hardware.servo_L(
            pose,
            velocity=velocity,
            acceleration=acceleration,
            lookahead_time=lookahead_time,
            gain=gain,
        )
        gripper_bin = self._apply_gripper(gripper_goal)

        # Return actual joints for dataset / logging (policy space stays joints)
        q = self.hardware.get_robot_q().tolist()
        return {
            **{name: float(value) for name, value in zip(self.joint_names[:6], q, strict=True)},
            self.joint_names[6]: gripper_bin,
        }

    @check_if_not_connected
    def servo_stop(self) -> None:
        assert self.hardware is not None
        self.hardware.servo_stop()
        self._last_control_mode = None

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.hardware is None:
            return

        try:
            self.hardware.servo_stop()
        except Exception:
            pass

        for pipeline_name in ("pipeline1", "pipeline2"):
            pipeline = getattr(self.hardware, pipeline_name, None)
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

        self.hardware = None
        self._last_control_mode = None
        logger.info("%s disconnected.", self)


UR5eRobot = UR5


# ======================================================================
# Hardware
# ======================================================================


class HardwareSetup:
    def __init__(
        self,
        robot_ip,
        gripper_port=63352,
        control_hz=30,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
    ):
        self.robot_ip = robot_ip
        self.gripper_port = gripper_port
        self.control_hz = control_hz
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.home_q = np.deg2rad([0, -90, 90, -90, -90, -180])

        self.rtde_c, self.rtde_r, self.gripper = self.connect_to_robot()
        self.grip_min_pos = self.gripper.get_min_position()
        self.grip_max_pos = self.gripper.get_max_position()
        print("ROBOT CONNECTED")

        self.pipeline1 = None
        self.pipeline2 = None
        self.connect_to_cameras()
        time.sleep(1)
        print("CAMERAS CONNECTED")

    def connect_to_robot(self):
        last_err = None
        for attempt in range(1, 4):
            try:
                print(f"[connect] attempt {attempt}/3 ...")
                rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                gripper = RobotiqGripper()
                gripper.connect(self.robot_ip, self.gripper_port)
                gripper.activate()
                print("[connect] OK")
                return rtde_c, rtde_r, gripper
            except Exception as e:
                last_err = e
                print(f"[connect] failed: {e}")
                # Best-effort cleanup before retry
                try:
                    if "rtde_c" in dir() and rtde_c is not None:
                        rtde_c.disconnect()
                except Exception:
                    pass
                try:
                    if "rtde_r" in dir() and rtde_r is not None:
                        rtde_r.disconnect()
                except Exception:
                    pass
                time.sleep(2.0)

        raise RuntimeError(
            "Failed to start RTDE control script after 3 attempts. "
            "Unlock protective stop, set Remote Control, kill other RTDE clients, wait 10s. "
            f"Last error: {last_err}"
        )

    def connect_to_cameras(self):
        realsense_ctx = rs.context()
        connected_devices = []
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            connected_devices.append(detected_camera)

        if len(connected_devices) < 2:
            raise RuntimeError(
                f"Expected 2 RealSense cameras, found {len(connected_devices)}: {connected_devices}"
            )

        self.pipeline1 = rs.pipeline()
        config1 = rs.config()
        config1.enable_device(connected_devices[0])
        config1.enable_stream(
            rs.stream.color, self.camera_width, self.camera_height, rs.format.rgb8, self.camera_fps
        )
        self.pipeline1.start(config1)

        self.pipeline2 = rs.pipeline()
        config2 = rs.config()
        config2.enable_device(connected_devices[1])
        config2.enable_stream(
            rs.stream.color, self.camera_width, self.camera_height, rs.format.rgb8, self.camera_fps
        )
        self.pipeline2.start(config2)
    
    def get_robot_state(self):
        joint_state = list(self.rtde_r.getActualQ())
        try:
            gripper_state = self.binarize_gripper_state()
        except Exception:
            gripper_state = getattr(self, "_last_grip_bin", 0.0)
        self._last_grip_bin = gripper_state
        joint_state.append(gripper_state)
        return joint_state

    def get_images(self):
        images = []
        for pipeline, cache_name in (
            (self.pipeline1, "_last_img1"),
            (self.pipeline2, "_last_img2"),
        ):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=30)  # was infinite
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("no color")
                img = np.asanyarray(color_frame.get_data()).astype(np.uint8)
                setattr(self, cache_name, img)
            except Exception:
                img = getattr(self, cache_name, None)
                if img is None:
                    img = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            images.append(img)
        return images
    
    def binarize_gripper_state(self):
        try:
            gripper_pos = self.gripper.get_current_position()
            if gripper_pos > self.grip_min_pos + 5:
                return 1.0
            return 0.0
        except Exception as e:
            print(f"[gripper] read failed, using last state: {e}")
            # fallback: keep previous binary state if you store it, else open
            return getattr(self, "_last_grip_bin", 0.0)

    def get_observation(self, task):
        images = self.get_images()
        state = self.get_robot_state()

        front_image = torch.tensor(images[0], dtype=torch.float32).permute(2, 0, 1) / 255.0
        wrist_image = torch.tensor(images[1], dtype=torch.float32).permute(2, 0, 1) / 255.0

        return {
            "observation.images.front": front_image,
            "observation.images.wrist": wrist_image,
            "observation.state": torch.tensor(state, dtype=torch.float32),
            "task": task,
        }

    def move_robot_home(self):
        self._rtde_call(self.rtde_c.moveJ, self.home_q, 1.0, 1.4)
        self.gripper.move(self.grip_min_pos, 255, 0)

    def move_robot_q(self, q, speed=0, acceleration=0, dt=None, lookahead_time=0.2, gain=100):
        # velocity/acceleration unused by ur_rtde servoJ in current versions
        if dt is None:
            dt = 1.0 / self.control_hz
        try:
            ok = self.rtde_c.servoJ(q, speed, acceleration, dt, lookahead_time, gain)
            # ur_rtde may return False or None when script is dead
            if ok is False:
                raise RuntimeError("RTDE control script is not running!")
            # Optional extra check
            if hasattr(self.rtde_c, "isProgramRunning") and not self.rtde_c.isProgramRunning():
                raise RuntimeError("RTDE control script is not running!")
        except Exception as e:
            if self._is_rtde_dead_error(e) or "not running" in str(e).lower():
                print(f"[HardwareSetup] servoJ failed → recovering: {e}")
                if not self.recover_rtde():
                    raise
                self.rtde_c.servoJ(q, speed, acceleration, dt, lookahead_time, gain)
            else:
                raise

    def servo_L(
        self,
        pose,
        velocity: float = 0.15,
        acceleration: float = 0.25,
        dt: float | None = None,
        lookahead_time: float = 0.1,
        gain: int = 150,
    ):
        if dt is None:
            dt = 1.0 / self.control_hz
        t0 = self.rtde_c.initPeriod()
        self._rtde_call(self.rtde_c.servoL, list(pose), velocity, acceleration, dt, lookahead_time, gain)
        self._rtde_call(self.rtde_c.waitPeriod, t0)

    def servo_stop(self):
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass

    def get_singularity_status(self) -> int:
        try:
            return int(self.rtde_c.getFreedriveStatus())
        except Exception:
            return 0

    def move_robot_eef(self, tcp):
        self.rtde_c.moveL(tcp, 0.15, 0.6)

    def get_robot_q(self):
        return np.array(self._rtde_call(self.rtde_r.getActualQ))

    def get_robot_eef(self):
        return np.array(self._rtde_call(self.rtde_r.getActualTCPPose))

    def _rtde_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if any(
                s in msg
                for s in (
                    "rtde control script is not running",
                    "registers are already in use",
                    "protective",
                    "socket disconnected",
                    "robot is disconnected",
                    "could not receive data",
                )
            ):
                logger.warning("[HardwareSetup] RTDE error → recover: %s", e)
                if self.recover_rtde():
                    return fn(*args, **kwargs)
            raise

    def recover_rtde(self, home_q=None, max_retries: int = 5, retry_delay: float = 1.5) -> bool:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[HardwareSetup recover] attempt {attempt}/{max_retries}")

                # 1. Stop motion / script (ignore errors)
                try:
                    self.rtde_c.servoStop()
                except Exception:
                    pass
                try:
                    self.rtde_c.stopScript()
                except Exception:
                    pass

                # 2. Disconnect old interfaces (critical – avoids "registers already in use")
                try:
                    if hasattr(self.rtde_c, "disconnect"):
                        self.rtde_c.disconnect()
                    elif hasattr(self.rtde_c, "stop"):
                        self.rtde_c.stop()
                except Exception as e:
                    print(f"  (disconnect rtde_c ignored: {e})")

                try:
                    if hasattr(self.rtde_r, "disconnect"):
                        self.rtde_r.disconnect()
                    elif hasattr(self.rtde_r, "stop"):
                        self.rtde_r.stop()
                except Exception as e:
                    print(f"  (disconnect rtde_r ignored: {e})")

                time.sleep(1.0)  # let robot release RTDE registers

                # 3. Brand-new interfaces (same as working script)
                self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                time.sleep(0.5)

                # 4. Unlock protective stop if still active
                try:
                    if hasattr(self.rtde_r, "isProtectiveStopped") and self.rtde_r.isProtectiveStopped():
                        print("  Protective stop still active → unlocking...")
                        self.rtde_c.unlockProtectiveStop()
                        time.sleep(0.5)
                except Exception as e:
                    print(f"  (unlock ignored: {e})")

                # 5. Health check on RECEIVE (works even if control was dead)
                q = self.rtde_r.getActualQ()
                print(f"[HardwareSetup recover] SUCCESS – joints = {q}")

                return True

            except Exception as e:
                print(f"[HardwareSetup recover] attempt {attempt} failed: {e}")
                time.sleep(retry_delay)

        print("[HardwareSetup recover] FAILED – unlock Protective Stop on pendant, then Enter")
        input()
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            print("[HardwareSetup recover] SUCCESS after manual unlock")
            return True
        except Exception as e:
            print(f"[HardwareSetup recover] still failed: {e}")
            return False


    def _is_rtde_dead_error(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "rtde control script is not running" in msg
            or "registers are already in use" in msg
            or "protective" in msg
            or "socket disconnected" in msg
            or "robot is disconnected" in msg
            or "could not receive data" in msg
        )


    def _rtde_call(self, fn, *args, **kwargs):
        """
        Call RTDE fn. If it raises a control-script / protective error, recover and retry once.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if self._is_rtde_dead_error(e):
                print(f"[HardwareSetup] RTDE error → recovering: {e}")
                if self.recover_rtde():
                    return fn(*args, **kwargs)
            raise