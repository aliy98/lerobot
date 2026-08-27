#!/usr/bin/env python

"""
Meta Quest teleoperator for LeRobot — absolute EE targets (URTeleop-compatible).

Output action (when enabled):
  eef_x, eef_y, eef_z, eef_rx, eef_ry, eef_rz  – absolute TCP (metres + rotvec)
  gripper.pos                                 – 0 close / 1 open
  enabled                                     – bool

Requires a robot handle with get_tcp_pose() so enable (left X) can re-anchor
to the current TCP, matching URTeleop.reset_teleop_state().

Usage:
  teleop = MetaQuestTeleop(cfg)
  teleop.connect()
  teleop.set_robot(robot)   # must expose get_tcp_pose() -> [x,y,z,rx,ry,rz]
  action = teleop.get_action()
  robot.send_action(action)  # cartesian branch -> servoL
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_meta_quest import MetaQuestTeleopConfig

logger = logging.getLogger(__name__)


def _button_flag(button_data: dict, *names: str) -> bool:
    for name in names:
        value = button_data.get(name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else False
        return bool(value)
    return False


class MetaQuestTeleop(Teleoperator):
    config_class = MetaQuestTeleopConfig
    name = "meta_quest"

    def __init__(self, config: MetaQuestTeleopConfig):
        super().__init__(config)
        self.config = config

        self._receiver = None
        self._connected = False
        self._robot = None  # set via set_robot()

        self._C2R = np.asarray(config.controller_to_robot_rotation, dtype=np.float64)

        # Relative anchors (controller + robot), same as URTeleop
        self._first_controller_reading = True
        self._T_ctrl_t0: np.ndarray | None = None
        self._R_ctrl_t0: np.ndarray | None = None
        self._T_robot_t0: np.ndarray | None = None
        self._R_robot_t0: np.ndarray | None = None

        self._enabled = False
        self._prev_enable_btn = False
        self._gripper_open = True
        self._prev_grip = False

        self._safety_hold_active = False
        self._last_safety_print_time = 0.0


    # ------------------------------------------------------------------
    # Robot binding (needed for absolute TCP)
    # ------------------------------------------------------------------

    def set_robot(self, robot) -> None:
        """
        robot must implement get_tcp_pose() -> sequence of 6 floats
        [x, y, z, rx, ry, rz] (UR axis-angle), e.g. your lerobot UR5.
        """
        if not hasattr(robot, "get_tcp_pose"):
            raise TypeError("robot must implement get_tcp_pose() -> [x,y,z,rx,ry,rz]")
        self._robot = robot

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        names = {
            "eef_x": 0,
            "eef_y": 1,
            "eef_z": 2,
            "eef_rx": 3,
            "eef_ry": 4,
            "eef_rz": 5,
        }
        if self.config.use_gripper:
            names["gripper.pos"] = 6
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self._receiver is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from .udp_receiver import UDPControllerReceiver

        self._receiver = UDPControllerReceiver(
            ip=self.config.udp_ip,
            port=self.config.udp_port,
        )
        if hasattr(self._receiver, "connect"):
            self._receiver.connect()
        self._connected = True
        self._reset_controller_anchor()
        logger.info(
            "%s connected (UDP %s:%s, hand=%s)",
            self,
            self.config.udp_ip,
            self.config.udp_port,
            self.config.dominant_hand,
        )
        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._receiver is not None:
            if hasattr(self._receiver, "close"):
                self._receiver.close()
            elif hasattr(self._receiver, "disconnect"):
                self._receiver.disconnect()
            self._receiver = None
        self._connected = False
        self._enabled = False
        self._reset_controller_anchor()
        logger.info("%s disconnected", self)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # Anchors (URTeleop.reset_teleop_state)
    # ------------------------------------------------------------------

    def _reset_controller_anchor(self) -> None:
        self._first_controller_reading = True
        self._T_ctrl_t0 = None
        self._R_ctrl_t0 = None

    def reset_teleop_state(self) -> None:
        """Re-anchor robot TCP + next controller reading (call on enable)."""
        if self._robot is None:
            raise RuntimeError(
                "MetaQuestTeleop.set_robot(robot) must be called before enable/get_action"
            )
        tcp = np.asarray(self._robot.get_tcp_pose(), dtype=np.float64)
        self._T_robot_t0 = tcp[:3].copy()
        self._R_robot_t0 = R.from_rotvec(tcp[3:]).as_matrix()
        self._reset_controller_anchor()
        self._safety_hold_active = False
        logger.info("%s re-anchored to TCP %s", self, np.round(tcp, 3).tolist())

    # ------------------------------------------------------------------
    # VR I/O
    # ------------------------------------------------------------------

    def _read_controllers(self) -> tuple[dict | None, dict | None]:
        pose_data, button_data = self._receiver.receive_once()
        if pose_data is None or button_data is None:
            return None, None

        right = {
            "transform": pose_data.get("r"),
            "trigger": button_data.get("rightTrig"),
            "button": [button_data.get("A"), button_data.get("B")],
            "grip": _button_flag(
                button_data, "rightGrip", "rightGripPressed", "grip", "gripPressed"
            ),
        }
        left = {
            "transform": pose_data.get("l"),
            "trigger": button_data.get("leftTrig"),
            "button": [button_data.get("X"), button_data.get("Y")],
            "grip": _button_flag(button_data, "leftGrip", "leftGripPressed"),
        }
        return right, left

    # ------------------------------------------------------------------
    # Core: absolute TCP action (URTeleop.teleop_step math)
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        right, left = self._read_controllers()
        if right is None or left is None:
            return self._hold_action(enabled=False)

        if self.config.dominant_hand == "left":
            hand, other = left, right
        else:
            hand, other = right, left

        if hand["transform"] is None:
            return self._hold_action(enabled=False)

        # Enable / pause
        other_btns = other.get("button") or [False, False]

        enable_btn = bool(other_btns[0]) if len(other_btns) > 0 else False
        pause_btn = bool(other_btns[1]) if len(other_btns) > 1 else False

        grip_active = bool(hand.get("grip", False))


        # Y button pause
        if pause_btn:
            self._enabled = False


        # X button enable
        if enable_btn and not self._prev_enable_btn:
            if self.config.reanchor_on_enable:
                self.reset_teleop_state()
            self._enabled = True


        # Right grip rising edge = enable
        if grip_active and not self._prev_grip:
            if self.config.reanchor_on_enable:
                self.reset_teleop_state()
            self._enabled = True


        # Right grip falling edge = pause
        if self._prev_grip and not grip_active:
            self._enabled = False
            if self._robot is not None:
                try:
                    self._robot.servo_stop()
                except Exception:
                    pass


        self._prev_enable_btn = enable_btn
        self._prev_grip = grip_active

        # Gripper from trigger
        gripper_val = 0.0 if self._gripper_open else 1.0
        if self.config.use_gripper:
            trig = 0.0
            if isinstance(hand["trigger"], (list, tuple)):
                trig = float(hand["trigger"][0]) if hand["trigger"] else 0.0
            elif hand["trigger"] is not None:
                trig = float(hand["trigger"])
            if trig > self.config.trigger_close_threshold:
                self._gripper_open = False
            else:
                self._gripper_open = True
            gripper_val = 0.0 if self._gripper_open else 1.0

        if not self._enabled:
            return self._hold_action(enabled=False, gripper=gripper_val)

        if self._T_robot_t0 is None or self._R_robot_t0 is None:
            self.reset_teleop_state()

        # --- Controller pose in robot frame (URTeleop.teleop_step) ---
        tf = np.asarray(hand["transform"], dtype=np.float64)
        T_raw = tf[:3, 3].copy()
        R_raw = tf[:3, :3].copy()

        T_ctrl = self._C2R @ T_raw
        # Orientation basis change applied on relative rotation below

        if self._first_controller_reading:
            self._T_ctrl_t0 = T_ctrl.copy()
            self._R_ctrl_t0 = R_raw.copy()
            self._first_controller_reading = False

        T_controller = (T_ctrl - self._T_ctrl_t0) * self.config.translation_motion_scale

        delta_R_controller = R_raw @ self._R_ctrl_t0.T
        delta_R_swapped = self._C2R @ delta_R_controller @ self._C2R.T

        roll, pitch, yaw = R.from_matrix(delta_R_swapped).as_euler("xyz", degrees=False)
        roll *= self.config.orientation_motion_scale
        pitch *= self.config.orientation_motion_scale
        yaw *= self.config.orientation_motion_scale
        delta_R_swapped = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

        R_target = delta_R_swapped @ self._R_robot_t0
        T_target = self._T_robot_t0 + T_controller
        rotvec = R.from_matrix(R_target).as_rotvec()

        target = np.zeros(6, dtype=np.float64)
        target[:3] = T_target
        target[3:] = rotvec

        # Task-space safety gate (optional; same as URTeleop)
        if not self._is_target_within_task_space_thresholds(target):
            return self._hold_action(enabled=True, gripper=gripper_val)

        action: RobotAction = {
            "eef_x": float(target[0]),
            "eef_y": float(target[1]),
            "eef_z": float(target[2]),
            "eef_rx": float(target[3]),
            "eef_ry": float(target[4]),
            "eef_rz": float(target[5]),
            "enabled": True,
        }
        if self.config.use_gripper:
            action["gripper.pos"] = float(gripper_val)
        return action

    def _hold_action(self, enabled: bool = False, gripper: float | None = None) -> RobotAction:
        """
        When disabled / unsafe: return *current* TCP so servoL holds pose
        (if caller still streams), plus gripper.
        """
        if self._robot is not None:
            try:
                tcp = np.asarray(self._robot.get_tcp_pose(), dtype=np.float64)
            except Exception:
                tcp = np.zeros(6)
        else:
            tcp = np.zeros(6)

        g = gripper if gripper is not None else (1.0 if self._gripper_open else 0.0)
        action: RobotAction = {
            "eef_x": float(tcp[0]),
            "eef_y": float(tcp[1]),
            "eef_z": float(tcp[2]),
            "eef_rx": float(tcp[3]),
            "eef_ry": float(tcp[4]),
            "eef_rz": float(tcp[5]),
            "enabled": enabled,
        }
        if self.config.use_gripper:
            action["gripper.pos"] = float(g)
        return action

    def _is_target_within_task_space_thresholds(self, target: np.ndarray) -> bool:
        if self._robot is None:
            return True
        try:
            current = np.asarray(self._robot.get_tcp_pose(), dtype=np.float64)
        except Exception:
            return True

        pos_err = float(np.linalg.norm(target[:3] - current[:3]))
        ori_err = float(
            (R.from_rotvec(target[3:]) * R.from_rotvec(current[3:]).inv()).magnitude()
        )

        ratio = self.config.safety_release_ratio
        if self._safety_hold_active:
            pos_lim = self.config.max_ee_position_delta_m * ratio
            ori_lim = self.config.max_ee_orientation_delta_rad * ratio
        else:
            pos_lim = self.config.max_ee_position_delta_m
            ori_lim = self.config.max_ee_orientation_delta_rad

        if pos_err <= pos_lim and ori_err <= ori_lim:
            if self._safety_hold_active:
                logger.info("[MetaQuest] Safety hold released")
            self._safety_hold_active = False
            return True

        now = time.time()
        if now - self._last_safety_print_time > 0.5:
            logger.warning(
                "[MetaQuest] Safety hold: |Δx|=%.3fm (lim %.3f), |ΔR|=%.1f° (lim %.1f°)",
                pos_err,
                pos_lim,
                np.rad2deg(ori_err),
                np.rad2deg(ori_lim),
            )
            self._last_safety_print_time = now

        if not self._safety_hold_active:
            self._safety_hold_active = True
            try:
                self.reset_teleop_state()
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Events for HIL / recording
    # ------------------------------------------------------------------

    def get_teleop_events(self) -> dict[str, Any]:
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        right, left = self._read_controllers()
        if right is None:
            return {
                TeleopEvents.IS_INTERVENTION: self._enabled,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        btns = right.get("button") or [False, False]
        a_btn = bool(btns[0]) if len(btns) > 0 else False
        b_btn = bool(btns[1]) if len(btns) > 1 else False

        return {
            TeleopEvents.IS_INTERVENTION: self._enabled,
            TeleopEvents.TERMINATE_EPISODE: a_btn or b_btn,
            TeleopEvents.SUCCESS: a_btn,
            TeleopEvents.RERECORD_EPISODE: b_btn,
        }