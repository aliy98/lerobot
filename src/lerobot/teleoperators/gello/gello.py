#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
GELLO leader-arm teleoperator for LeRobot.

Wraps the Dynamixel-based GELLO hardware (same stack as
https://github.com/tlpss/gello_software) and exposes the standard
LeRobot Teleoperator interface.

Action format (matches leader-arm convention):
  {joint_name}.pos : float   (radians for arm joints, [0,1] for gripper)

Typical usage with a UR / any joint-space follower:

  from lerobot.teleoperators.gello import GelloTeleop, GelloTeleopConfig

  cfg = GelloTeleopConfig(
      port="/dev/serial/by-id/usb-FTDI_... ",
      joint_offsets=(...),   # from gello_get_offset.py
      joint_signs=(1, 1, -1, 1, 1, 1),
      gripper_config=(7, 195, 153),
  )
  teleop = GelloTeleop(cfg)
  teleop.connect()

  while True:
      action = teleop.get_action()
      robot.send_action(action)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_gello import GelloTeleopConfig

logger = logging.getLogger(__name__)


# Dynamixel control table (Protocol 2.0)
ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102
ADDR_POSITION_P_GAIN = 84
CURRENT_BASED_POSITION_MODE = 5   # better for adjustable effort
POSITION_CONTROL_MODE = 3

# XL330-M288: datasheet current limit ≈ 1193 mA.
# Start moderate; increase only if the PSU can supply it.
DEFAULT_GOAL_CURRENT_MA = 1000     # try 400–800
DEFAULT_POSITION_P_GAIN = 800     # factory is often lower; try 200–800


class GelloTeleop(Teleoperator):
    """
    Teleoperator backed by a GELLO Dynamixel leader arm.

    Reads joint state (including normalised gripper) and returns it as a
    flat action dict so any LeRobot Robot that accepts joint positions can
    be driven in leader-follower mode.

    Also supports active position control (enable_torque / write_goal_positions)
    so HIL can smoothly move the leader to the follower pose before takeover.
    """

    config_class = GelloTeleopConfig
    name = "gello"

    def __init__(self, config: GelloTeleopConfig):
        super().__init__(config)
        self.config = config

        self._robot = None  # gello.robots.dynamixel.DynamixelRobot
        self._connected = False
        self._last_pos: np.ndarray | None = None
        self._torque_enabled = False

    def _write_two_byte(self, dxl_id: int, address: int, value: int) -> None:
        driver = self._robot._driver
        lo = value & 0xFF
        hi = (value >> 8) & 0xFF
        # packetHandler is on the real driver
        ph = driver._packetHandler
        port = driver._portHandler
        with driver._lock:
            dxl_comm_result, dxl_error = ph.write2ByteTxRx(port, dxl_id, address, value)
            if dxl_comm_result != 0 or dxl_error != 0:
                raise RuntimeError(
                    f"write2Byte id={dxl_id} addr={address} val={value} "
                    f"comm={dxl_comm_result} err={dxl_error}"
                )


    def _configure_effort(self, goal_current_mA: int = DEFAULT_GOAL_CURRENT_MA,
                        position_p_gain: int = DEFAULT_POSITION_P_GAIN) -> None:
        """Call only while torque is OFF."""
        driver = self._robot._driver
        if getattr(driver, "_is_fake", False):
            return

        # Operating mode (1 byte)
        if hasattr(driver, "set_operating_mode"):
            driver.set_operating_mode(CURRENT_BASED_POSITION_MODE)
        else:
            for dxl_id in self._robot._joint_ids:
                ph = driver._packetHandler
                port = driver._portHandler
                with driver._lock:
                    ph.write1ByteTxRx(port, dxl_id, ADDR_OPERATING_MODE, CURRENT_BASED_POSITION_MODE)

        for dxl_id in self._robot._joint_ids:
            # Goal Current (mA, signed 2-byte; positive is enough for position mode)
            self._write_two_byte(dxl_id, ADDR_GOAL_CURRENT, int(goal_current_mA))
            # Stiffer position loop
            self._write_two_byte(dxl_id, ADDR_POSITION_P_GAIN, int(position_p_gain))

        logger.info(
            "%s effort configured: mode=5, goal_current=%d mA, P_gain=%d",
            self, goal_current_mA, position_p_gain,
        )

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_names}

    @property
    def feedback_features(self) -> dict:
        # GELLO hardware has no force-feedback path in the open-source stack
        return {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self._robot is not None

    @property
    def is_calibrated(self) -> bool:
        # Offsets are supplied via config (measured with gello_get_offset.py)
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Prefer the installed gello package; fall back is not provided
        # because the Dynamixel driver lives inside gello_software.
        try:
            from gello.robots.dynamixel import DynamixelRobot
        except ImportError as e:
            raise ImportError(
                "gello package not found. Install gello_software:\n"
                "  git clone https://github.com/tlpss/gello_software\n"
                "  cd gello_software && pip install -e . "
                "&& pip install -e third_party/DynamixelSDK/python"
            ) from e

        start_joints = None
        if self.config.start_joints is not None:
            start_joints = np.asarray(self.config.start_joints, dtype=np.float64)

        self._robot = DynamixelRobot(
            joint_ids=list(self.config.joint_ids),
            joint_offsets=list(self.config.joint_offsets),
            joint_signs=list(self.config.joint_signs),
            real=True,
            port=self.config.port,
            baudrate=self.config.baudrate,
            gripper_config=self.config.gripper_config,
            start_joints=start_joints,
        )
        # Leader should be torque-free so the human can move it
        self._robot.set_torque_mode(False)
        self._torque_enabled = False
        self._connected = True
        self._last_pos = None
        logger.info("%s connected on %s", self, self.config.port)

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        # GELLO offsets are measured offline with scripts/gello_get_offset.py
        # and stored in the config. Nothing interactive needed at runtime.
        pass

    def configure(self) -> None:
        if self._robot is not None:
            self._robot.set_torque_mode(False)
            self._torque_enabled = False

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot._torque_on = True  # force real write
                self._robot.set_torque_mode(False)
            except Exception as e:
                logger.warning("%s: torque off on disconnect failed: %s", self, e)
                try:
                    self._robot._driver.set_torque_mode(False)
                except Exception:
                    pass
            try:
                driver = getattr(self._robot, "_driver", None)
                if driver is not None and hasattr(driver, "close"):
                    driver.close()
            except Exception:
                pass
            self._robot = None
        self._connected = False
        self._torque_enabled = False
        self._last_pos = None
        logger.info("%s disconnected", self)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Optional future: command a weak holding torque or vibration
        pass

    # ------------------------------------------------------------------
    # Motor control (used by HIL teleop_smooth_move_to)
    # ------------------------------------------------------------------

    @check_if_not_connected
    def enable_torque(self) -> None:
        if self._robot is None:
            return

        try:
            self._robot.set_torque_mode(False)
        except Exception:
            pass

        # Optional: read from config if you add fields there
        goal_mA = getattr(self.config, "goal_current_mA", DEFAULT_GOAL_CURRENT_MA)
        p_gain = getattr(self.config, "position_p_gain", DEFAULT_POSITION_P_GAIN)

        try:
            self._configure_effort(goal_current_mA=goal_mA, position_p_gain=p_gain)
        except Exception as e:
            logger.warning("%s: effort config failed (%s) – falling back to mode 3", self, e)
            try:
                self._robot._driver.set_operating_mode(POSITION_CONTROL_MODE)
            except Exception:
                pass

        self._robot.set_torque_mode(True)
        self._torque_enabled = True
        logger.info("%s torque enabled", self)


    @check_if_not_connected
    def disable_torque(self) -> None:
        """Make the arm backdrivable again (human can move it freely)."""
        if self._robot is None:
            return

        # Force the flag so the early-return inside DynamixelRobot cannot skip the write.
        try:
            self._robot._torque_on = True
        except Exception:
            pass

        try:
            self._robot.set_torque_mode(False)
        except Exception as e:
            logger.warning("%s: set_torque_mode(False) failed: %s – trying driver directly", self, e)
            try:
                self._robot._driver.set_torque_mode(False)
            except Exception as e2:
                logger.warning("%s: driver.set_torque_mode(False) also failed: %s", self, e2)

        self._torque_enabled = False
        logger.info("%s torque disabled", self)


    def _action_dict_to_joint_array(self, action: dict[str, Any]) -> np.ndarray:
        names = list(self.config.joint_names)
        q = np.zeros(len(names), dtype=np.float64)

        for i, name in enumerate(names):
            key = f"{name}.pos"
            if key not in action:
                raise KeyError(f"Missing key '{key}' in action for write_goal_positions")
            val = float(action[key])
            if not np.isfinite(val):
                raise ValueError(f"Non-finite value for '{key}': {val}")
            q[i] = val

        if self._robot.gripper_open_close is not None and len(q) == len(names):
            open_rad, close_rad = self._robot.gripper_open_close
            g = float(np.clip(q[-1], 0.0, 1.0))
            q[-1] = g * (close_rad - open_rad) + open_rad

        return q


    def _joint_array_to_driver_command(self, joint_state: np.ndarray) -> np.ndarray:
        offsets = np.asarray(self._robot._joint_offsets, dtype=np.float64)
        signs = np.asarray(self._robot._joint_signs, dtype=np.float64)
        return joint_state / signs + offsets


    @check_if_not_connected
    def write_goal_positions(self, action: dict[str, Any]) -> None:
        if self._robot is None:
            return
        if not self._torque_enabled:
            raise RuntimeError(
                "Torque must be enabled before write_goal_positions. "
                "Call enable_torque() first."
            )

        q = self._action_dict_to_joint_array(action)
        raw = self._joint_array_to_driver_command(q)

        if np.any(np.abs(raw) > 50.0):
            logger.warning(
                "%s: large raw command |q|=%s – check joint_offsets/signs",
                self,
                np.round(raw, 3).tolist(),
            )

        driver = self._robot._driver
        lock = getattr(driver, "_lock", None)
        try:
            if lock is not None:
                with lock:
                    driver.set_joints(raw.tolist())
            else:
                driver.set_joints(raw.tolist())
        except RuntimeError as e:
            logger.warning("%s: set_joints failed (%s) – retrying once", self, e)
            import time
            time.sleep(0.02)
            if lock is not None:
                with lock:
                    driver.set_joints(raw.tolist())
            else:
                driver.set_joints(raw.tolist())

        self._last_pos = q.copy()

    # ------------------------------------------------------------------
    # Core teleop
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Read current GELLO joint state and return it as a LeRobot action dict.

        Arm joints are in radians.
        Gripper is in [0, 1] (0 = open, 1 = closed) following GELLO convention.
        """
        # DynamixelRobot.get_joint_state already applies offsets, signs,
        # gripper normalisation and exponential smoothing.
        q = self._robot.get_joint_state()  # shape (n_arm + 1,)

        # Optional extra smoothing on top of the internal alpha
        alpha = float(np.clip(self.config.smoothing_alpha, 0.0, 1.0))
        if self._last_pos is None:
            self._last_pos = q.copy()
        else:
            q = self._last_pos * (1.0 - alpha) + q * alpha
            self._last_pos = q

        names = list(self.config.joint_names)
        if len(q) != len(names):
            raise RuntimeError(
                f"GELLO returned {len(q)} values but joint_names has {len(names)}. "
                "Check joint_ids / gripper_config / joint_names in the config."
            )

        action: RobotAction = {
            f"{name}.pos": float(val) for name, val in zip(names, q)
        }
        return action