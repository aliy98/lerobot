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

import numpy as np

from dataclasses import dataclass, field
from typing import Sequence

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloTeleopConfig(TeleoperatorConfig):
    """Configuration for a GELLO Dynamixel leader arm."""

    # Max effort while holding / moving to the UR5 pose (mA). XL330 ~ max 1193.
    goal_current_mA: int = 1000
    # Position P gain (higher = stiffer). Typical useful range 200–800.
    position_p_gain: int = 800

    # Serial port of the U2D2 / FTDI converter
    # e.g. /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAU58VV-if00-port0"

    # Dynamixel joint IDs (arm only, gripper is added via gripper_config)
    joint_ids: Sequence[int] = field(default_factory=lambda: (1, 2, 3, 4, 5, 6))

    # Joint offsets (radians). Must be a multiple of pi/2 per GELLO convention.
    # Run scripts/gello_get_offset.py from the gello_software repo to measure them.
    joint_offsets: Sequence[float] = field(
        default_factory=lambda: (
            np.pi + np.pi/2,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            2 * np.pi
        )
    )

    # Joint direction signs (+1 or -1). UR default: (1, 1, -1, 1, 1, 1)
    joint_signs: Sequence[int] = field(default_factory=lambda: (1, 1, -1, 1, 1, 1))

    # (gripper_joint_id, open_degrees, closed_degrees)
    # Gripper is normalised to [0, 1] where 0=open, 1=closed (GELLO convention).
    gripper_config: tuple[int, float, float] = (7, 170, 150)

    # Optional known start pose (radians, arm only). Used to resolve 2-pi ambiguity
    # of the Dynamixel offsets at connect time.
    start_joints: Sequence[float] | None = None

    # Baudrate for the Dynamixel bus
    baudrate: int = 57600

    # Names used in the action dict (must match the follower robot action space
    # if you want direct joint mirroring).
    joint_names: Sequence[str] = field(
        default_factory=lambda: (
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        )
        #default_factory=lambda: (
        #    "shoulder_pan",
        #    "shoulder_lift",
        #    "elbow_flex",
        #    "wrist_flex",
        #    "wrist_roll",
        #    "wrist_yaw",
        #    "gripper",
        #)
    )

    # Exponential smoothing factor applied to joint readings (0 = no smoothing, 1 = frozen)
    # Matches the alpha used inside gello.robots.dynamixel.DynamixelRobot
    smoothing_alpha: float = 0.99