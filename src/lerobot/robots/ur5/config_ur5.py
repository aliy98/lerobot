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

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("ur5")
@dataclass
class UR5Config(RobotConfig):
    robot_ip: str = "192.168.1.100"
    gripper_port: int = 63352
    control_hz: float = 90.0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None