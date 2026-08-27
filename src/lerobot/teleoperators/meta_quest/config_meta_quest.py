#!/usr/bin/env python

from dataclasses import dataclass, field
import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("meta_quest")
@dataclass
class MetaQuestTeleopConfig(TeleoperatorConfig):
    udp_ip: str = "100.72.159.127"
    udp_port: int = 12345

    dominant_hand: str = "right"  # "right" or "left"

    translation_motion_scale: float = 1.0
    orientation_motion_scale: float = 1.0

    controller_to_robot_rotation: list[list[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    trigger_close_threshold: float = 0.8
    use_gripper: bool = True
    reanchor_on_enable: bool = True

    max_ee_position_delta_m: float = 0.10
    max_ee_orientation_delta_rad: float = float(np.deg2rad(30.0))
    safety_release_ratio: float = 0.8