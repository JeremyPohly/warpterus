"""Warp-FWUAV core modules."""

from .backend import get_xp, has_jax
from .geometry import RectangularWing, WingMesh, transform_vertices
from .kinematics import (
    KinematicsParams,
    flapping_angle,
    pitch_angle,
    deviation_angle,
    right_wing_rotation,
    right_wing_angular_velocity,
)

__all__ = [
    "RectangularWing",
    "WingMesh",
    "transform_vertices",
    "get_xp",
    "has_jax",
    "KinematicsParams",
    "flapping_angle",
    "pitch_angle",
    "deviation_angle",
    "right_wing_rotation",
    "right_wing_angular_velocity",
]
