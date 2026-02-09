"""SO(3) utilities aligned with QS_model.pdf conventions."""

from __future__ import annotations

import numpy as np


def _asarray(xp, v):
    if hasattr(xp, "asarray"):
        return xp.asarray(v)
    return xp.array(v)


def hat(v, xp=np):
    """Skew-symmetric matrix such that hat(v) @ w == v x w."""
    v = _asarray(xp, v).reshape(3)
    return xp.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
    )


def rot_x(angle, xp=np):
    c = xp.cos(angle)
    s = xp.sin(angle)
    return xp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle, xp=np):
    c = xp.cos(angle)
    s = xp.sin(angle)
    return xp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle, xp=np):
    c = xp.cos(angle)
    s = xp.sin(angle)
    return xp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def exp_so3(axis, angle, xp=np):
    """Rodrigues' rotation formula for axis-angle."""
    axis = _asarray(xp, axis).reshape(3)
    n = xp.linalg.norm(axis)
    if hasattr(xp, "where"):
        # Avoid Python branching for JAX compatibility.
        a = axis / xp.where(n == 0.0, 1.0, n)
        a_hat = hat(a, xp=xp)
        c = xp.cos(angle)
        s = xp.sin(angle)
        R = xp.eye(3) + s * a_hat + (1.0 - c) * (a_hat @ a_hat)
        return xp.where(n == 0.0, xp.eye(3), R)
    if n == 0.0:
        return xp.eye(3)
    a = axis / n
    a_hat = hat(a, xp=xp)
    c = xp.cos(angle)
    s = xp.sin(angle)
    return xp.eye(3) + s * a_hat + (1.0 - c) * (a_hat @ a_hat)
