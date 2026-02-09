"""Wing kinematics following doc/QS_model.pdf."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .so3 import rot_x, rot_y, rot_z

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class KinematicsParams:
    """Kinematic parameters for the QS wing model (right wing).

    Angles are in radians. Frequency is in Hz.
    """

    frequency: float
    # Flapping angle ϕ(t) parameters (Eq. 9)
    phi_m: float
    phi_0: float
    phi_K: float
    # Pitch angle θ(t) parameters (Eq. 10)
    theta_m: float
    theta_0: float
    theta_C: float
    theta_a: float
    # Deviation angle ψ(t) parameters (Eq. 11)
    psi_m: float
    psi_0: float
    psi_N: float
    psi_a: float
    # Stroke plane angle β (Eq. 3)
    beta: float = 0.0


# Optional: make the dataclass a JAX pytree for differentiation.
try:  # pragma: no cover - optional dependency
    from jax.tree_util import register_pytree_node_dataclass  # type: ignore

    register_pytree_node_dataclass(KinematicsParams)
except Exception:
    pass


def _safe_sin_ratio(k, value, xp=np):
    """Return asin(k*value)/asin(k) with a stable limit as k->0."""
    k = xp.asarray(k)
    value = xp.asarray(value)
    if hasattr(xp, "where"):
        eps = 1.0e-8
        return xp.where(xp.abs(k) < eps, value, xp.arcsin(k * value) / xp.arcsin(k))
    if abs(k) < 1.0e-8:
        return value
    return xp.arcsin(k * value) / xp.arcsin(k)


def flapping_angle(t, p: KinematicsParams, xp=np):
    """Eq. (9): smoothed triangular waveform for flapping angle ϕ(t)."""
    return p.phi_m * _safe_sin_ratio(p.phi_K, xp.cos(TWO_PI * p.frequency * t), xp=xp) + p.phi_0


def flapping_angle_dot(t, p: KinematicsParams, xp=np):
    """Time derivative of ϕ(t)."""
    w = TWO_PI * p.frequency
    eps = 1.0e-8
    denom = xp.sqrt(1.0 - (p.phi_K * xp.cos(w * t)) ** 2)
    num = p.phi_m * (-p.phi_K * w * xp.sin(w * t))
    if hasattr(xp, "where"):
        return xp.where(
            xp.abs(p.phi_K) < eps,
            -p.phi_m * w * xp.sin(w * t),
            num / (xp.arcsin(p.phi_K) * denom),
        )
    if abs(p.phi_K) < eps:
        return -p.phi_m * w * xp.sin(w * t)
    return num / (xp.arcsin(p.phi_K) * denom)


def pitch_angle(t, p: KinematicsParams, xp=np):
    """Eq. (10): hyperbolic waveform for pitch angle θ(t)."""
    s = xp.sin(TWO_PI * p.frequency * t + p.theta_a)
    if hasattr(xp, "where"):
        eps = 1.0e-8
        return xp.where(
            xp.abs(p.theta_C) < eps,
            p.theta_m * s + p.theta_0,
            p.theta_m * xp.tanh(p.theta_C * s) / xp.tanh(p.theta_C) + p.theta_0,
        )
    if abs(p.theta_C) < 1.0e-8:
        return p.theta_m * s + p.theta_0
    return p.theta_m * xp.tanh(p.theta_C * s) / xp.tanh(p.theta_C) + p.theta_0


def pitch_angle_dot(t, p: KinematicsParams, xp=np):
    """Time derivative of θ(t)."""
    w = TWO_PI * p.frequency
    c = xp.cos(w * t + p.theta_a)
    if hasattr(xp, "where"):
        eps = 1.0e-8
        sech2 = 1.0 / xp.cosh(p.theta_C * xp.sin(w * t + p.theta_a)) ** 2
        return xp.where(
            xp.abs(p.theta_C) < eps,
            p.theta_m * w * c,
            p.theta_m * (p.theta_C * w * c) * sech2 / xp.tanh(p.theta_C),
        )
    if abs(p.theta_C) < 1.0e-8:
        return p.theta_m * w * c
    sech2 = 1.0 / xp.cosh(p.theta_C * xp.sin(w * t + p.theta_a)) ** 2
    return p.theta_m * (p.theta_C * w * c) * sech2 / xp.tanh(p.theta_C)


def deviation_angle(t, p: KinematicsParams, xp=np):
    """Eq. (11): sinusoidal deviation angle ψ(t)."""
    return p.psi_m * xp.cos(TWO_PI * p.psi_N * p.frequency * t + p.psi_a) + p.psi_0


def deviation_angle_dot(t, p: KinematicsParams, xp=np):
    """Time derivative of ψ(t)."""
    w = TWO_PI * p.psi_N * p.frequency
    return -p.psi_m * w * xp.sin(w * t + p.psi_a)


def right_wing_rotation(t, p: KinematicsParams, xp=np):
    """Right wing rotation QR (Eq. 3), mapping FR -> FB."""
    phi = flapping_angle(t, p, xp=xp)
    theta = pitch_angle(t, p, xp=xp)
    psi = deviation_angle(t, p, xp=xp)

    # QR = exp(beta e2^) exp(phi e1^) exp(-psi e3^) exp(theta e2^)
    return rot_y(p.beta, xp=xp) @ rot_x(phi, xp=xp) @ rot_z(-psi, xp=xp) @ rot_y(theta, xp=xp)


def right_wing_angular_velocity(t, p: KinematicsParams, xp=np):
    """Angular velocity ΩR resolved in FR (Eq. 5)."""
    phi = flapping_angle(t, p, xp=xp)
    theta = pitch_angle(t, p, xp=xp)
    psi = deviation_angle(t, p, xp=xp)
    phi_d = flapping_angle_dot(t, p, xp=xp)
    theta_d = pitch_angle_dot(t, p, xp=xp)
    psi_d = deviation_angle_dot(t, p, xp=xp)

    cpsi = xp.cos(psi)
    spsi = xp.sin(psi)
    cth = xp.cos(theta)
    sth = xp.sin(theta)

    m = xp.array(
        [
            [cpsi * cth, 0.0, sth],
            [spsi, 1.0, 0.0],
            [cpsi * sth, 0.0, -cth],
        ],
    )
    rates = xp.array([phi_d, theta_d, psi_d])
    return m @ rates
