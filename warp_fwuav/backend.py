"""Backend selection helpers for NumPy/JAX."""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jnp = None


def get_xp(backend: str | None):
    """Return numpy-like module based on backend string."""
    if backend is None or backend in ("numpy", "np"):
        return np
    if backend in ("jax", "jnp"):
        if jnp is None:
            raise RuntimeError("JAX requested but not available.")
        return jnp
    raise ValueError(f"Unknown backend: {backend}")


def has_jax() -> bool:
    return jnp is not None
