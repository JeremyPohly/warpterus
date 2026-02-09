import importlib
import os


def test_imports():
    for mod in ("jax", "jaxlib", "warp", "xlb", "yaml", "numpy", "scipy"):
        importlib.import_module(mod)


def test_jax_basic():
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    assert devices, "No JAX devices detected."

    require_gpu = os.getenv("REQUIRE_GPU", "0") == "1"
    has_gpu = any(d.platform in ("gpu", "cuda") for d in devices)
    if require_gpu:
        assert has_gpu, f"REQUIRE_GPU=1 but devices are: {devices}"

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sin(x).sum()
    assert float(y) != 0.0


def test_warp_init():
    import warp as wp

    if hasattr(wp, "init"):
        wp.init()
    if hasattr(wp, "get_preferred_device"):
        dev = wp.get_preferred_device()
        assert dev is not None
