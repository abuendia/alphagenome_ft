#!/usr/bin/env python3
"""Check CUDA package compatibility and execute a tiny JAX GPU convolution."""

import argparse
import ctypes
import importlib.metadata as metadata
import os
import subprocess
import sys
import traceback

from packaging.requirements import Requirement


def package_errors():
    """Check installed CUDA wheels against this JAX plugin's requirements."""
    errors = []
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cudnn-cu12"):
        try:
            print(f"{name}={metadata.version(name)}", flush=True)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    try:
        requirements = metadata.requires("jax-cuda12-plugin") or []
    except metadata.PackageNotFoundError:
        return errors
    for text in requirements:
        requirement = Requirement(text)
        if not requirement.name.startswith("nvidia-"):
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": "with-cuda"}):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing CUDA wheel: {requirement.name}")
            continue
        if installed not in requirement.specifier:
            errors.append(f"{requirement.name}=={installed}; JAX requires {requirement.specifier}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--packages-only", action="store_true")
    args = parser.parse_args()
    print(f"Python: {sys.executable}", flush=True)
    for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        print(f"{name}={os.environ.get(name, '<unset>')}", flush=True)
    errors = package_errors()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    if args.packages_only:
        print("CUDA package requirements passed (GPU execution not checked).")
        return 0
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.free,memory.total", "--format=csv"],
            check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"nvidia-smi unavailable: {exc}", file=sys.stderr)
    try:
        # Read the actually loaded library's version, not just wheel metadata.
        # The C8 launcher puts its private cuDNN directory first in the loader path.
        library = ctypes.CDLL("libcudnn.so.9")
        library.cudnnGetVersion.restype = ctypes.c_size_t
        from jax_cuda12_plugin import _versions
        runtime = library.cudnnGetVersion()
        built = _versions.cudnn_build_version()
        print(f"Loaded cuDNN={runtime}; JAX build cuDNN={built}", flush=True)
        if runtime // 100 < built // 100:
            raise RuntimeError(f"Loaded cuDNN {runtime} is older than JAX build {built}")
        import jax
        import jax.numpy as jnp

        devices = jax.devices("gpu")  # Never silently fall back to CPU.
        print(f"JAX GPUs: {devices}", flush=True)
        if len(devices) < args.num_devices:
            raise RuntimeError(f"Need {args.num_devices} GPUs; JAX sees {len(devices)}")
        convolution = jax.jit(lambda x, w: jax.lax.conv_general_dilated(
            x, w, (1,), "SAME", dimension_numbers=("NWC", "WIO", "NWC")
        ))
        for device in devices[:args.num_devices]:
            with jax.default_device(device):
                # Exercise compilation and cuDNN, not just GPU enumeration.
                jax.random.PRNGKey(0).block_until_ready()
                result = convolution(jnp.ones((1, 64, 4), jnp.bfloat16),
                                     jnp.ones((3, 4, 8), jnp.bfloat16))
                result.block_until_ready()
                assert result.shape == (1, 64, 8)
                assert bool(jnp.all(jnp.isfinite(result)))
        print("JAX GPU initialization and bf16 convolution passed.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("GPU preflight failed before data/model loading. Inspect the CUDA/cuDNN "
              "messages above, including free GPU memory and loaded-library versions.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
