"""Set PRISM's virtual precision and rounding mode from Python.

An instrumented PyTorch rounds every floating-point operation through PRISM.
This module changes the precision and the rounding rule it uses, at any point
during a run, so that a forward pre-hook can lower precision for the duration
of one module and a post-hook can restore it.

The settings are process-wide: they apply to every thread ATen dispatches to,
including workers that have already executed instrumented arithmetic. That is a
property of PRISM's C API rather than of this module -- PRISM keeps precision
and rounding mode in thread-local storage and reconciles changes through a
configuration epoch, which each thread checks on its next operation. Older
PRISM builds exposed setters that wrote a process-wide default no thread ever
re-read, so a precision change was a silent no-op while the getters reported
success; `_bind` refuses to load against such a build rather than let a sweep
run at full precision and report a plausible-looking number.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from contextlib import contextmanager

SR = 0
RN = 1

_MODE_NAMES = {SR: "SR", RN: "RN"}

# binary32 significand width. PRISM asserts on anything outside [2, precision],
# and every sweep in this project stays within binary32.
PRECISION_MIN = 2
PRECISION_MAX_BINARY32 = 24
PRECISION_MAX_BINARY64 = 53

_lib = None


class PrismUnavailable(RuntimeError):
    """PRISM is missing, too old, or not the copy PyTorch rounds through."""


def _libtorch_path() -> str:
    import torch  # imported lazily: this module is useful without it

    path = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so")
    if not os.path.exists(path):
        raise PrismUnavailable(f"cannot find {path}")
    return path


def _prism_path() -> str:
    """Absolute path of the PRISM library libtorch_cpu.so is linked against.

    Binding to a different copy would give this module a distinct set of PRISM
    globals from the one the instrumented arithmetic reads, and every setting
    would be silently ignored, so the path is resolved from libtorch rather
    than guessed.
    """
    libtorch = _libtorch_path()

    try:
        ldd = subprocess.run(
            ["ldd", libtorch], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrismUnavailable(f"could not resolve the libraries of {libtorch}: {exc}")

    match = re.search(r"^\s*libprism-(?:static|dynamic)\.so\s*=>\s*(\S+)", ldd, re.M)
    if match is not None:
        return match.group(1)

    # ldd resolved nothing; say which of the two builds was wanted, if any.
    try:
        needed = subprocess.run(
            ["readelf", "-d", libtorch], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        needed = ""

    wanted = re.search(r"libprism-(static|dynamic)\.so", needed)
    if wanted is None:
        raise PrismUnavailable(
            f"{libtorch} is not linked against PRISM. This PyTorch build is not "
            "Verificarlo-instrumented with the PRISM backend, so per-module "
            "precision scoping cannot work."
        )
    raise PrismUnavailable(
        f"{libtorch} needs libprism-{wanted.group(1)}.so but the loader cannot "
        "find it; check LD_LIBRARY_PATH."
    )


def _bind() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    path = _prism_path()
    lib = ctypes.CDLL(path)

    required = {
        "interflop_prism_set_default_virtual_precision_binary32": ([ctypes.c_int32], None),
        "interflop_prism_set_default_virtual_precision_binary64": ([ctypes.c_int32], None),
        "interflop_prism_get_virtual_precision_binary32": ([], ctypes.c_int32),
        "interflop_prism_get_virtual_precision_binary64": ([], ctypes.c_int32),
        "interflop_prism_set_rounding_mode": ([ctypes.c_int32], None),
        "interflop_prism_get_rounding_mode": ([], ctypes.c_int32),
    }

    for name, (argtypes, restype) in required.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            raise PrismUnavailable(
                f"{path} does not export {name}. This PRISM predates the "
                "configuration epoch, where a runtime precision change reached "
                "only threads that had not yet rounded -- silently leaving the "
                "rest at full precision. Rebuild the image against a current "
                "PRISM and Verificarlo."
            )
        fn.argtypes = argtypes
        fn.restype = restype

    _lib = lib
    return _lib


def _check_precision(t: int, maximum: int) -> None:
    if not isinstance(t, int) or not PRECISION_MIN <= t <= maximum:
        raise ValueError(
            f"precision {t!r} outside [{PRECISION_MIN}, {maximum}]"
        )


def set_precision(t: int, binary64: bool = True) -> None:
    """Round binary32 (and, unless disabled, binary64) at `t` significand bits.

    Raises if the setting does not read back, which is the symptom of being
    bound to a PRISM the instrumented arithmetic does not use.
    """
    lib = _bind()
    _check_precision(t, PRECISION_MAX_BINARY32)

    lib.interflop_prism_set_default_virtual_precision_binary32(t)
    if binary64:
        lib.interflop_prism_set_default_virtual_precision_binary64(t)

    applied = lib.interflop_prism_get_virtual_precision_binary32()
    if applied != t:
        raise RuntimeError(
            f"PRISM precision did not take effect: asked for {t}, "
            f"this thread reports {applied}"
        )


def get_precision() -> int:
    """Precision the calling thread will actually round binary32 at."""
    return _bind().interflop_prism_get_virtual_precision_binary32()


def get_precision_binary64() -> int:
    return _bind().interflop_prism_get_virtual_precision_binary64()


def set_rounding_mode(mode: int) -> None:
    """Select stochastic rounding (`SR`) or untied round-to-nearest (`RN`)."""
    lib = _bind()
    if mode not in _MODE_NAMES:
        raise ValueError(f"rounding mode {mode!r} is neither SR ({SR}) nor RN ({RN})")

    lib.interflop_prism_set_rounding_mode(mode)

    applied = lib.interflop_prism_get_rounding_mode()
    if applied != mode:
        raise RuntimeError(
            f"PRISM rounding mode did not take effect: asked for "
            f"{_MODE_NAMES[mode]}, this thread reports "
            f"{_MODE_NAMES.get(applied, applied)}"
        )


def get_rounding_mode() -> int:
    """Mode the calling thread will actually round with."""
    return _bind().interflop_prism_get_rounding_mode()


@contextmanager
def rounding(t: int | None = None, mode: int | None = None):
    """Round at `t` bits with `mode` inside the block, restoring both after."""
    previous_t = get_precision()
    previous_mode = get_rounding_mode()
    try:
        if t is not None:
            set_precision(t)
        if mode is not None:
            set_rounding_mode(mode)
        yield
    finally:
        set_precision(previous_t)
        set_rounding_mode(previous_mode)


def scoped_rounding(t: int, mode: int | None = None, default_t: int = 24,
                    default_mode: int | None = None):
    """Forward hooks that apply `t` (and `mode`) for one module's forward pass.

    Returns the (pre, post) pair to pass to `register_forward_pre_hook` and
    `register_forward_hook`.
    """

    def pre_hook(module, args):
        set_precision(t)
        if mode is not None:
            set_rounding_mode(mode)

    def post_hook(module, args, output):
        set_precision(default_t)
        if default_mode is not None:
            set_rounding_mode(default_mode)

    return pre_hook, post_hook


def instrument(module, t: int, mode: int | None = None, default_t: int = 24,
               default_mode: int | None = None):
    """Register the hooks of `scoped_rounding` on `module`.

    Returns the handles, so a caller can remove them.
    """
    pre_hook, post_hook = scoped_rounding(t, mode, default_t, default_mode)
    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(post_hook),
    ]


def assert_effective() -> None:
    """Check that a precision change reaches the arithmetic PyTorch executes.

    A read-back through the C API only proves this module and PRISM agree. This
    runs an operation through ATen and checks the result moves with `t`, which
    is what the sweeps depend on: without it, a misconfigured run reports a
    plausible perplexity at full precision and nothing distinguishes it from a
    real measurement.
    """
    import torch

    # Exact in binary32; on the t=8 grid RN rounds it back to 1.0.
    #
    # The tensors are wide on purpose. A one-element operation does not go
    # through the instrumented kernel -- it stays on a scalar path Verificarlo
    # does not see -- so a single-element probe reports failure on a working
    # installation. Anything the sweeps actually measure is far wider than one
    # lane.
    width = 1024
    a = torch.full((width,), 1.0, dtype=torch.float32)
    b = torch.full((width,), float.fromhex("0x1.8p-10"), dtype=torch.float32)

    previous_t = get_precision()
    previous_mode = get_rounding_mode()
    try:
        set_rounding_mode(RN)

        set_precision(24)
        exact = (a + b)[0].item()
        set_precision(8)
        coarse = (a + b)[0].item()
    finally:
        set_precision(previous_t)
        set_rounding_mode(previous_mode)

    if exact == coarse:
        raise RuntimeError(
            "changing PRISM's virtual precision did not change what PyTorch "
            f"computes ({exact!r} at both t=24 and t=8). The arithmetic is not "
            "going through the PRISM this module is bound to, so any sweep "
            "would run at full precision and report a plausible-looking result."
        )
