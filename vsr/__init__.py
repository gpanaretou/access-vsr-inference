"""Video super-resolution inference: diffusion SR + RIFE interpolation + NATTEN.

Symbols are resolved lazily so that `import vsr` -- and in particular
`vsr.scheduling`, which is pure Python -- does not pull in torch and diffusers.
"""

from .scheduling import Gap, Plan, is_keyframe, make_gap, plan_schedule

__all__ = [
    "Gap",
    "InterpolationPipeline",
    "Plan",
    "SuperResolutionPipeline",
    "VSRPipeline",
    "VSRStream",
    "is_keyframe",
    "make_gap",
    "plan_schedule",
]

_LAZY = {
    "VSRPipeline": (".pipeline", "VSRPipeline"),
    "VSRStream": (".streaming", "VSRStream"),
    "SuperResolutionPipeline": (".super_resolution", "SuperResolutionPipeline"),
    "InterpolationPipeline": (".interpolation", "InterpolationPipeline"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
