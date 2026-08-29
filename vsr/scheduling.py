"""Keyframe scheduling for K-strided VSR.

The pipeline trades diffusion-SR calls for cheap RIFE interpolation: only
every K-th frame is super-resolved, and the frames in between are synthesized
from the surrounding HR keyframe pair. The frame count is preserved -- N
frames in, N frames out -- so K is a cost knob, not a frame-rate knob.

Scheduling is expressed as keyframe indices plus per-gap timesteps rather than
fixed-size chunks. RIFE accepts an arbitrary float timestep, so a gap of any
length is handled by the same code path; this is what makes the plan exact for
every (num_frames, k) pair instead of only when (num_frames - 1) % k == 0.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gap:
    """A run of non-keyframes bounded by two super-resolved keyframes.

    targets[i] is the output frame index produced by interpolating between
    `start` and `end` at timesteps[i], where a timestep of t means
    "t of the way from start to end" in the source frame numbering.
    """

    start: int
    end: int
    targets: tuple[int, ...]
    timesteps: tuple[float, ...]

    @property
    def span(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Plan:
    num_frames: int
    k: int
    keyframes: tuple[int, ...]
    gaps: tuple[Gap, ...]

    @property
    def num_interpolated(self) -> int:
        return sum(len(g.targets) for g in self.gaps)

    def validate(self) -> None:
        """Asserts the plan accounts for every output frame exactly once."""
        produced = list(self.keyframes) + [t for g in self.gaps for t in g.targets]
        assert sorted(produced) == list(range(self.num_frames)), (
            f"plan does not cover [0, {self.num_frames}) exactly once: {sorted(produced)}"
        )


def make_gap(start: int, end: int) -> Gap | None:
    """Builds the gap between two keyframes, or None if they are adjacent.

    Shared by whole-clip planning and streaming, so both derive timesteps the
    same way and a tail gap of any width is handled identically.
    """
    if end - start < 2:
        return None
    targets = tuple(range(start + 1, end))
    timesteps = tuple((i - start) / (end - start) for i in targets)
    return Gap(start=start, end=end, targets=targets, timesteps=timesteps)


def is_keyframe(index: int, k: int) -> bool:
    """Keyframe membership by absolute frame index, independent of batching."""
    return index % k == 0


def plan_schedule(num_frames: int, k: int) -> Plan:
    """Builds the keyframe/gap plan for a clip of `num_frames` frames.

    The final frame is always a keyframe: there is no later keyframe to
    interpolate it from. That costs at most one extra SR call per clip and is
    what keeps the tail of the clip temporally correct when the clip length is
    not congruent to 1 mod k.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if k == 1 or num_frames <= 2:
        keyframes = tuple(range(num_frames))
        plan = Plan(num_frames=num_frames, k=k, keyframes=keyframes, gaps=())
        plan.validate()
        return plan

    keyframes = sorted(set(range(0, num_frames, k)) | {num_frames - 1})

    gaps = [
        gap
        for gap in (make_gap(start, end) for start, end in zip(keyframes, keyframes[1:]))
        if gap is not None
    ]

    plan = Plan(
        num_frames=num_frames, k=k, keyframes=tuple(keyframes), gaps=tuple(gaps)
    )
    plan.validate()
    return plan


def group_gaps_by_timesteps(gaps) -> dict[tuple[float, ...], list[Gap]]:
    """Buckets gaps that share a timestep tuple so they can be batched together.

    Gaps of equal span produce identical timesteps, so in practice a clip has
    at most two buckets: the full-length K gaps, and the shorter tail gap.
    """
    grouped: dict[tuple[float, ...], list[Gap]] = {}
    for gap in gaps:
        grouped.setdefault(gap.timesteps, []).append(gap)
    return grouped
