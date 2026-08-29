"""Streaming VSR for callers that feed frames in batches rather than whole clips.

Calling `VSRPipeline` once per batch would restart the keyframe stride at every
batch boundary, because the last frame of each call is forced to be a keyframe.
At K=2 with 2-frame batches that degenerates to `SSSSSS` -- every frame super-
resolved, K doing nothing at all.

`VSRStream` keeps the stride global to the clip by carrying the previous
keyframe across calls. Keyframe membership is decided by absolute frame index
(`index % k == 0`), so it is independent of how the caller happens to chunk
its input.

The cost is latency: a frame between two keyframes cannot be emitted until the
*following* keyframe has arrived, so `push()` may return fewer frames than it
was given (or none). Call `flush()` at the end of the clip to drain the tail.
Frames come out in order, and pushes plus the flush yield exactly as many
frames as went in.
"""

import numpy as np
import torch

from . import frames as frame_io
from .scheduling import is_keyframe, make_gap


class VSRStream:
    """Stateful wrapper around a VSRPipeline for batch-at-a-time input.

    Example:
        stream = pipeline.stream()
        for batch in batches:
            for frame in stream.push(batch):
                write(frame)
        for frame in stream.flush():
            write(frame)
    """

    def __init__(self, pipeline, k: int | None = None):
        self.pipeline = pipeline
        self.k = pipeline.k if k is None else k

        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.k > 1 and pipeline.interpolation is None:
            raise ValueError(
                f"k={self.k} requires the interpolation model, but the pipeline was "
                f"built with k=1. Rebuild it with k>1."
            )
        self.reset()

    def reset(self) -> None:
        """Clears all carried state. Call between independent clips."""
        self._next_index = 0
        # Frames received but not yet emitted, as (absolute index, (1,C,H,W) LR tensor).
        # Only ever holds frames after the most recent keyframe, so at most k-1.
        self._pending: list[tuple[int, torch.Tensor]] = []
        self._previous_keyframe: tuple[int, torch.Tensor] | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self._emitted = 0

    @property
    def frames_received(self) -> int:
        return self._next_index

    @property
    def frames_emitted(self) -> int:
        return self._emitted

    @property
    def frames_buffered(self) -> int:
        """Frames received but not yet emitted; released as keyframes arrive."""
        return len(self._pending)

    def push(self, images) -> list[np.ndarray]:
        """Feeds a batch of LR frames, returning whatever can now be emitted."""
        if not images:
            return []

        batch = frame_io.to_batch(
            images, device=self.pipeline.device, dtype=self.pipeline.dtype
        )

        shape = tuple(batch.shape[1:])
        if self._frame_shape is None:
            self._frame_shape = shape
        elif shape != self._frame_shape:
            raise ValueError(
                f"frame shape changed mid-stream: {self._frame_shape} -> {shape}. "
                "Call reset() between clips."
            )

        for offset in range(batch.shape[0]):
            self._pending.append((self._next_index, batch[offset : offset + 1]))
            self._next_index += 1

        return self._drain(final=False)

    def flush(self) -> list[np.ndarray]:
        """Drains the tail of the clip and resets the stream for the next one."""
        output = self._drain(final=True)
        self.reset()
        return output

    def _drain(self, final: bool) -> list[np.ndarray]:
        """Emits every frame whose bounding keyframes are now known."""
        if not self._pending:
            return []

        anchor_positions = [
            position
            for position, (index, _) in enumerate(self._pending)
            if is_keyframe(index, self.k)
        ]

        # At end of clip the final frame must become a keyframe: there is no
        # later frame to interpolate it from.
        if final and (len(self._pending) - 1) not in anchor_positions:
            anchor_positions.append(len(self._pending) - 1)

        if not anchor_positions:
            return []  # no keyframe yet -- keep buffering

        keyframe_indices = [self._pending[p][0] for p in anchor_positions]
        produced = self.pipeline.run_sr(
            torch.cat([self._pending[p][1] for p in anchor_positions])
        )
        hr = dict(zip(keyframe_indices, produced))

        anchors = list(keyframe_indices)
        if self._previous_keyframe is not None:
            previous_index, previous_hr = self._previous_keyframe
            hr[previous_index] = previous_hr
            anchors.insert(0, previous_index)

        gaps = [
            gap
            for gap in (make_gap(a, b) for a, b in zip(anchors, anchors[1:]))
            if gap is not None
        ]
        if gaps:
            self.pipeline.fill_gaps(hr, gaps)

        # The previous keyframe was emitted on an earlier call; start after it.
        first = anchors[0] + 1 if self._previous_keyframe is not None else anchors[0]
        emitted = [frame_io.to_numpy(hr[i])[0] for i in range(first, anchors[-1] + 1)]
        self._emitted += len(emitted)

        last_position = anchor_positions[-1]
        self._previous_keyframe = (keyframe_indices[-1], hr[keyframe_indices[-1]])
        self._pending = self._pending[last_position + 1 :]

        return emitted
