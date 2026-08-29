# vsr

Inference-only video super-resolution: a single-step diffusion image SR model,
RIFE frame interpolation, and optional NATTEN neighborhood attention.

```python
import numpy as np
from vsr import VSRPipeline

pipeline = VSRPipeline(k=2, natten_layers=3)

lr_frames = [...]            # list of uint8 (H, W, 3) RGB arrays, in order
hr_frames = pipeline(lr_frames)   # list of uint8 (4H, 4W, 3) RGB arrays
```

## The K knob

`K` controls how often the diffusion model actually runs. **The frame count is
always preserved** — N frames in, N frames out. K is a cost/quality knob, not a
frame-rate knob.

| K | Diffusion SR runs on | In-between frames |
|---|----------------------|-------------------|
| 1 | every frame | none |
| 2 | every 2nd frame | 1 synthesized per gap |
| 3 | every 3rd frame | 2 synthesized per gap |

Frames that are not keyframes are synthesized by RIFE from the two surrounding
**super-resolved** keyframes — interpolation happens at HR, after SR, never at LR.

## How scheduling works

Scheduling is expressed as keyframe indices plus per-gap timesteps, not
fixed-size chunks:

```
keyframes = sorted(set(range(0, N, K)) | {N - 1})
# for each consecutive pair (a, b), frame i in (a, b) is RIFE'd at t = (i - a) / (b - a)
```

RIFE takes an arbitrary float timestep, so a gap of any length works through
the same path. The final frame is always a keyframe (there is no later frame to
interpolate it from), which costs at most one extra SR call per clip and keeps
the tail temporally correct when `N` is not congruent to 1 mod K.

`vsr.scheduling` is pure Python with no torch dependency, so you can inspect the
plan for a clip without loading anything:

```python
from vsr.scheduling import plan_schedule

plan = plan_schedule(num_frames=100, k=3)
print(plan.keyframes)         # frames the diffusion model will run on
print(plan.num_interpolated)  # frames RIFE will synthesize
```

## Streaming (batch-at-a-time input)

**Do not call the pipeline once per batch.** Each call forces its own last
frame to be a keyframe, so the stride restarts at every boundary — at K=2 with
2-frame batches you get `SSSSSS`, every frame super-resolved, and K buys nothing.

Use `VSRStream`, which carries the previous keyframe across calls and decides
keyframes by absolute frame index:

```python
stream = pipeline.stream()

for batch in batches:
    for frame in stream.push(batch):
        write(frame)

for frame in stream.flush():      # drains the tail, then resets
    write(frame)
```

Diffusion calls over a 24-frame clip at K=2 (whole-clip optimum is 13):

| Batch size | One call per batch | `VSRStream` |
|---|---|---|
| 2 | 24 | **13** |
| 3 | 16 | **13** |
| 4 | 18 | **13** |
| 8 | 15 | **13** |

Streamed output is byte-identical to processing the whole clip in one call, for
every batch size — that equivalence is what `tests/test_streaming.py` asserts.

**`push()` returns fewer frames than you give it.** A frame between two
keyframes cannot be emitted until the *following* keyframe arrives, so output
lags input by up to K-1 frames and an early `push()` may return `[]`. Frames
still come out in order, and pushes plus `flush()` yield exactly as many frames
as went in. `frames_received`, `frames_emitted` and `frames_buffered` expose the
progress. `flush()` resets the stream, so the same object can process the next
clip; call `reset()` to abandon one partway.

## Input format

Canonical input is a list of **uint8 `(H, W, 3)` RGB** arrays. Float arrays in
`[0, 1]` and torch tensors are also accepted. `(C, H, W)` is detected and
transposed. All frames in a call must share a resolution.

Output is always uint8 `(H, W, 3)`. 0–255 float input is rejected rather than
silently interpreted as `[0, 1]`.

Any resolution works — both stages pad internally (SR to a multiple of 16, RIFE
to a multiple of 64) and crop back.

## Weights

Fetched from the Hugging Face Hub on first use into `~/.cache/vsr` (override
with `cache_dir=`, or point at local files with `sr_model_ckpt=`,
`sr_decoder_ckpt=`, `interpolation_model_dir=`). The base SD-2.1 UNet is also
pulled from the Hub by `diffusers`. RIFE weights are only fetched when `k > 1`.

## Options

| Argument | Default | Notes |
|---|---|---|
| `k` | `1` | Keyframe stride. Also overridable per call: `pipeline(frames, k=3)`. |
| `natten_layers` | `0` | Replace this many 240-dim attention layers with NATTEN. Requires the `natten` extra; without it the pipeline prints a notice and keeps standard attention. |
| `sr_batch_size` | `4` | Frames per diffusion call. Larger is markedly cheaper per frame; bounded by VRAM. |
| `interpolation_batch_size` | `8` | Keyframe pairs per RIFE call. |
| `compile_decoder` | `False` | `torch.compile` the decoder. Recompiles on every new input resolution — only worth it for a fixed-resolution workload. |
| `device` / `dtype` | auto | Defaults to CUDA with bf16 (fp16 if bf16 is unsupported), else CPU/fp32. |

## Memory

A clip is processed in one call and all HR frames are held on-device until it
returns. At 4x, a 720p output frame is ~5.3 MB in bf16, so a 300-frame clip is
roughly 1.6 GB on top of model weights. Split long clips at the call boundary;
because the first and last frames of every call are keyframes, splitting is
seam-free (at the cost of one extra SR call per split).

## Layout

```
vsr/
  pipeline.py            VSRPipeline — orchestration, batching, numpy boundary
  streaming.py           VSRStream — stride-preserving batch-at-a-time input
  scheduling.py          keyframe/gap planning (pure Python, no torch)
  frames.py              numpy <-> tensor conversion
  padding.py             pad/unpad helpers
  weights.py             Hugging Face weight resolution
  super_resolution/      pruned SD-2.1 UNet + half-decoder, NATTEN processor
  interpolation/         RIFE v4.25 (inference path only)
```

## Tests

```bash
uv sync --group dev
uv run pytest
```

Nothing in the suite needs a GPU, and nothing downloads on its own. The routing
test drives the pipeline with stub models over a clip whose frame *i* is a solid
value *i*, so a misrouted frame or a wrong timestep shows up as a wrong pixel.

A few RIFE tests — real-checkpoint key coverage and interpolated motion — are
skipped unless the weights are already cached. Prime them with:

```bash
uv run python -c "from vsr.weights import ensure_interpolation_weights as e; e()"
```
