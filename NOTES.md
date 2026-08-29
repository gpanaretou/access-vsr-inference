# Notes on the extraction

Extracted from the `access_vsr_pipeline` research repo. Behaviour is preserved
except where noted below.

## Fixed

**Tail-chunk scheduling.** The research `process_clip` used
`range(0, N - 1, kfi)` chunks and interpolated every chunk with `multi=kfi`.
That is only correct when `(N - 1) % K == 0`; otherwise the final short chunk
still produced `K - 1` in-between frames, so the clip came out too long with
in-between frames at temporal positions that do not exist in the source. For
N=6, K=2 it returned 7 frames, one of them synthesized at t=0.5 between source
frames 4 and 5. Replaced with keyframe/timestep scheduling, which is exact for
every (N, K) — see `vsr/scheduling.py` and the regression tests.

**Numpy input.** `_prepare_image` called `torch.from_numpy(...)` with no
HWC→CHW transpose and no `/255`, so a standard `(H, W, 3)` uint8 array became a
`(1, H, W, 3)` tensor with 0–255 values. Now handled in `vsr/frames.py` with an
explicit documented contract and validation.

**Reflect padding on small frames.** `F.pad(mode="reflect")` requires each pad
to be smaller than its dimension, so frames below the pad divisor crashed. Falls
back to `replicate` only in cases that previously raised; sizes that already
worked are bit-identical.

**In-place output conversion.** `Tensor.float()` returns `self` when the
tensor is already float32, so converting results to numpy with in-place
`clamp_`/`mul_` scaled the source frame by 255 behind the caller's back. All
RIFE output is float32, so this affected every interpolated frame. Now
out-of-place, with a regression test.

**NATTEN `dilation=0`.** The processor's default meant "derive dilation from
the first feature map seen", which then stuck for every other layer sharing the
instance. Default is now an explicit `1` (what the pipeline always passed).

**Device binding.** `interpolation_pipeline` asserted CUDA, and
`warplayer`/`IFNet_HDv3` captured a global `device` at import. Device is now
threaded through; the warp grid cache is keyed on device and dtype.

**`torch.compile`.** Was an unconditional decorator on `RefactoredNet.decode`,
which recompiles on every new input resolution. Now opt-in via
`compile_decoder=True`.

## Added

**`VSRStream`.** The research code only ever processed whole clips from disk.
Feeding `VSRPipeline` one batch at a time restarts the keyframe stride at each
call boundary (`SiS` + `SiS` = `SiSSiS`), which at K=2 with 2-frame batches
super-resolves every frame and makes K a no-op. `VSRStream` keeps the stride
global by carrying the previous keyframe across calls, and is tested to produce
byte-identical output to the whole-clip path for every batch size.

## Dropped

- `motion_score` / Farneback motion gating (research-only; removes the OpenCV dependency)
- kfi/natten sweeps, dataset loaders, metrics, comparison scripts
- `interpolation/model/loss.py` and `pytorch_msssim/` — training-only, and the
  only reason the interpolation stage needed torchvision
- `interpolation/train_log/refine.py` — dead. Its `from model.warplayer import warp`
  is not a valid import from that location; the module was never actually loaded.
- `InterpolationPipeline`'s `resolution` argument and square-resolution assert —
  it set four attributes that were never read
- `return_aux` on RIFE inference, and the `output_format="pil"` branch that raised

## Preserved deliberately

The UNet surgery in `super_resolution/pipeline.py` (75% channel pruning,
stripping time embeddings and cross-attention, the custom block forwards, the
`body.N` checkpoint key remapping) is carried over as-is. It has to match the
surgery used at training time or the checkpoint will not load onto it.

## Verification status

Verified in the new repo (`uv run pytest`, 1160 passing):

- **Scheduling** — every (N, K) pair up to N=40 covers each output frame exactly
  once, preserves frame count, and puts each timestep at its true source position.
- **Routing** — the pipeline drives stub models over a clip whose frame *i* is a
  solid value *i*; every frame comes back at its own index for K=1,2,3.
- **numpy boundary and padding** — uint8 roundtrip, layout detection, rejection
  of 0-255 floats / ragged clips / non-RGB, pad-unpad identity, reflect fallback.
- **RIFE stage against the real checkpoint** — the stripped `IFNet` loads
  `flownet.pkl` with **zero missing keys**; the only unexpected keys are
  `teacher.*` and `caltime.*`, the training-only blocks. A forward pass at
  unaligned resolutions roundtrips, and a square moving 10px→50px interpolates
  monotonically (34.6 / 44.5 / 54.0 at t=0.25/0.5/0.75).
- **Dependencies** — every `diffusers` import path resolves against 0.33.0, and
  the weight paths in `weights.py` match both Hub repos, which are public and ungated.

Not verified:

- **The SR stage has never been executed.** No CUDA on the machine this was
  assembled on, and running it needs the ~3.5 GB `sd-research/stable-diffusion-2-1-base`
  download. In particular the checkpoint-key match for `RefactoredNet` is
  unconfirmed — the equivalent check that caught nothing on the RIFE side has
  not been run here. `_load_model_weights` uses `strict=False`, so a broken key
  remap would load silently and produce garbage rather than raising.
  **Run the same key audit before trusting output:**

  ```python
  result = pipeline.model.load_state_dict(remapped, strict=False)
  print(len(result.missing_keys), len(result.unexpected_keys))
  ```

- **Numerical parity with the research repo.** At K=1 output should match the
  old `main.py` exactly; at K>1 it should match for any clip where
  `(N - 1) % K == 0`. Where it differs at K>1 with a short tail, the new result
  is the correct one — that is the tail bug documented above.
