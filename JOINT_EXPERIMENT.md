# Shared-27 joint Gaussian experiment

## Hypothesis

R8 trains the 19 shape channels from point-space reconstruction and the 8
appearance channels from rendering. With detached render positions, the render
gradient on shape is exactly zero. A Gaussian is joint, so xyz and attributes
must be decoded from a shared representation rather than two objective-specific
channel islands.

## J1 design

- Preserve the trained R8 geometry and attribute decoders as base functions.
- Read all 27 non-anchor compact channels (19 shape + 8 appearance).
- Build one shared per-slot representation with a one-cell neighbour window.
- Emit bounded residuals for xyz, log-scale, quaternion, opacity, colour and
  presence from that representation.
- Zero-initialise every residual head: J1 starts at the R8 function instead of
  paying for a new decoder.
- Detach xyz only inside the rasteriser. Geometry losses train xyz; render and
  parameter losses still reach all shared channels through the attribute heads.
- Reuse the geometry Sinkhorn plan for attributes. This removes the old
  many-to-one scale-growing responsibility target.

## Comparisons

`J0` has the same R8 checkpoint, optimiser, matching, loss weights, views and
schedule but disables the joint decoder. Therefore:

- `J1 - J0`: shared joint-decoder contribution.
- `J0 - R8`: changed matching/polish contribution.
- Do not interpret `J1 - R8` as a single causal effect.

## Launch

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/launch_j0_joint_control.sh --detached
CUDA_VISIBLE_DEVICES=1 bash scripts/launch_j1_shared_joint.sh --detached
```

Primary gates are held-out photo/teacher PSNR, intra-group Sinkhorn, relative
offset, projected-radius p99.9, scale/opacity/color errors and predicted-presence
metrics. A non-zero gradient norm is necessary but not evidence of a useful
gradient; a follow-up audit should record geometry-vs-render gradient cosine on
the shared code.
