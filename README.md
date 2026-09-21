# Ego-Trajectory & BEV Mapping — Wisconsin Autonomous Perception Challenge
**Lohith Tadiboyana**

## Results
The ego vehicle travels **32.5 m in 10.0 s** (mean 3.25 m/s, peaking ~6.5 m/s, braking to ~1 m/s
at the intersection) while turning **left through 43.9°**. Start **(38.90, 0.00) m** → end
**(7.85, 2.17) m**; range to the light closes 38.90 → 8.14 m.

**Part A:** `trajectory.png`, `trajectory.mp4` · **Part B:** `bev_scene.png`, `bev_scene.mp4` ·
**validation:** `out/method_comparison.png`, `out/bev_validation.png`, `out/diagnostics.png` ·
**numeric track:** `out/trajectory.csv` (and `out/light_track.csv`, the per-frame 3D landmark
measurement the solver consumes). Run order: `extract_light` → `visual_odometry` →
`solve_trajectory` → `plot_trajectory`, `animate_trajectory`, `compare_methods` →
`track_objects` → `bev_scene`.

## Setup
`pip install numpy opencv-python matplotlib scipy` (plus `ffmpeg` on PATH for the `.mp4`s).
Download the dataset via the link in `CHALLENGE.md` and unzip so that `dataset/rgb/`,
`dataset/xyz/` and `dataset/bbox_light.csv` sit at the repo root (4 GB, gitignored).

## Method
Each frame yields one measurement: the 3D vector from the car to the light, `q_t` (bbox → robust
median of the valid XYZ inside it; 295/299 frames). The light is static at the world origin, so

> `C_t = −R(θ_t) q_t`  →  in polar form about the light, `C_t = r_t(cos φ_t, sin φ_t)`

`r_t` is measured every frame and **never integrated, so the trajectory cannot drift in scale.**
All uncertainty collapses into the single angle `φ_t`.

**The catch:** a pose has 3 DOF but one landmark gives only 2 numbers per frame, so `φ` is
*unobservable* — orbiting the light while counter-rotating the heading is indistinguishable. I
close the gap with the **non-holonomic constraint**: a car rolls along its heading and cannot
slide sideways. Per frame it advances `s` and turns `Δθ`, making the system exactly determined:

```
lengths:  r²(t+1) = (X − s)² + y²                  →  s   (recovers speed from range alone)
angles :  Δφ = atan2(y, X − s) − atan2(y, X)       →  angular progress around the light
```

The derivation, and the rewrite that removes a catastrophic-cancellation instability, are
documented in `src/solve_trajectory.py`.

## Assumptions
Non-holonomic vehicle (validated below, not merely assumed); planar motion; 29.9 fps; `φ₀ = 0`,
placing the car on **+X** at t=0 per the spec. Frame-to-frame range noise (σ ≈ 0.48 m) is **5×
the per-frame motion** (0.10 m), so Savitzky–Golay smoothing before differentiation is mandatory.

## ⚠ `+Y` is LEFT, not right
The spec states *"+X forward, +Y right, +Z upward (right-handed)"* — but X-fwd/Y-right/Z-up is a
**left**-handed triad, so the statement contradicts itself. Verified two ways
(`src/verify_axes.py`): ground pixels at the image's left edge give `Y=+3.21`, the right edge
`Y=−3.27`; and the light's `Y` sign tracks its pixel column. **Taking the spec literally inverts
the turn direction.** Also: arrays are `(1200,1920,4)` under key `"xyz"`; invalid pixels are
`NaN` **and `inf`**; frames 3, 4, 5, 23 have no bbox.

## Validation — three independent checks
1. **Visual odometry.** To *test* the non-holonomic assumption rather than trust it, I measured
   the rotation instead: ORB matches → Kabsch inside RANSAC on their 3D points. RANSAC is
   load-bearing — the cart and pedestrians move and are outvoted by the static background.
   296/296 pairs, median 274 inliers. **Heading +43.94° vs +43.87°**; mean position disagreement
   **0.22 m over a 32.5 m path (0.68%)**.
2. **Landmark height.** The light's `Z` is **3.64 ± 0.15 m** across all 299 frames — and `Z` is
   discarded before solving, so its constancy is free evidence.
3. **Static objects stay static.** Part B detections are lifted to world coordinates using the
   Part A pose. Barrels seen in the first and last 5 s coincide; RMS wander **0.50 m** — an upper
   bound, since driving past a barrel shifts its visible centroid.

## Part B
**Barrels — colour:** HSV, height-gated at both ends (above rejects the amber light housings;
below rejects the double-yellow road line, which passes an orange test but lies flat).
**Cart and pedestrians — geometry:** ground removal, then 3D voxel clustering, which separates
objects that overlap in the image but differ in depth. Cart tracked **284/299** frames;
pedestrians must persist across ≥12 frames. **Light state:** the lit blob's hue *plus* lamp
position — naive hue fails twice, since the tan housing outnumbers the lamp pixels and the green
LED saturates to cyan. Result: **red frames 0–27, green 28–298**.

## Limitations
`φ` is the only integrated quantity, so slow angular drift is the failure mode (bounded at ~1.3°
here); a second landmark would make the pose fully observable. The non-holonomic assumption would
degrade on ice or under hard cornering, and the smoothing window would attenuate a sharp manoeuvre.
