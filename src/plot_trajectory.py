"""Step 3: render trajectory.png (required) plus a diagnostics figure."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
FPS = 29.9
d = np.load(os.path.join(OUT, "trajectory.npz"))
pos, theta, r, speed = d["pos"], d["theta"], d["r"], d["speed"]
t = np.arange(len(pos)) / FPS


def bev_axes(ax):
    ax.set_aspect("equal")
    ax.grid(alpha=.3, ls="--", lw=.6)
    ax.axhline(0, color="k", lw=.8, alpha=.5)
    ax.axvline(0, color="k", lw=.8, alpha=.5)
    ax.set_xlabel("World X  (m)")
    ax.set_ylabel("World Y  (m)")


# ---------------------------------------------------------------- trajectory
fig, ax = plt.subplots(figsize=(13.5, 6.4))
bev_axes(ax)

# Colour the path by time so the direction of travel is unambiguous.
seg = np.stack([pos[:-1], pos[1:]], axis=1)
lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, t[-1]), lw=3, zorder=3)
lc.set_array(t[:-1])
ax.add_collection(lc)
plt.colorbar(lc, ax=ax, label="time (s)", shrink=.85, pad=.015, aspect=30)

# Heading arrows: position alone doesn't show where the car was POINTING.
for i in range(0, len(pos), 25):
    ax.arrow(pos[i, 0], pos[i, 1], 2.2*np.cos(theta[i]), 2.2*np.sin(theta[i]),
             head_width=.6, head_length=.7, fc="#d62728", ec="#d62728",
             alpha=.65, lw=.8, zorder=4, length_includes_head=True)

ax.plot(*pos[0],  "X", ms=15, color="#d62728", mec="k", mew=1, zorder=6, label="Start (t=0)")
ax.plot(*pos[-1], "o", ms=13, color="#2ca02c", mec="k", mew=1, zorder=6, label="End (t=10 s)")
ax.plot(0, 0, "*", ms=26, color="#111", zorder=6, label="Traffic light (world origin)")
ax.annotate("traffic light", (0, 0), textcoords="offset points", xytext=(12, -16),
            fontsize=10, weight="bold")
ax.plot([], [], color="#d62728", lw=1.5, label="Vehicle heading")

# Report path/duration (not the mean of the instantaneous estimate) so the
# figure and the README quote the same number.
_L = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
_v = _L / (len(pos) / FPS)   # 299 frames / 29.9 fps = the stated 10.0 s clip
stats = (f"path length   {_L:.1f} m\n"
         f"mean speed    {_v:.2f} m/s  ({_v*2.237:.1f} mph)\n"
         f"heading swing {np.degrees(theta[-1]-theta[0]):+.0f}°  (left turn) \n"
         f"range  {r[0]:.1f} m → {r[-1]:.1f} m")
ax.text(.975, .97, stats, transform=ax.transAxes, ha="right", va="top",
        family="monospace", fontsize=9,
        bbox=dict(fc="white", ec="#bbb", alpha=.9, boxstyle="round,pad=0.5"))

ax.set_title("Ego-Vehicle Trajectory in the Ground Frame\n"
             "origin = under the traffic light   |   car starts on the +X axis",
             fontsize=13, weight="bold")
pad = 4.0
lo = np.minimum(pos.min(0), 0) - pad
hi = np.maximum(pos.max(0), 0) + pad
cy = (lo[1] + hi[1]) / 2
half = max((hi[0] - lo[0]) / 2 / 2.35, (hi[1] - lo[1]) / 2)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(cy - half, cy + half)
ax.legend(loc="lower left", framealpha=.95, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "trajectory.png"), dpi=160)
print("wrote trajectory.png")

# --------------------------------------------------------------- diagnostics
fig, axs = plt.subplots(2, 2, figsize=(13, 8))

a = axs[0, 0]
a.plot(t, d["X_raw"], lw=.8, alpha=.45, color="#1f77b4", label="X forward (raw)")
a.plot(t, d["X"], lw=2, color="#1f77b4", label="X forward (smoothed)")
a.plot(t, d["y_raw"], lw=.8, alpha=.45, color="#ff7f0e", label="y left (raw)")
a.plot(t, d["y"], lw=2, color="#ff7f0e", label="y left (smoothed)")
a.set_title("Traffic light in the car frame\n(noise is 5× the per-frame motion → smoothing is mandatory)",
            fontsize=10)
a.set_xlabel("time (s)"); a.set_ylabel("m"); a.legend(fontsize=8); a.grid(alpha=.3)

a = axs[0, 1]
a.plot(t, d["Z"], lw=1.5, color="#9467bd")
a.axhline(np.nanmean(d["Z"]), ls="--", color="k", lw=.9)
a.set_ylim(0, 6)
a.set_title(f"Sanity check: light height is constant\n{np.nanmean(d['Z']):.2f} ± {np.nanstd(d['Z']):.2f} m "
            "(never used by the solver)", fontsize=10)
a.set_xlabel("time (s)"); a.set_ylabel("Z (m)"); a.grid(alpha=.3)

a = axs[1, 0]
a.plot(t, r, lw=2, color="#2ca02c")
a.set_title("Range to the light\n(measured every frame — cannot drift)", fontsize=10)
a.set_xlabel("time (s)"); a.set_ylabel("m"); a.grid(alpha=.3)

a = axs[1, 1]
a.plot(t, speed, lw=1.2, color="#8c564b", alpha=.6, label="speed (m/s)")
a2 = a.twinx()
a2.plot(t, np.degrees(theta - theta[0]), lw=2, color="#d62728", label="heading (deg)")
a2.set_ylabel("heading change (deg)", color="#d62728")
a.set_title("Recovered speed and heading", fontsize=10)
a.set_xlabel("time (s)"); a.set_ylabel("speed (m/s)", color="#8c564b"); a.grid(alpha=.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "diagnostics.png"), dpi=150)
print("wrote out/diagnostics.png")
