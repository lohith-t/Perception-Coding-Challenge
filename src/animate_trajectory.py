"""Step 5: render trajectory.mp4 -- the BEV trajectory drawn as a function of time."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
FPS = 29.9

d = np.load(os.path.join(OUT, "trajectory.npz"))
pos, theta, speed, r = d["pos"], d["theta"], d["speed"], d["r"]
t = np.arange(len(pos)) / FPS

fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.set_aspect("equal")
ax.grid(alpha=.3, ls="--", lw=.6)
ax.axhline(0, color="k", lw=.8, alpha=.4)
ax.axvline(0, color="k", lw=.8, alpha=.4)
ax.set_xlabel("World X  (m)")
ax.set_ylabel("World Y  (m)")
ax.set_title("Ego-Vehicle Trajectory in the Ground Frame  (origin = traffic light)",
             fontsize=13, weight="bold")

pad = 4.0
lo, hi = np.minimum(pos.min(0), 0) - pad, np.maximum(pos.max(0), 0) + pad
cy, half = (lo[1] + hi[1]) / 2, max((hi[0] - lo[0]) / 2 / 1.78, (hi[1] - lo[1]) / 2)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(cy - half, cy + half)

ax.plot(pos[:, 0], pos[:, 1], color="#ccc", lw=1.2, zorder=1, label="full path")
ax.plot(0, 0, "*", ms=26, color="#111", zorder=6, label="Traffic light (origin)")
ax.plot(*pos[0], "X", ms=13, color="#d62728", mec="k", zorder=5, label="Start")

trail, = ax.plot([], [], lw=3.5, color="#1f77b4", zorder=3, label="travelled")
car,   = ax.plot([], [], "o", ms=13, color="#1f77b4", mec="k", mew=1.2, zorder=7)
sight, = ax.plot([], [], ls=":", lw=1.2, color="#888", zorder=2)
head = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=8,
                   arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2))
hud = ax.text(.985, .97, "", transform=ax.transAxes, ha="right", va="top",
              family="monospace", fontsize=11,
              bbox=dict(fc="white", ec="#bbb", alpha=.92, boxstyle="round,pad=0.45"))
ax.legend(loc="lower left", fontsize=9, framealpha=.95)


def update(i):
    trail.set_data(pos[:i + 1, 0], pos[:i + 1, 1])
    car.set_data([pos[i, 0]], [pos[i, 1]])
    sight.set_data([0, pos[i, 0]], [0, pos[i, 1]])   # line of sight to the light
    L = 3.0
    head.set_position((pos[i, 0], pos[i, 1]))
    head.xy = (pos[i, 0] + L * np.cos(theta[i]), pos[i, 1] + L * np.sin(theta[i]))
    hud.set_text(f"t        {t[i]:5.2f} s\n"
                 f"pos    ({pos[i,0]:6.2f}, {pos[i,1]:6.2f}) m\n"
                 f"range   {r[i]:5.2f} m\n"
                 f"speed   {speed[i]:5.2f} m/s\n"
                 f"heading {np.degrees(theta[i]-theta[0]):+6.1f}°")
    return trail, car, sight, head, hud


anim = FuncAnimation(fig, update, frames=len(pos), interval=1000 / FPS, blit=False)
path = os.path.join(ROOT, "trajectory.mp4")
anim.save(path, writer=FFMpegWriter(fps=FPS, bitrate=3500,
                                    extra_args=["-pix_fmt", "yuv420p"]), dpi=110)
print("wrote", path)
