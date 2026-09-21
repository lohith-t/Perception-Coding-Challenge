"""Step 6: validate Method A against Method B.

Method A  non-holonomic constraint  -> heading ASSUMED from vehicle kinematics
Method B  visual odometry (Kabsch/RANSAC on the 3D scene) -> heading MEASURED

Both then place the car with the SAME landmark equation, C_t = -R(theta_t) q_t,
so any difference between the two trajectories comes purely from the heading.
If they agree, the non-holonomic assumption is validated on this clip.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
FPS = 29.9

d = np.load(os.path.join(OUT, "trajectory.npz"))
v = np.load(os.path.join(OUT, "vo.npz"))
X, y, theta_a, pos_a = d["X"], d["y"], d["theta"], d["pos"]
t = np.arange(len(X)) / FPS

# --- Method B heading -------------------------------------------------------
# dyaw[i] is the rotation over the STRIDE frames ending at i, and the windows
# overlap one per frame, so dyaw/STRIDE is the per-frame yaw rate.
stride = int(v["stride"])
rate = np.nan_to_num(v["dyaw"] / stride)
# Light smoothing: per-frame VO yaw is quantised by feature noise.
k = 15
rate = np.convolve(rate, np.ones(k) / k, mode="same")
# Anchor to the same t=0 heading; the world frame's absolute orientation is a
# definition (the spec's "+X axis at t=0"), not something either method measures.
theta_b = theta_a[0] + np.cumsum(rate)


def place(theta):
    """C_t = -R(theta_t) q_t  -- identical landmark equation for both methods."""
    c, s = np.cos(theta), np.sin(theta)
    return -np.stack([X * c - y * s, X * s + y * c], axis=1)


pos_b = place(theta_b)
err = np.linalg.norm(pos_a - pos_b, axis=1)
dh = np.degrees(theta_b - theta_a)

fig, axs = plt.subplots(1, 3, figsize=(17, 6.2))

a = axs[0]
a.plot(pos_a[:, 0], pos_a[:, 1], lw=3, color="#1f77b4", label="A: non-holonomic")
a.plot(pos_b[:, 0], pos_b[:, 1], lw=2, ls="--", color="#d62728", label="B: visual odometry")
a.plot(0, 0, "*", ms=22, color="#111", label="traffic light")
a.plot(*pos_a[0], "X", ms=11, color="#333")
a.set_aspect("equal"); a.grid(alpha=.3, ls="--")
_lo = np.minimum(pos_a.min(0), 0) - 3; _hi = np.maximum(pos_a.max(0), 0) + 3
_cy = (_lo[1] + _hi[1]) / 2; _h = max((_hi[0] - _lo[0]) / 2 / 1.1, (_hi[1] - _lo[1]) / 2)
a.set_xlim(_lo[0], _hi[0]); a.set_ylim(_cy - _h, _cy + _h)
a.set_xlabel("World X (m)"); a.set_ylabel("World Y (m)")
a.set_title("Two independent trajectories", weight="bold")
a.legend(fontsize=8, loc="lower right")

a = axs[1]
a.plot(t, np.degrees(theta_a - theta_a[0]), lw=3, color="#1f77b4", label="A: assumed (kinematics)")
a.plot(t, np.degrees(theta_b - theta_b[0]), lw=2, ls="--", color="#d62728", label="B: measured (scene)")
a.grid(alpha=.3); a.set_xlabel("time (s)"); a.set_ylabel("heading change (deg)")
a.set_title(f"Heading: A ends {np.degrees(theta_a[-1]-theta_a[0]):+.1f}°, "
            f"B ends {np.degrees(theta_b[-1]-theta_b[0]):+.1f}°", weight="bold")
a.legend(fontsize=9)

a = axs[2]
a.plot(t, err, lw=2, color="#2ca02c", label="position disagreement")
a.plot(t, np.abs(dh - dh[0]) * 0 + np.abs(dh), lw=1.5, ls=":", color="#9467bd", label="|heading disagreement| (deg)")
a.grid(alpha=.3); a.set_xlabel("time (s)"); a.set_ylabel("m   /   deg")
a.set_title(f"Disagreement: mean {err.mean():.2f} m, max {err.max():.2f} m\n"
            f"({100*err.mean()/np.linalg.norm(np.diff(pos_a,axis=0),axis=1).sum():.1f}% of a "
            f"{np.linalg.norm(np.diff(pos_a,axis=0),axis=1).sum():.0f} m path)", weight="bold")
a.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "method_comparison.png"), dpi=150)
np.savez(os.path.join(OUT, "vo_trajectory.npz"), pos=pos_b, theta=theta_b)

path_len = np.linalg.norm(np.diff(pos_a, axis=0), axis=1).sum()
print(f"heading, method A : {np.degrees(theta_a[-1]-theta_a[0]):+.2f} deg")
print(f"heading, method B : {np.degrees(theta_b[-1]-theta_b[0]):+.2f} deg")
print(f"max heading disagreement : {np.abs(dh).max():.2f} deg")
print(f"position disagreement    : mean {err.mean():.3f} m, max {err.max():.3f} m")
print(f"path length {path_len:.1f} m -> mean disagreement is {100*err.mean()/path_len:.2f}% of path")
print("wrote out/method_comparison.png")
