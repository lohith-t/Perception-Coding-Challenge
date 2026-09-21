"""Part B step 3: render the enriched BEV, in the GROUND frame.

Outputs
  bev_scene.png      static world-frame BEV: ego path, barrels, golf cart, people
  bev_validation.png the static-object consistency test (see below)
  bev_scene.mp4      animated camera view + BEV, with live traffic-light state

The validation figure is the point of doing this in the world frame. Barrels do
not move, so every observation of a barrel from any of the 299 frames must land
on the same world coordinate. Plotting early observations against late ones
shows immediately whether the Part A ego pose is right: they overlap if it is,
and separate into two offset copies if it is not.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
FPS = 29.9
LAMP = {"red": "#e8112d", "yellow": "#f5a623", "green": "#21b563", "unknown": "#999999"}

traj = np.load(os.path.join(OUT, "trajectory.npz"))
obj = np.load(os.path.join(OUT, "objects.npz"))
pos, theta = traj["pos"], traj["theta"]
bw, bf = obj["barrels_w"], obj["barrel_frame"]
cart_w, peds_w, peds_f = obj["cart_world"], obj["peds_w"], obj["peds_frame"]
colours = obj["colours"]
t = np.arange(len(pos)) / FPS
N = len(pos)


def frame_axes(ax):
    ax.set_aspect("equal")
    ax.grid(alpha=.25, ls="--", lw=.6)
    ax.set_xlabel("World X  (m)")
    ax.set_ylabel("World Y  (m)")


def limits():
    pts = [pos, bw, cart_w[np.isfinite(cart_w[:, 0])], np.zeros((1, 2))]
    allp = np.concatenate([p for p in pts if len(p)])
    lo, hi = allp.min(0) - 3, allp.max(0) + 3
    cy, half = (lo[1] + hi[1]) / 2, max((hi[0] - lo[0]) / 2 / 1.7, (hi[1] - lo[1]) / 2)
    return (lo[0], hi[0]), (cy - half, cy + half)


XL, YL = limits()
cok = np.isfinite(cart_w[:, 0])

# ------------------------------------------------------------ static BEV ----
fig, ax = plt.subplots(figsize=(13, 7.6))
frame_axes(ax)
ax.scatter(bw[:, 0], bw[:, 1], s=3, c="#ff7f0e", alpha=.16, lw=0, label="barrels (all frames)")
if peds_w.size:
    ax.scatter(peds_w[:, 0], peds_w[:, 1], s=14, c="#8c564b", alpha=.35, lw=0, label="pedestrians")

seg = np.stack([cart_w[cok][:-1], cart_w[cok][1:]], axis=1)
lc = LineCollection(seg, cmap="autumn", norm=plt.Normalize(0, t[-1]), lw=2.5, zorder=4)
lc.set_array(t[cok][:-1]); ax.add_collection(lc)
ax.plot([], [], color="#ff2200", lw=2.5, label="golf cart (dynamic)")

seg = np.stack([pos[:-1], pos[1:]], axis=1)
le = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, t[-1]), lw=4, zorder=5)
le.set_array(t[:-1]); ax.add_collection(le)
plt.colorbar(le, ax=ax, label="time (s)", shrink=.85, pad=.015)
ax.plot([], [], color="#1f77b4", lw=4, label="ego trajectory")

ax.plot(*pos[0], "X", ms=13, color="#d62728", mec="k", zorder=7, label="ego start")
ax.plot(0, 0, "*", ms=26, color=LAMP[str(colours[-1])], mec="k", mew=1.2, zorder=8,
        label=f"traffic light (ends {colours[-1]})")
ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_title("Bird's-Eye View in the Ground Frame\n"
             "origin = under the traffic light   |   all objects placed with the Part A ego pose",
             fontsize=13, weight="bold")
ax.legend(loc="lower left", fontsize=9, framealpha=.95)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "bev_scene.png"), dpi=150)
print("wrote bev_scene.png")

# ------------------------------------------------- static-object validation --
early, late = bf < N // 2, bf >= N // 2
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
a = axs[0]
frame_axes(a)
a.scatter(bw[early, 0], bw[early, 1], s=6, c="#1f77b4", alpha=.35, lw=0, label="seen in first 5 s")
a.scatter(bw[late, 0], bw[late, 1], s=6, c="#d62728", alpha=.35, lw=0, label="seen in last 5 s")
a.plot(pos[:, 0], pos[:, 1], color="#444", lw=1.5, label="ego path")
a.plot(0, 0, "*", ms=20, color="k")
a.set_xlim(*XL); a.set_ylim(*YL)
a.set_title("Do static objects stay static?\nbarrels seen early vs late, in world coords", weight="bold")
a.legend(fontsize=9, loc="lower left")

# Quantify it honestly. A nearest-neighbour distance between two DENSE clouds
# is small almost regardless of alignment -- it measures point density, not
# accuracy. Instead, isolate individual barrels and ask how much each one's
# estimated WORLD position wanders over the frames it is visible in. A static
# object's estimate should not move at all.
#
# Caveat stated up front: as we drive past a barrel we see different faces of
# it, so the centroid of its visible surface genuinely shifts by up to ~a
# barrel radius. This number therefore bounds the ego-pose error from ABOVE.
from scipy import ndimage as _nd

VOX2 = 0.5
_lo = bw.min(0)
_idx = np.floor((bw - _lo) / VOX2).astype(int)
_grid = np.zeros(_idx.max(0) + 1, bool)
_grid[_idx[:, 0], _idx[:, 1]] = True
_lab, _n = _nd.label(_grid, structure=np.ones((3, 3)))
_pl = _lab[_idx[:, 0], _idx[:, 1]]

drifts, spans = [], []
for c in range(1, _n + 1):
    m = _pl == c
    if m.sum() < 200:
        continue
    fr, pts = bf[m], bw[m]
    uf = np.unique(fr)
    if len(uf) < 60:                      # needs a long observation history
        continue
    cen = np.array([pts[fr == f].mean(0) for f in uf])
    drifts.append(np.sqrt(((cen - cen.mean(0)) ** 2).sum(1).mean()))   # RMS wander
    spans.append(len(uf))

a2 = axs[1]
if drifts:
    a2.bar(range(len(drifts)), drifts, color="#2ca02c", alpha=.85)
    a2.axhline(np.median(drifts), color="k", ls="--",
               label=f"median {np.median(drifts):.2f} m")
    a2.set_xlabel(f"barrel group (n={len(drifts)}, each seen in {min(spans)}-{max(spans)} frames)")
    a2.set_ylabel("RMS wander of its world position (m)")
    a2.set_title(f"Do static objects stay static?\nmedian {np.median(drifts):.2f} m, "
                 f"worst {max(drifts):.2f} m  (upper bound on pose error)", weight="bold")
    a2.legend(); a2.grid(alpha=.3, axis="y")
    print(f"per-barrel world-position RMS wander: median {np.median(drifts):.3f} m, "
          f"worst {max(drifts):.3f} m, over {len(drifts)} barrel groups")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "bev_validation.png"), dpi=150)
print("wrote out/bev_validation.png")

# --------------------------------------------------------------- animation --
rgb = np.load(os.path.join(OUT, "rgb_ds.npy"), mmap_mode="r")
fig = plt.figure(figsize=(16, 6.4))
axc = fig.add_axes([0.005, 0.06, 0.44, 0.88]); axc.axis("off")
axb = fig.add_axes([0.50, 0.09, 0.46, 0.85])
frame_axes(axb)
axb.set_xlim(*XL); axb.set_ylim(*YL)
axb.set_title("Bird's-Eye View (ground frame)", fontsize=12, weight="bold")

im = axc.imshow(np.asarray(rgb[0]))
ctitle = axc.set_title("", fontsize=12, weight="bold")

axb.plot(pos[:, 0], pos[:, 1], color="#ddd", lw=1.2, zorder=1)
bar = axb.scatter([], [], s=5, c="#ff7f0e", alpha=.3, lw=0, label="barrels")
ped = axb.scatter([], [], s=40, c="#8c564b", marker="^", lw=0, label="pedestrians")
ego, = axb.plot([], [], lw=3.5, color="#1f77b4", zorder=5, label="ego path")
cart, = axb.plot([], [], lw=2.2, color="#ff2200", zorder=4, label="golf cart")
cartm, = axb.plot([], [], "s", ms=10, color="#ff2200", mec="k", zorder=6)
lamp, = axb.plot([0], [0], "*", ms=26, color="#999", mec="k", mew=1.2, zorder=8, label="traffic light")
carbox = Rectangle((0, 0), 4.2, 1.9, angle=0, fc="#1f77b4", ec="k", lw=1, zorder=7)
axb.add_patch(carbox)
axb.legend(loc="lower left", fontsize=9, framealpha=.95)
hud = axb.text(.985, .97, "", transform=axb.transAxes, ha="right", va="top",
               family="monospace", fontsize=10,
               bbox=dict(fc="white", ec="#bbb", alpha=.92, boxstyle="round,pad=0.4"))


def update(i):
    im.set_data(np.asarray(rgb[i]))
    c = str(colours[i])
    ctitle.set_text(f"camera  ·  t = {t[i]:4.2f} s  ·  light: {c.upper()}")
    ctitle.set_color(LAMP[c])
    m = bf <= i
    bar.set_offsets(bw[m] if m.any() else np.zeros((0, 2)))
    mp = peds_f <= i
    ped.set_offsets(peds_w[mp] if mp.any() else np.zeros((0, 2)))
    ego.set_data(pos[:i + 1, 0], pos[:i + 1, 1])
    k = cok[:i + 1]
    cart.set_data(cart_w[:i + 1][k, 0], cart_w[:i + 1][k, 1])
    if cok[i]:
        cartm.set_data([cart_w[i, 0]], [cart_w[i, 1]])
    lamp.set_color(LAMP[c])
    # draw the ego vehicle as an oriented box rather than a dot
    ang = np.degrees(theta[i])
    ca, sa = np.cos(theta[i]), np.sin(theta[i])
    carbox.set_xy((pos[i, 0] - 2.1 * ca + 0.95 * sa, pos[i, 1] - 2.1 * sa - 0.95 * ca))
    carbox.set_angle(ang)
    hud.set_text(f"t       {t[i]:5.2f} s\n"
                 f"ego   ({pos[i,0]:6.2f},{pos[i,1]:6.2f})\n"
                 f"head  {np.degrees(theta[i]-theta[0]):+6.1f}°\n"
                 f"light {c}")
    return ()


anim = FuncAnimation(fig, update, frames=N, interval=1000 / FPS, blit=False)
anim.save(os.path.join(ROOT, "bev_scene.mp4"),
          writer=FFMpegWriter(fps=FPS, bitrate=4500, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
print("wrote bev_scene.mp4")
