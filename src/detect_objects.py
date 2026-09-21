"""Part B step 1: find the other actors in the scene and place them in 3D.

Two complementary detectors, because the objects differ in what makes them
distinctive:

  BARRELS      - vivid orange/white, so COLOUR is the strongest cue.
                 HSV threshold, then gated by height so the amber traffic-light
                 housings (also orange, but 3-4 m up) cannot masquerade as them.

  CART / PEOPLE- no reliable colour signature, so use GEOMETRY. Remove the
                 ground plane, drop what is left into a 3D voxel grid, and take
                 connected components. This is classic LiDAR-style Euclidean
                 clustering; it separates objects that overlap in the image but
                 sit at different depths, which image-space blob finding cannot.

Everything is also written out in WORLD coordinates using the Part A ego pose,
which lets us check that static objects actually stay static.
"""
import os
import numpy as np
import cv2
from scipy import ndimage

GROUND_Z = -1.45      # ground sits near -1.85 m; anything above this is an obstacle
VOX = 0.30            # voxel edge, metres
XLIM, YLIM, ZLIM = (2.0, 38.0), (-16.0, 16.0), (GROUND_Z, 1.6)


def car_frame(p):
    """Dataset axes -> our convention (x fwd, y LEFT, z up).

    Despite the README's "+Y -> right axis", the dataset's Y is LEFT-positive
    (X-fwd/Y-right/Z-up would be left-handed; see verify_axes.py). So this is
    the identity, kept as a named function to make the convention explicit.
    """
    return p


def orange_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (H > 3) & (H < 22) & (S > 110) & (V > 90)


def cluster3d(pts):
    """Euclidean clustering via a 3D occupancy grid + connected components.
    Returns a list of (points, size) per cluster, biggest first."""
    if len(pts) < 8:
        return []
    lo = np.array([XLIM[0], YLIM[0], ZLIM[0]])
    idx = np.floor((pts - lo) / VOX).astype(int)
    shape = np.ceil([(XLIM[1] - XLIM[0]) / VOX, (YLIM[1] - YLIM[0]) / VOX,
                     (ZLIM[1] - ZLIM[0]) / VOX]).astype(int) + 1
    keep = ((idx >= 0) & (idx < shape)).all(1)
    idx, pts = idx[keep], pts[keep]
    if len(pts) < 8:
        return []
    grid = np.zeros(shape, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))   # 26-connectivity
    pl = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    out = []
    for c in range(1, n + 1):
        m = pl == c
        if m.sum() >= 8:
            out.append(pts[m])
    out.sort(key=len, reverse=True)
    return out


def light_colour(rgb, box):
    """Classify the lit lamp. Two independent cues, which agree on this clip.

    Naive hue-over-the-whole-box fails for two reasons visible in the data:
      * the housing is tan/amber and has far more pixels than the lamp, so any
        median over the box reports "yellow";
      * the green LED is bright enough to SATURATE the sensor and comes back
        CYAN (hue ~90 on OpenCV's 0-179 scale), not green (~60).

    So isolate the lit blob first -- a lamp is both saturated and bright, while
    the housing is neither and the unlit lamps are dark -- then read its hue.
    Measured on this clip: red lamps give hue ~8 and sit at ~19% of the box
    height (top lamp); green lamps give hue ~90 at ~66-82% (bottom lamp). The
    vertical position is kept as a cross-check, since a vertical traffic light
    always stacks red / yellow / green from top to bottom.
    """
    x1, y1, x2, y2 = box
    if x2 - x1 < 2 or y2 - y1 < 6:
        return "unknown"
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    score = S * V * V                      # squaring V sharpens lamp vs housing
    lit = score > max(0.18, 0.55 * score.max())
    if lit.sum() < 6:
        return "unknown"
    h = float(np.median(H[lit]))
    vfrac = float(np.where(lit)[0].mean()) / lit.shape[0]   # 0 = top of box

    if h <= 20 or h >= 160:
        return "red"
    if h <= 105:
        # Covers true green (~60) and saturation-shifted cyan (~90).
        return "green"
    if h < 40:
        # Genuinely amber only if it is also the MIDDLE lamp; otherwise this is
        # housing bleed-through rather than a lit lamp.
        return "yellow" if 0.3 < vfrac < 0.7 else "unknown"
    return "unknown"
