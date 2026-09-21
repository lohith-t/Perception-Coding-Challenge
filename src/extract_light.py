"""Step 1: pull a robust 3D position for the traffic light out of every frame.

For each frame we know the traffic light's bounding box in the image. The .npz
gives us, for every pixel, the 3D point that pixel sees, in camera coordinates.
So the light's 3D position is "the 3D points inside the box" -- but the box also
contains background seen through/around the light housing, plus invalid pixels.
We therefore take a robust (median-based, depth-gated) estimate rather than the
single centre pixel.
"""
import csv, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DS = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "out", "light_track.csv")

SHRINK = 0.25  # trim this fraction off each side of the bbox before sampling


def robust_xyz(pts):
    """pts: (N,3) valid camera-frame points inside the box. Returns (3,) or None."""
    if len(pts) < 10:
        return None
    # The light is the nearest coherent surface in the box. Gate on depth (X =
    # forward) around the median to reject background bleeding into the box.
    x = pts[:, 0]
    med = np.median(x)
    keep = np.abs(x - med) < max(0.5, 0.05 * med)
    if keep.sum() < 10:
        keep = np.abs(x - med) < max(1.5, 0.10 * med)
    if keep.sum() < 5:
        return None
    return np.median(pts[keep], axis=0)


def main():
    rows = list(csv.DictReader(open(os.path.join(DS, "bbox_light.csv"))))
    out = []
    for r in rows:
        f = int(r["frame"])
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        rec = dict(frame=f, u="", v="", X="", Y="", Z="", n="")
        if x2 > x1 and y2 > y1:
            p = np.load(os.path.join(DS, f"depth{f:06d}.npz".replace("depth", "xyz/depth")))["xyz"][..., :3]
            bw, bh = x2 - x1, y2 - y1
            sx, sy = int(bw * SHRINK), int(bh * SHRINK)
            sub = p[y1 + sy:y2 - sy, x1 + sx:x2 - sx]
            if sub.size == 0:
                sub = p[y1:y2, x1:x2]
            m = np.isfinite(sub).all(axis=2)
            est = robust_xyz(sub[m])
            if est is not None:
                rec.update(u=(x1 + x2) / 2.0, v=(y1 + y2) / 2.0,
                           X=est[0], Y=est[1], Z=est[2], n=int(m.sum()))
        out.append(rec)
        if f % 25 == 0:
            print(f"frame {f}: {rec['X']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "u", "v", "X", "Y", "Z", "n"])
        w.writeheader()
        w.writerows(out)
    good = sum(1 for r in out if r["X"] != "")
    print(f"wrote {OUT}: {good}/{len(out)} frames with a 3D light fix")


if __name__ == "__main__":
    main()
