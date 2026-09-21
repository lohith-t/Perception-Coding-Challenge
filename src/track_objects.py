"""Part B step 2: run the detectors over every frame and lift them to WORLD coords.

Part A gives us the full ego pose (position AND heading) at every frame, so a
detection made in the car frame can be pushed into the world frame with

        p_world = C_t + R(theta_t) p_car

That is worth doing for its own sake -- a BEV in the ground frame is richer
than one in the car frame -- but it also buys a free accuracy test:

    The barrels are bolted to the ground. If every barrel detection from all
    299 frames lands in the same place in world coordinates, the ego pose is
    right. If the pose were wrong, they would smear.

That is an independent check on Part A using data Part A never touched.
"""
import csv, os
import numpy as np

from detect_objects import (car_frame, orange_mask, cluster3d, light_colour,
                            GROUND_Z, XLIM, YLIM, ZLIM)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
DS = os.path.join(ROOT, "dataset")
N = 299
MAX_BARREL_PTS = 260        # per frame, subsampled for a manageable file


def smooth_colours(seq, win=6):
    from collections import Counter
    out = []
    for i in range(len(seq)):
        w = [c for c in seq[max(0, i - win):i + win + 1] if c != "unknown"]
        out.append(Counter(w).most_common(1)[0][0] if w else "unknown")
    return out


def reject_jumps(cart_c, cart_w, pos, theta, win=9, tol=1.5):
    """Drop frames whose cart position jumps away from the local median."""
    c = cart_c.copy()
    good = np.isfinite(c[:, 0])
    med = np.full_like(c, np.nan)
    for i in range(len(c)):
        w = c[max(0, i - win):i + win + 1]
        w = w[np.isfinite(w[:, 0])]
        if len(w):
            med[i] = np.median(w, axis=0)
    bad = good & (np.linalg.norm(c - med, axis=1) > tol)
    c[bad] = np.nan
    print(f"  cart: rejected {bad.sum()} outlier frames")
    # re-derive world positions from the cleaned car-frame track
    w = np.full_like(c, np.nan)
    ok = np.isfinite(c[:, 0])
    for i in np.where(ok)[0]:
        ca, sa = np.cos(theta[i]), np.sin(theta[i])
        w[i] = pos[i] + np.array([[ca, -sa], [sa, ca]]) @ c[i]
    return c, w


def persistent_only(pw, pf, radius=1.2, min_hits=12):
    """Keep only pedestrian detections that recur near the same world point."""
    if not len(pw):
        return pw, pf
    keep = np.zeros(len(pw), bool)
    for i in range(len(pw)):
        keep[i] = (np.linalg.norm(pw - pw[i], axis=1) < radius).sum() >= min_hits
    print(f"  pedestrians: kept {keep.sum()}/{len(pw)} persistent detections")
    return pw[keep], pf[keep]


def classify(clusters):
    """Split geometric clusters into (cart candidates, pedestrian candidates)."""
    carts, peds = [], []
    for c in clusters:
        ctr = np.median(c, axis=0)
        lo, hi = np.percentile(c, 2, axis=0), np.percentile(c, 98, axis=0)
        ext = hi - lo
        w = max(ext[1], 0.01)          # lateral width
        h = ext[2]                     # height
        # Stereo depth uncertainty smears objects ALONG the viewing ray, so the
        # X (range) extent is unreliable for sizing. Judge on width and height.
        if len(c) < 60:
            continue
        if 0.9 < w < 3.2 and 0.7 < h < 2.4 and ctr[0] < 32 and abs(ctr[1]) < 8:
            carts.append((ctr, len(c), ext))
        if w < 1.3 and 0.9 < h < 2.3 and ctr[0] < 30:
            peds.append((ctr, len(c), ext))
    return carts, peds


def main():
    traj = np.load(os.path.join(OUT, "trajectory.npz"))
    pos, theta = traj["pos"], traj["theta"]
    rgb_c = np.load(os.path.join(OUT, "rgb_ds.npy"), mmap_mode="r")
    xyz_c = np.load(os.path.join(OUT, "xyz_ds.npy"), mmap_mode="r")
    boxes = {int(r["frame"]): (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
             for r in csv.DictReader(open(os.path.join(DS, "bbox_light.csv")))}

    def to_world(p, i):
        """p: (...,2 or 3) in the car frame -> world XY."""
        c, s = np.cos(theta[i]), np.sin(theta[i])
        R = np.array([[c, -s], [s, c]])
        return pos[i] + p[..., :2] @ R.T

    barrels_w, barrel_frame = [], []
    cart_c, cart_w = np.full((N, 2), np.nan), np.full((N, 2), np.nan)
    peds_w, peds_frame = [], []
    colours = []
    prev_cart = None

    for i in range(N):
        im = np.asarray(rgb_c[i])
        p = car_frame(np.asarray(xyz_c[i], np.float32))
        valid = np.isfinite(p).all(2)
        Z, Xf, Yf = p[..., 2], p[..., 0], p[..., 1]
        inbox = valid & (Xf > XLIM[0]) & (Xf < XLIM[1]) & (np.abs(Yf) < YLIM[1])

        om = orange_mask(im)
        # Height gate: barrels stand on the ground. Without this the amber
        # traffic-light housings (also orange, but 3-4 m up) get picked up.
        # Range gate: stereo depth error grows roughly with distance squared,
        # so a barrel seen at 35 m smears over several metres. Only accumulate
        # observations made from close enough to be trustworthy -- we drive
        # past all of them anyway, so nothing is lost.
        # Height gate, both ends. The upper bound rejects the amber traffic-
        # light housings (3-4 m up). The LOWER bound matters just as much: the
        # road's double-yellow centre line passes an orange colour test, but it
        # lies flat ON the ground, whereas a barrel stands ~1 m tall. Requiring
        # at least 0.3 m of height above the -1.85 m ground plane removes it.
        bm = om & inbox & (Z > -1.55) & (Z < 0.0) & (Xf < 22.0)
        bp = p[bm]
        if len(bp):
            if len(bp) > MAX_BARREL_PTS:
                bp = bp[np.random.default_rng(i).choice(len(bp), MAX_BARREL_PTS, replace=False)]
            barrels_w.append(to_world(bp, i))
            barrel_frame.append(np.full(len(bp), i))

        obstacle = inbox & (Z > GROUND_Z) & (Z < ZLIM[1]) & ~om
        carts, peds = classify(cluster3d(p[obstacle]))

        if carts:
            # Track continuity: prefer the candidate nearest the last known
            # cart, otherwise fall back to the closest one ahead of us.
            if prev_cart is not None:
                ctr = min(carts, key=lambda c: np.linalg.norm(c[0][:2] - prev_cart[:2]))
                if np.linalg.norm(ctr[0][:2] - prev_cart[:2]) > 4.0:
                    ctr = min(carts, key=lambda c: c[0][0])
            else:
                ctr = min(carts, key=lambda c: c[0][0])
            prev_cart = ctr[0]
            cart_c[i] = ctr[0][:2]
            cart_w[i] = to_world(ctr[0], i)

        for ctr, n, ext in peds:
            peds_w.append(to_world(ctr, i))
            peds_frame.append(i)

        colours.append(light_colour(im, tuple(v // 4 for v in boxes.get(i, (0, 0, 0, 0)))))
        if i % 50 == 0:
            print(f"frame {i:3d}  barrels={bm.sum():5d}px  cart={cart_c[i]}  peds={len(peds)}  light={colours[-1]}", flush=True)

    # ---- reject the handful of frames where the cart tracker latched onto
    # something else. The cart's real motion is smooth (it is a vehicle we are
    # following), so a jump away from the local median is a mis-association.
    cart_c, cart_w = reject_jumps(cart_c, cart_w, pos, theta)

    # ---- require pedestrians to PERSIST. A real person is re-detected near
    # the same world position across many frames; a fence post or a barrel
    # edge that momentarily passes the size gate is not.
    peds_w, peds_frame = persistent_only(np.array(peds_w) if peds_w else np.zeros((0, 2)),
                                         np.array(peds_frame, int))

    # Temporal mode filter: per-frame colour is occasionally "unknown" when the
    # lamp is small, dim, or mid-transition. A traffic light changes state on a
    # timescale of seconds, so a short majority vote removes the flicker without
    # hiding the genuine red -> green switch.
    colours = smooth_colours(colours, win=6)

    np.savez(os.path.join(OUT, "objects.npz"),
             barrels_w=np.concatenate(barrels_w) if barrels_w else np.zeros((0, 2)),
             barrel_frame=np.concatenate(barrel_frame) if barrel_frame else np.zeros(0, int),
             cart_car=cart_c, cart_world=cart_w,
             peds_w=peds_w, peds_frame=peds_frame,
             colours=np.array(colours))

    from collections import Counter
    print(f"\nbarrel points (world) : {sum(len(b) for b in barrels_w)}")
    print(f"cart tracked          : {np.isfinite(cart_c[:,0]).sum()}/{N} frames")
    print(f"pedestrian detections : {len(peds_w)}")
    print(f"light colours         : {Counter(colours).most_common()}")


if __name__ == "__main__":
    main()
