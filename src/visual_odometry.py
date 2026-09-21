"""Step 4: an INDEPENDENT estimate of vehicle yaw, from scene geometry alone.

Method A (solve_trajectory.py) assumes the vehicle is non-holonomic -- it
rolls along its heading and cannot slide sideways. That is a good assumption
but it is still an assumption. Here we measure the rotation instead.

For a pair of frames t and t+STRIDE:
  1. ORB keypoints in both RGB images, matched with a ratio test.
  2. Look up each match's 3D point in the .npz (camera coords, metres).
  3. Fit the rigid transform between the two 3D point sets (Kabsch) inside
     RANSAC.  The scene is mostly static, so the transform that the majority
     of points agree on IS the vehicle's motion.
  4. Read the yaw out of that rotation.

RANSAC is load-bearing: the golf cart and the pedestrians are MOVING, so they
break the rigid-world assumption. They lose the vote to the static background
(pavement, fence, barrels) and are discarded as outliers.

This file also builds the downsampled RGB+XYZ cache that it and all of Part B
read from, so the 4 GB dataset is decoded exactly once.
"""
import os
import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
DS = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "out")
N = 299
STRIDE = 3          # ~0.3 m of baseline; consecutive frames barely move
DOWN = 4            # cache downsample factor -> 300 x 480
RANSAC_THRESH = 0.40   # metres; stereo noise grows with range
RANSAC_ITERS = 200


def kabsch(P, Q):
    """Rigid transform taking P onto Q (both (N,3)). Returns (R, t)."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # guard against a reflection
    return R, cq - R @ cp


def ransac_rigid(P, Q):
    """Kabsch inside RANSAC, so moving objects cannot drag the fit."""
    n = len(P)
    if n < 6:
        return None, None, 0
    best_in, best = None, 0
    rng = np.random.default_rng(0)
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(P[idx], Q[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        inl = err < RANSAC_THRESH
        if inl.sum() > best:
            best, best_in = inl.sum(), inl
    if best < 6:
        return None, None, 0
    R, t = kabsch(P[best_in], Q[best_in])   # refit on all inliers
    return R, t, int(best)


def build_cache():
    """One pass over the 4 GB dataset, writing a downsampled RGB+XYZ cache.

    Every later stage (this file's VO, and all of Part B) reads the cache
    instead of re-decoding PNGs and .npz archives, which turns a multi-minute
    pass into seconds and makes the detector parameters practical to tune.
    """
    xp = os.path.join(OUT, "xyz_ds.npy")
    rp = os.path.join(OUT, "rgb_ds.npy")
    if os.path.exists(xp) and os.path.exists(rp):
        return
    os.makedirs(OUT, exist_ok=True)
    print("building downsampled cache (one pass over the dataset)...", flush=True)
    H, W = 1200 // DOWN, 1920 // DOWN
    xyz_c = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float16, shape=(N, H, W, 3))
    rgb_c = np.lib.format.open_memmap(rp, mode="w+", dtype=np.uint8, shape=(N, H, W, 3))
    for i in range(N):
        bgr = cv2.imread(os.path.join(DS, f"rgb/left{i:06d}.png"))
        xyz = np.load(os.path.join(DS, f"xyz/depth{i:06d}.npz"))["xyz"][..., :3]
        rgb_c[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[::DOWN, ::DOWN]
        xyz_c[i] = xyz[::DOWN, ::DOWN].astype(np.float16)
        if i % 50 == 0:
            print(f"  cached {i}/{N}", flush=True)
    xyz_c.flush()
    rgb_c.flush()
    print("  cache written")


def main():
    build_cache()
    xyz_c = np.load(os.path.join(OUT, "xyz_ds.npy"), mmap_mode="r")
    rgb_c = np.load(os.path.join(OUT, "rgb_ds.npy"), mmap_mode="r")
    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    hist = {}   # keep the last STRIDE frames, not just the previous one
    dyaw = np.full(N, np.nan)    # yaw change from frame i-STRIDE to frame i
    ninl = np.zeros(N, int)

    for i in range(N):
        xyz = np.asarray(xyz_c[i], np.float32)
        gray = cv2.cvtColor(np.asarray(rgb_c[i]), cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kp], np.float32) if kp else np.zeros((0, 2), np.float32)
        # Attach each keypoint's 3D point now, so history stays small.
        P3 = (xyz[pts[:, 1].astype(int), pts[:, 0].astype(int)]
              if len(pts) else np.zeros((0, 3), np.float32))
        cur = dict(des=des, P3=P3)

        j = i - STRIDE
        if j in hist and des is not None and hist[j]["des"] is not None:
            a = hist[j]
            m = bf.knnMatch(a["des"], des, k=2)
            good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) >= 8:
                Pa = a["P3"][[g.queryIdx for g in good]]
                Pb = cur["P3"][[g.trainIdx for g in good]]
                ok = np.isfinite(Pa).all(1) & np.isfinite(Pb).all(1)
                # Near points have the best stereo accuracy; far ones are noise.
                ok &= (Pa[:, 0] > 2) & (Pa[:, 0] < 40) & (Pb[:, 0] > 2) & (Pb[:, 0] < 40)
                if ok.sum() >= 8:
                    # Transform mapping frame-j points into frame-i coords.
                    R, t, k = ransac_rigid(Pa[ok], Pb[ok])
                    if R is not None:
                        # R maps frame-j coords into frame-i coords, i.e. it is
                        # the INVERSE of the vehicle's rotation, so negate.
                        # The dataset's Y is already LEFT-positive (see
                        # verify_axes.py), matching our world convention, so
                        # that is the only sign flip. (+ = turn left.)
                        dyaw[i] = -np.arctan2(R[1, 0], R[0, 0])
                        ninl[i] = k
        hist[i] = cur
        hist.pop(i - STRIDE, None)      # only the last STRIDE frames are needed
        if i % 25 == 0:
            print(f"frame {i:3d}  dyaw={np.degrees(dyaw[i]) if np.isfinite(dyaw[i]) else float('nan'):+.3f} deg  inliers={ninl[i]}", flush=True)

    np.savez(os.path.join(OUT, "vo.npz"), dyaw=dyaw, ninl=ninl, stride=STRIDE)
    ok = np.isfinite(dyaw)
    print(f"\nVO solved {ok.sum()}/{N-STRIDE} pairs, median inliers {np.median(ninl[ok]):.0f}")
    # Each dyaw spans STRIDE frames and the windows overlap one per frame, so
    # the per-frame yaw RATE is dyaw/STRIDE; summing that recovers the total.
    print(f"total yaw change {np.degrees(np.nansum(dyaw)/STRIDE):+.1f} deg")


if __name__ == "__main__":
    main()
