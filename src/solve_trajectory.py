"""Step 2: recover the ego-vehicle trajectory in the ground frame.

Geometry recap (all angles radians, all lengths metres)
-------------------------------------------------------
Car frame : x = forward, y = LEFT, z = up -- which is what the dataset
            already uses, so y = +Y.  NOTE: the challenge README says
            "+Y -> right axis (right-handed)", but X-fwd/Y-right/Z-up is a
            LEFT-handed triad, so that description contradicts itself. The
            data is genuinely right-handed: verified empirically in
            verify_axes.py (ground pixels on the left edge of the image give
            Y>0, the right edge gives Y<0) and confirmed by the traffic
            light's bearing sign tracking its pixel column. We ground-project
            by discarding Z.
World frame: origin on the ground directly under the traffic light, Z up.
            At t=0 the car lies on the +X axis (challenge spec).

The light sits at the world origin, so if the car is at C_t with heading
theta_t, the light *as seen from the car* is

        q_t = R(theta_t)^T (0 - C_t)      =>      C_t = -R(theta_t) q_t

Writing q_t = r_t (cos a_t, sin a_t) and defining  phi_t = theta_t + a_t + pi
collapses that to polar coordinates about the light:

        C_t = r_t (cos phi_t, sin phi_t)

r_t is measured directly every frame and is never integrated, so the
trajectory cannot drift in scale.  Only phi_t must be integrated, and one
static landmark does not observe it (3 pose unknowns, 2 measurements).  We
close the gap with the non-holonomic constraint: a car rolls along its
heading and cannot slide sideways.  Per frame it advances s_t and turns
dtheta_t, which makes the system exactly determined:

    lengths:  r_{t+1}^2 = (X_t - s_t)^2 + y_t^2   ->  s_t  (the speed)
    angles :  dphi_t = atan2(y_t, X_t - s_t) - atan2(y_t, X_t)

i.e. "slide forward by s_t and watch how far the bearing to the light
swings; that swing is your angular progress around the light".
"""
import csv, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
FPS = 29.9  # 299 frames over the stated 10 s clip

# Frame-to-frame range noise (0.48 m) is ~5x the frame-to-frame motion
# (0.10 m), so the raw signal must be smoothed before it is differenced.
SMOOTH_WIN = 31     # ~1 s of context
SMOOTH_ORDER = 2    # locally constant acceleration


def savgol(y, win, order, deriv=0):
    """Savitzky-Golay: slide a window, least-squares fit a low-order
    polynomial, keep its value (or its DERIVATIVE) at the centre. Removes
    noise while preserving the shape of a smooth trajectory (unlike a boxcar,
    which flattens turns).

    deriv=1 returns d/d(sample) of the fitted polynomial. This is the key to
    a stable solve: we differentiate a fitted curve rather than differencing
    noisy samples, which would amplify the noise instead of the signal."""
    n = len(y)
    win = min(win | 1, n if n % 2 else n - 1)
    half = win // 2
    out = np.empty(n)
    idx = np.arange(-half, half + 1)
    V = np.vander(idx, order + 1)
    pinv = np.linalg.pinv(V)
    for i in range(n):
        lo, hi = i - half, i + half + 1
        if lo < 0:      lo, hi = 0, win
        elif hi > n:    lo, hi = n - win, n
        coef = pinv @ y[lo:hi]
        if deriv:
            coef = np.polyder(coef, deriv)
        out[i] = np.polyval(coef, i - (lo + half))
    return out


def load_track():
    """Read the per-frame light fixes, fill the handful of missing frames."""
    rows = list(csv.DictReader(open(os.path.join(OUT, "light_track.csv"))))
    f = np.array([int(r["frame"]) for r in rows], float)
    X = np.array([float(r["X"]) if r["X"] else np.nan for r in rows])
    Y = np.array([float(r["Y"]) if r["Y"] else np.nan for r in rows])
    Z = np.array([float(r["Z"]) if r["Z"] else np.nan for r in rows])
    good = np.isfinite(X)
    # 4 frames have a 0,0,0,0 bbox (no detection). The vehicle moves ~0.1 m
    # per frame, so linear interpolation across a 1-3 frame gap is far below
    # the measurement noise and keeps the arrays rectangular.
    for a in (X, Y, Z):
        a[~good] = np.interp(f[~good], f[good], a[good])
    return f, X, Y, Z, good


def solve(X, y, Xd, yd):
    """Recover (phi, theta, speed) per frame.

    X, y   : light position in the car frame (forward, left), smoothed
    Xd, yd : their time derivatives, per second, from the Savitzky-Golay fit

    The exact discrete non-holonomic solve is

        s_t = X_t - sqrt(r_{t+1}^2 - y_t^2)

    which is algebraically right but numerically fragile: it subtracts two
    nearly-equal ~36 m numbers to get ~0.1 m, so noise is amplified ~300x
    (catastrophic cancellation). Its continuous limit is the same physics in
    a stable form -- the closing rate on the light is the radial component of
    our velocity:

        rdot = -v cos(alpha)      =>  v     = -rdot / cos(alpha)
                                      phidot = -(rdot / r) tan(alpha)

    Only phi is integrated; r is re-measured every frame.
    """
    r = np.hypot(X, y)
    a = np.arctan2(y, X)                  # bearing to the light, + = left
    rdot = (X * Xd + y * yd) / r          # range rate, m/s

    ca = np.cos(a)
    speed = -rdot / np.clip(ca, 0.2, None)   # guard the (never-hit) 90 deg case
    phidot = -(rdot / r) * np.tan(a)         # rad/s

    # Integrate the single unobservable DOF. phi_0 = 0 puts the car on the
    # +X axis at t=0, per the spec.
    phi = np.concatenate([[0.0], np.cumsum(phidot[:-1]) / FPS])
    theta = phi - a - np.pi               # vehicle heading in the world frame
    return phi, theta, speed


def main():
    f, X, Y, Z, good = load_track()
    y = Y                             # dataset Y is already LEFT-positive

    Xs = savgol(X, SMOOTH_WIN, SMOOTH_ORDER)
    ys = savgol(y, SMOOTH_WIN, SMOOTH_ORDER)
    Xd = savgol(X, SMOOTH_WIN, SMOOTH_ORDER, deriv=1) * FPS   # m/s
    yd = savgol(y, SMOOTH_WIN, SMOOTH_ORDER, deriv=1) * FPS
    phi, theta, speed = solve(Xs, ys, Xd, yd)

    r = np.hypot(Xs, ys)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    np.savez(os.path.join(OUT, "trajectory.npz"),
             frame=f, pos=pos, phi=phi, theta=theta, speed=speed,
             r=r, X=Xs, y=ys, X_raw=X, y_raw=y, Z=Z, detected=good)

    with open(os.path.join(OUT, "trajectory.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "x_m", "y_m", "heading_deg", "range_m", "speed_mps"])
        for i in range(len(f)):
            w.writerow([int(f[i]), round(i / FPS, 4), round(pos[i, 0], 4),
                        round(pos[i, 1], 4), round(np.degrees(theta[i]), 3),
                        round(r[i], 4), round(speed[i], 4)])

    d = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
    print(f"start      {pos[0]}   range {r[0]:.2f} m")
    print(f"end        {pos[-1]}   range {r[-1]:.2f} m")
    print(f"path length        {d:.2f} m over {len(f)/FPS:.1f} s")
    print(f"mean speed         {d/(len(f)/FPS):.2f} m/s  ({d/(len(f)/FPS)*2.237:.1f} mph)")
    print(f"heading change     {np.degrees(theta[-1]-theta[0]):+.1f} deg")
    print(f"arc swept (phi)    {np.degrees(phi[-1]-phi[0]):+.1f} deg")
    print(f"light height Z     {np.nanmean(Z):.2f} +/- {np.nanstd(Z):.2f} m")


if __name__ == "__main__":
    main()
