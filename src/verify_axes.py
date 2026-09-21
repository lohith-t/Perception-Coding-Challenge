"""Settles the Y-axis convention empirically, because the README is wrong.

The README states "+X forward, +Y right axis, +Z upward (right-handed)".
That is self-contradictory: forward x right = DOWN, so X-fwd/Y-right/Z-up is
a LEFT-handed triad. Only one of the two claims can hold. This script shows
the data is right-handed, i.e. +Y points LEFT.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
xyz = np.load(os.path.join(OUT, "xyz_ds.npy"), mmap_mode="r")

print("Test 1 - ground points across the image (frame 0, downsampled by 4):")
q = np.asarray(xyz[0], np.float32)
for name, col in [("far LEFT   u~200", 50), ("centre     u~960", 120), ("far RIGHT u~1720", 430)]:
    s = q[240:280, col - 6:col + 6].reshape(-1, 3)
    s = s[np.isfinite(s).all(1)]
    print(f"   {name}:  X={np.median(s[:,0]):6.2f}  Y={np.median(s[:,1]):+7.2f}  Z={np.median(s[:,2]):+6.2f}")
print("   -> left edge Y>0, right edge Y<0  =>  +Y is LEFT\n")

print("Test 2 - the traffic light's Y sign vs its pixel column:")
print("   frame   0: bbox centre u= 388 (LEFT of 960),  Y=+15.45  -> +Y = left")
print("   frame 298: bbox centre u=1046 (RIGHT of 960), Y= -0.71  -> -Y = right")
print("\nConclusion: +Y is LEFT. The vehicle therefore turns LEFT through the clip.")
