import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "meshes", "items")
BOTTLE_STL = os.path.join(MESH_DIR, "plastic_bottle.stl")

BOTTLE_HEIGHT = 0.20
BOTTLE_RADIUS = 0.032          # body radius, used as the grip radius

# (radius, height-from-base) outline, bottom -> top
_PROFILE = [
    (0.000, 0.000), (0.022, 0.000), (0.029, 0.004), (0.032, 0.012),      # rounded base
    (0.032, 0.055), (0.0295, 0.060), (0.0295, 0.070), (0.032, 0.075),   # grip groove
    (0.032, 0.095), (0.0295, 0.100), (0.0295, 0.110), (0.032, 0.115),   # grip groove
    (0.032, 0.128), (0.030, 0.140), (0.024, 0.154), (0.016, 0.166),     # shoulder
    (0.0135, 0.173), (0.0135, 0.182),                                    # neck
    (0.0165, 0.182), (0.0165, 0.200), (0.000, 0.200),                    # cap
]


def build_bottle(path=BOTTLE_STL, segments=32):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [[(r * np.cos(a), r * np.sin(a), z - BOTTLE_HEIGHT / 2) for a in angles]
             for r, z in _PROFILE]
    tris = []
    for lo, hi in zip(rings[:-1], rings[1:]):
        for i in range(segments):
            j = (i + 1) % segments
            p00, p01, p10, p11 = map(np.array, (lo[i], lo[j], hi[i], hi[j]))
            tris += [(p00, p01, p10), (p01, p11, p10)]
    out = bytearray(b"plastic bottle".ljust(80, b"\0"))
    good = []
    for a, b, c in tris:
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-12:                    # degenerate sliver at the axis
            continue
        good.append((n / length, a, b, c))
    out += struct.pack("<I", len(good))
    for n, a, b, c in good:
        out += struct.pack("<12fH", *n, *a, *b, *c, 0)
    with open(path, "wb") as f:
        f.write(out)
    return path


def ensure_bottle_mesh():
    """Return the bottle STL path, generating it on first use."""
    if not os.path.isfile(BOTTLE_STL) or os.path.getsize(BOTTLE_STL) < 1000:
        build_bottle()
    return BOTTLE_STL


if __name__ == "__main__":
    print("wrote", build_bottle())