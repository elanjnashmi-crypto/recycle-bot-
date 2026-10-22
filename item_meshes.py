import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "meshes", "items")
BOTTLE_STL = os.path.join(MESH_DIR, "plastic_bottle.stl")

BOTTLE_HEIGHT = 0.20
BOTTLE_RADIUS = 0.032          # body radius, used as the grip radius

CAN_RADIUS, CAN_HEIGHT = 0.033, 0.122      # both metal cans are the same size
APPLE_RADIUS, APPLE_HEIGHT = 0.040, 0.080

# (radius, height-from-base) outline, bottom -> top
_PROFILE = [
    (0.000, 0.000), (0.022, 0.000), (0.029, 0.004), (0.032, 0.012),      # rounded base
    (0.032, 0.055), (0.0295, 0.060), (0.0295, 0.070), (0.032, 0.075),   # grip groove
    (0.032, 0.095), (0.0295, 0.100), (0.0295, 0.110), (0.032, 0.115),   # grip groove
    (0.032, 0.128), (0.030, 0.140), (0.024, 0.154), (0.016, 0.166),     # shoulder
    (0.0135, 0.173), (0.0135, 0.182),                                    # neck
    (0.0165, 0.182), (0.0165, 0.200), (0.000, 0.200),                    # cap
]

# 375 ml drink can: domed base on a stand ring, straight wall, necked top, recessed lid
_AL_CAN_PROFILE = [
    (0.000, 0.012), (0.012, 0.010), (0.020, 0.006), (0.024, 0.002),      # dome (pushed up into the can)
    (0.026, 0.000), (0.029, 0.000), (0.032, 0.004), (0.033, 0.010),      # stand ring
    (0.033, 0.105),                                                      # wall
    (0.031, 0.111), (0.028, 0.116), (0.027, 0.119),                      # neck
    (0.0275, 0.122), (0.0265, 0.122), (0.0255, 0.118),                   # rim lip
    (0.024, 0.117), (0.000, 0.117),                                      # recessed lid
]

# Food tin: double-seamed rims top and bottom, beaded wall, recessed lids with expansion rings
_STEEL_CAN_PROFILE = [
    (0.000, 0.003), (0.020, 0.003), (0.022, 0.001), (0.024, 0.003),      # bottom lid + ring
    (0.031, 0.003), (0.031, 0.000), (0.0335, 0.000), (0.0335, 0.006),    # bottom seam
    (0.032, 0.008),
] + [pt for z in (0.024, 0.040, 0.082, 0.098)                             # shallow beads, label band between
     for pt in ((0.032, z - 0.004), (0.0315, z - 0.0015), (0.0313, z), (0.0315, z + 0.0015),
                (0.032, z + 0.004))] + [
    (0.032, 0.114), (0.0335, 0.116), (0.0335, 0.122), (0.031, 0.122),    # top seam
    (0.031, 0.119), (0.024, 0.119), (0.022, 0.121), (0.020, 0.119),      # top lid + ring
    (0.000, 0.119),
]


def _apple_profile():
    """Apple outline: wider shoulders, narrower base, dimples top and bottom, short stem."""
    pts = [(0.0, 0.007), (0.007, 0.004)]                                 # bottom dimple
    for t in np.linspace(0.12, 0.88, 16):
        a = np.pi * t
        r = APPLE_RADIUS * np.sin(a) * (0.92 + 0.10 * (1 - np.cos(a)) / 2)
        z = APPLE_HEIGHT / 2 * (1 - np.cos(a))
        pts.append((r, z))
    pts += [(0.010, APPLE_HEIGHT - 0.0015), (0.004, APPLE_HEIGHT - 0.009),  # top dimple
            (0.0022, APPLE_HEIGHT - 0.009), (0.0018, APPLE_HEIGHT + 0.006),  # stem
            (0.0, APPLE_HEIGHT + 0.006)]
    z0 = min(z for _, z in pts)                                          # sit on its lowest point
    return [(r, z - z0) for r, z in pts]


def _revolve(profile, height, segments=32):
    """Spin a (radius, z) outline about z; origin at the middle of `height`."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [[np.array((r * np.cos(a), r * np.sin(a), z - height / 2)) for a in angles]
             for r, z in profile]
    tris = []
    for lo, hi in zip(rings[:-1], rings[1:]):
        for i in range(segments):
            j = (i + 1) % segments
            tris += [(lo[i], lo[j], hi[i]), (lo[j], hi[j], hi[i])]
    return tris


def _box(centre, size):
    """Closed box as outward-facing triangles."""
    c, h = np.array(centre, float), np.array(size, float) / 2
    v = [c + h * np.array(s) for s in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                                       (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return [(v[a], v[b], v[c_]) for a, b, c_ in faces]


def _write_stl(path, tris, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    good = []
    for a, b, c in tris:
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-12:                    # degenerate sliver at the axis
            continue
        good.append((n / length, a, b, c))
    out = bytearray(name.encode().ljust(80, b"\0"))
    out += struct.pack("<I", len(good))
    for n, a, b, c in good:
        out += struct.pack("<12fH", *n, *a, *b, *c, 0)
    with open(path, "wb") as f:
        f.write(out)
    return path


def build_bottle(path=BOTTLE_STL, segments=32):
    return _write_stl(path, _revolve(_PROFILE, BOTTLE_HEIGHT, segments), "plastic bottle")


def _al_can_tris():
    tris = _revolve(_AL_CAN_PROFILE, CAN_HEIGHT, 40)
    top = 0.117 - CAN_HEIGHT / 2
    tris += _box((0.008, 0, top + 0.0012), (0.020, 0.010, 0.0012))     # ring-pull tab
    tris += _box((-0.009, 0, top + 0.0006), (0.012, 0.008, 0.0004))    # opening score
    return tris


ITEM_MESHES = {
    "bottle": ("plastic_bottle.stl", lambda: _revolve(_PROFILE, BOTTLE_HEIGHT)),
    "al_can": ("aluminium_can.stl", _al_can_tris),
    "steel_can": ("steel_can.stl", lambda: _revolve(_STEEL_CAN_PROFILE, CAN_HEIGHT, 40)),
    "apple": ("apple.stl", lambda: _revolve(_apple_profile(), APPLE_HEIGHT, 36)),
}


def ensure_mesh(kind):
    """Return the STL path for an item mesh, generating it on first use."""
    fname, make = ITEM_MESHES[kind]
    path = os.path.join(MESH_DIR, fname)
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        _write_stl(path, make(), kind)
    return path


def ensure_bottle_mesh():
    return ensure_mesh("bottle")


if __name__ == "__main__":
    for kind, (fname, make) in ITEM_MESHES.items():
        print("wrote", _write_stl(os.path.join(MESH_DIR, fname), make(), kind))
