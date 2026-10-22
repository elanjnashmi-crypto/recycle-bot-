import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

# ---- building shell ----
X0, X1 = -8.0, 9.0
Y0, Y1 = -10.0, 10.0
WALL_H = 6.0
WALL_T = 0.20
DOOR_X = (-1.75, 1.75)          # roller door in the south wall, in line with the cell gate
DOOR_H = 4.0
COLUMN_YS = (-6.0, -2.0, 2.0, 6.0)   # truss lines (columns on the east/west walls)
COLUMN_XS = (-4.0, 4.5)              # extra columns on the north/south walls

# Where Swift's camera starts
CAMERA_POS = [7.6, -8.6, 5.2]
CAMERA_LOOK = [0.3, 0.6, 0.6]

# ---- colours ----
CLADDING = (0.80, 0.82, 0.84, 1)
WALL_BASE = (0.36, 0.38, 0.40, 1)
GLAZING = (0.55, 0.75, 0.90, 0.35)
STEEL_BLUE = (0.16, 0.24, 0.38, 1)
RACK_BLUE = (0.10, 0.30, 0.65, 1)
RACK_ORANGE = (0.95, 0.45, 0.05, 1)
SAFETY_YELLOW = (0.98, 0.80, 0.05, 1)
WALKWAY_GREEN = (0.20, 0.55, 0.30, 1)
BLACK = (0.08, 0.08, 0.08, 1)
PALLET_WOOD = (0.62, 0.47, 0.28, 1)
CARDBOARD = (0.70, 0.54, 0.34, 1)
SHRINK_WRAP = (0.90, 0.92, 0.95, 0.55)
LAMP = (1.00, 0.97, 0.80, 1)
RED = (0.80, 0.05, 0.05, 1)

BALE_COLOURS = {
    "aluminium": (0.78, 0.80, 0.84, 1),
    "steel": (0.46, 0.43, 0.40, 1),
    "plastic": (0.25, 0.55, 0.85, 1),
    "cardboard": (0.66, 0.50, 0.30, 1),
}


def _box(env, size, xyz, colour, yaw=0.0):
    env.add(Cuboid(list(size), pose=SE3(*xyz) * SE3.Rz(yaw), color=colour))


def _bar(env, a, b, thick, colour):
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    mid = (a + b) / 2
    yaw = float(np.arctan2(d[1], d[0]))
    pitch = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    env.add(Cuboid([L, thick, thick], pose=SE3(*mid) * SE3.Rz(yaw) * SE3.Ry(pitch), color=colour))


# ---------- shell ----------
def _wall_x(env, x, y_from, y_to, z_from, z_to, colour):
    _box(env, (WALL_T, y_to - y_from, z_to - z_from), (x, (y_from + y_to) / 2, (z_from + z_to) / 2), colour)


def _wall_y(env, y, x_from, x_to, z_from, z_to, colour):
    _box(env, (x_to - x_from, WALL_T, z_to - z_from), ((x_from + x_to) / 2, y, (z_from + z_to) / 2), colour)


# Wall bands bottom -> top
BANDS = [(0.0, 1.2, WALL_BASE), (1.2, 4.2, CLADDING), (4.2, 5.2, GLAZING), (5.2, WALL_H, CLADDING)]


def _walls(env):
    for z0, z1, c in BANDS:
        _wall_x(env, X0, Y0, Y1, z0, z1, c)
        _wall_x(env, X1, Y0, Y1, z0, z1, c)
        _wall_y(env, Y1, X0, X1, z0, z1, c)
        if z1 <= DOOR_H:                                    # south wall: leave the door opening
            _wall_y(env, Y0, X0, DOOR_X[0], z0, z1, c)
            _wall_y(env, Y0, DOOR_X[1], X1, z0, z1, c)
        elif z0 < DOOR_H:
            _wall_y(env, Y0, X0, DOOR_X[0], z0, DOOR_H, c)
            _wall_y(env, Y0, DOOR_X[1], X1, z0, DOOR_H, c)
            _wall_y(env, Y0, X0, X1, DOOR_H, z1, c)
        else:
            _wall_y(env, Y0, X0, X1, z0, z1, c)

    # roller door (closed): slats, hazard-striped jambs, header box
    w = DOOR_X[1] - DOOR_X[0]
    _box(env, (w, 0.08, DOOR_H), (0, Y0, DOOR_H / 2), (0.62, 0.64, 0.66, 1))
    for z in np.arange(0.25, DOOR_H, 0.25):
        _box(env, (w, 0.10, 0.02), (0, Y0, z), (0.48, 0.50, 0.52, 1))
    for x in DOOR_X:
        for i, z in enumerate(np.arange(0.0, DOOR_H, 0.25)):
            _box(env, (0.12, 0.26, 0.25), (x, Y0, z + 0.125), SAFETY_YELLOW if i % 2 == 0 else BLACK)
    _box(env, (w + 0.4, 0.45, 0.45), (0, Y0 + 0.25, DOOR_H + 0.25), STEEL_BLUE)


def _structure(env):
    """Steel columns on the walls, roof trusses across, purlins along."""
    cols = [(X0 + 0.25, y) for y in COLUMN_YS] + [(X1 - 0.25, y) for y in COLUMN_YS]
    cols += [(x, Y0 + 0.25) for x in COLUMN_XS] + [(x, Y1 - 0.25) for x in COLUMN_XS]
    for x, y in cols:
        _box(env, (0.30, 0.30, WALL_H), (x, y, WALL_H / 2), STEEL_BLUE)

    top, rise = WALL_H - 0.2, 1.0
    xm = (X0 + X1) / 2
    for y in COLUMN_YS:                                     # pitched trusses
        a, m, b = (X0 + 0.25, y, top), (xm, y, top + rise), (X1 - 0.25, y, top)
        _bar(env, a, m, 0.18, STEEL_BLUE)                   # rafters
        _bar(env, m, b, 0.18, STEEL_BLUE)
        _bar(env, a, b, 0.12, STEEL_BLUE)                   # bottom chord
        for f in (0.25, 0.5, 0.75):                         # web
            for side in (-1, 1):
                x = xm + side * f * (xm - X0 - 0.25)
                z_top = top + rise * (1 - f)
                _bar(env, (x, y, top), (x, y, z_top), 0.08, STEEL_BLUE)
    for f in (0.0, 0.5, 1.0):                               # purlins along the building
        for side in (-1, 1):
            x = xm + side * f * (xm - X0 - 0.25)
            z = top + rise * (1 - f) + 0.12
            _bar(env, (x, Y0 + 0.2, z), (x, Y1 - 0.2, z), 0.10, STEEL_BLUE)


def _lights(env):
    """High-bay light fittings hung from the trusses."""
    for y in COLUMN_YS:
        for x in (-5.0, -1.0, 3.0, 7.0):
            _box(env, (1.2, 0.30, 0.10), (x, y, 4.9), (0.25, 0.25, 0.27, 1))
            _box(env, (1.1, 0.24, 0.02), (x, y, 4.84), LAMP)
            env.add(Cylinder(0.01, 0.75, pose=SE3(x, y, 5.33), color=BLACK))


# ---------- floor markings ----------
def _markings(env):
    # Swift's own grey ground plane (z = 0) is the concrete floor.
    z = 0.0045

    def strip(x0, y0, x1, y1, w=0.10, colour=SAFETY_YELLOW):
        if x0 == x1:
            _box(env, (w, abs(y1 - y0), 0.003), (x0, (y0 + y1) / 2, z), colour)
        else:
            _box(env, (abs(x1 - x0), w, 0.003), ((x0 + x1) / 2, y0, z), colour)

    _box(env, (1.6, 5.15, 0.002), (0, -7.35, 0.003), WALKWAY_GREEN)
    strip(-0.85, -9.9, -0.85, -4.75)
    strip(0.85, -9.9, 0.85, -4.75)

    # keep-out line 0.5 m around the fence, broken at the walkway
    strip(-3.30, -4.75, -3.30, 5.50)
    strip(3.80, -4.75, 3.80, 5.50)
    strip(-3.30, 5.50, 3.80, 5.50)
    strip(-3.30, -4.75, -0.85, -4.75)
    strip(0.85, -4.75, 3.80, -4.75)

    # forklift aisle lines
    strip(4.4, -9.9, 4.4, 9.9, w=0.08)
    strip(-6.0, -9.9, -6.0, 9.9, w=0.08)


# ---------- storage ----------
def _racking(env, rng):
    """Two-level pallet racking along the west wall, five bays."""
    x_back, x_front = X0 + 0.45, X0 + 1.45
    bays = np.arange(-6.75, 6.75 + 1e-6, 2.7)
    levels = (1.6, 3.1)
    for y in bays:                                          # upright frames
        for x in (x_back, x_front):
            _box(env, (0.08, 0.08, 4.4), (x, y, 2.2), RACK_BLUE)
        for zz in (0.5, 2.0, 3.5):
            _box(env, (x_front - x_back, 0.05, 0.05), ((x_back + x_front) / 2, y, zz), RACK_BLUE)
    for y0, y1 in zip(bays[:-1], bays[1:]):
        for z in levels:                                    # beams
            for x in (x_back, x_front):
                _box(env, (0.06, y1 - y0, 0.12), (x, (y0 + y1) / 2, z - 0.06), RACK_ORANGE)
        for z in (0.0,) + levels:                           # pallets + loads
            for k in range(2):
                if rng.random() < 0.2:
                    continue                                # some empty slots
                yc = y0 + (k + 0.5) * (y1 - y0) / 2
                xc = (x_back + x_front) / 2
                _box(env, (1.0, 1.15, 0.14), (xc, yc, z + 0.07), PALLET_WOOD)
                h = rng.uniform(0.6, 1.1)
                colour = CARDBOARD if rng.random() < 0.6 else SHRINK_WRAP
                _box(env, (0.95, 1.1, h), (xc, yc, z + 0.14 + h / 2), colour)


def _bales(env, rng):
    kinds = list(BALE_COLOURS)
    spots = [(x, y) for x in (5.3, 6.6, 7.9) for y in (-3.0, -1.4, 0.2)]
    for i, (x, y) in enumerate(spots):
        kind = kinds[i % len(kinds)]
        stack = 2 if rng.random() < 0.5 else 1
        for s in range(stack):
            z = 0.4 + s * 0.8
            yaw = rng.uniform(-0.08, 0.08)
            _box(env, (1.1, 1.1, 0.8), (x, y, z), BALE_COLOURS[kind], yaw)
            for dx in (-0.3, 0.3):                          # wire straps
                _box(env, (0.02, 1.12, 0.82), (x + dx, y, z), BLACK, yaw)
    for i, y in enumerate((6.6, 7.9)):                      # stacked empty pallets by the north wall
        for k in range(4 + i * 2):
            _box(env, (1.2, 1.0, 0.14), (7.6, y, 0.07 + k * 0.145), PALLET_WOOD, 0.03 * (k % 2))


def _forklift(env, x, y, yaw):
    base = SE3(x, y, 0) * SE3.Rz(yaw)

    def part(size, xyz, colour):
        env.add(Cuboid(list(size), pose=base * SE3(*xyz), color=colour))

    part((1.3, 0.95, 0.55), (0, 0, 0.45), RACK_ORANGE)                  # body
    part((0.40, 0.95, 0.60), (-0.62, 0, 0.75), (0.30, 0.30, 0.32, 1))   # counterweight
    part((0.45, 0.45, 0.10), (-0.15, 0, 0.78), BLACK)                   # seat
    part((0.10, 0.45, 0.45), (-0.38, 0, 1.0), BLACK)                    # backrest
    for px in (-0.45, 0.35):                                            # overhead guard posts
        for py in (-0.42, 0.42):
            part((0.05, 0.05, 1.45), (px, py, 1.45), BLACK)
    part((0.90, 0.95, 0.05), (-0.05, 0, 2.18), BLACK)                   # guard roof
    for py in (-0.30, 0.30):                                            # mast + forks
        part((0.08, 0.10, 2.2), (0.72, py, 1.1), (0.25, 0.25, 0.27, 1))
        part((1.1, 0.10, 0.04), (1.25, py, 0.10), (0.25, 0.25, 0.27, 1))
    part((0.08, 0.80, 0.50), (0.78, 0, 0.40), (0.25, 0.25, 0.27, 1))    # fork carriage
    for wx, r in ((0.40, 0.24), (-0.45, 0.20)):                         # wheels
        for side in (-1, 1):
            env.add(Cylinder(r, 0.18, pose=base * SE3(wx, side * 0.50, r) * SE3.Rx(np.pi / 2),
                             color=BLACK))


def _extinguishers(env):
    """Fire extinguisher + red sign on the inside face of four columns."""
    for x, y, nx in ((X0 + 0.25, -2.0, 1), (X0 + 0.25, 6.0, 1), (X1 - 0.25, -6.0, -1), (X1 - 0.25, 2.0, -1)):
        px = x + nx * 0.25
        env.add(Cylinder(0.08, 0.50, pose=SE3(px, y, 0.55), color=RED))
        _box(env, (0.02, 0.30, 0.30), (x + nx * 0.16, y, 1.9), RED)


def build_factory(env, set_camera=True, seed=7):
    """Add the whole building to Swift. Call before the station threads start."""
    rng = np.random.default_rng(seed)           # fixed seed: same layout every run
    _markings(env)
    _walls(env)
    _structure(env)
    _lights(env)
    _racking(env, rng)
    _bales(env, rng)
    _forklift(env, 6.2, 4.8, np.pi)
    _extinguishers(env)
    if set_camera and not getattr(env, "headless", False):
        env.set_camera_pose(CAMERA_POS, CAMERA_LOOK)


if __name__ == "__main__":
    import swift
    env = swift.Swift()
    env.launch(realtime=True)
    build_factory(env)
    env.hold()
