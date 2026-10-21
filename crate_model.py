
from spatialmath import SE3
from spatialgeometry import Cuboid

WALL = 0.02
FLOOR_T = 0.015


def build_milk_crate(env, x, y, z0, A, W, H, colour, dark, yaw=0.0):
    base = SE3(x, y, z0) * SE3.Rz(yaw)

    def bar(da, dw, z, size, col):
        env.add(Cuboid(list(size), pose=base * SE3(da, dw, z), color=col))

    t = WALL
    for k in range(5):                                         # floor slats
        bar(0, -W / 2 + 0.04 + k * (W - 0.08) / 4, FLOOR_T / 2, (A, 0.05, FLOOR_T), colour)
    for da in (-A / 2 + 0.05, A / 2 - 0.05):                   # floor cross-bars
        bar(da, 0, FLOOR_T / 2, (0.04, W, FLOOR_T), dark)
    for side in (+1, -1):                                      # long walls
        for z in (0.045, H - 0.02):
            bar(0, side * (W - t) / 2, z, (A, t, 0.04), dark)
        for k in range(5):
            bar(-A / 2 + 0.06 + k * (A - 0.12) / 4, side * (W - t) / 2, H / 2, (0.03, t, H - 0.04), colour)
    for end in (+1, -1):                                       # short walls
        for z in (0.045, H - 0.02):
            bar(end * (A - t) / 2, 0, z, (t, W, 0.04), dark)
        for k in range(5):
            bar(end * (A - t) / 2, -W / 2 + 0.06 + k * (W - 0.12) / 4, H / 2, (t, 0.03, H - 0.04), colour)
    for sa in (+1, -1):                                        # corner posts
        for sw in (+1, -1):
            bar(sa * (A - 0.04) / 2, sw * (W - 0.04) / 2, H / 2, (0.04, 0.04, H), dark)
    return z0 + FLOOR_T, A / 2 - t, W / 2 - t