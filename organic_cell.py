"""
Organic waste basket at the end of general_line.

Organic items (the apples) that ride to the end of the belt are not left
sitting on the end stop: they roll off the end, fall, and land in a basket
on the floor -- a slatted plastic milk-crate. Apples pile up in it.

It is a station of its own so it runs on the shared World clock like the
others (main_sim registers it and gives it a thread). Nothing is spawned here.

Which belt it watches, and which items it takes, are set by the constants
below.
"""
import numpy as np
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid

from conveyors import BELT_HEIGHT
from world import World

DT = 0.05
GRAVITY = 9.81

BELT = "general_line"            # whatever FANUC leaves on the cross line ends up here
MATERIAL = "organic"             # tag set by the infeed; apples only

BASKET_ALONG = 0.40              # basket size along the belt direction
BASKET_ACROSS = 0.56             # across the belt (wider than the belt's legs)
BASKET_HEIGHT = 0.30
CRATE_COLOUR = (0.15, 0.50, 0.30, 1)     # milk-crate green
CRATE_DARK = (0.10, 0.33, 0.20, 1)       # rails and posts
BASKET_WALL = 0.02
BASKET_GAP = 0.18                # basket centre this far beyond the belt end


class _Falling:
    """One apple in flight: rolling off the end, then falling under gravity."""

    def __init__(self, shape, radius, v_forward, v_side):
        self.shape = shape
        self.radius = radius
        self.v = np.array([v_forward, v_side, 0.0])     # belt-frame (along, across, up)
        self.landed = False


class OrganicBasketCell:
    def __init__(self, world: World):
        self.world = world
        self.belt = world.conveyors[BELT]
        self.rng = np.random.default_rng()
        self.falling = []
        self.in_basket = 0

        b = self.belt
        self._build(world.env)

    # ---------- geometry ----------
    def _to_world(self, along, across, z):
        """Belt-frame (distance along belt, sideways, height) -> world xyz."""
        p = self.belt.start + self.belt.dir * along + self.belt.lateral * across
        return np.array([p[0], p[1], z])

    def _bar(self, env, along, across, z, size, colour):
        """One crate bar. along/across are offsets from the crate centre (belt
        frame), z is the bar's centre height, size is (along, across, up)."""
        c = self._to_world(self.mid + along, across, z)
        env.add(Cuboid(list(size), pose=SE3(c[0], c[1], c[2]) * SE3.Rz(self.belt.yaw),
                       color=colour))

    def _build(self, env):
        """A plastic milk-crate: slatted floor, four slatted walls with open
        gaps between the vertical bars, a rail round the top and bottom, and
        solid corner posts."""
        A, W, H, t = BASKET_ALONG, BASKET_ACROSS, BASKET_HEIGHT, BASKET_WALL
        self.mid = self.belt.length + BASKET_GAP
        col, dark = CRATE_COLOUR, CRATE_DARK
        bar = self._bar

        # floor slats (run along the belt) + two cross-bars underneath
        floor_t = 0.015
        for k in range(7):
            across = -W / 2 + 0.04 + k * (W - 0.08) / 6
            bar(env, 0, across, floor_t / 2, (A, 0.05, floor_t), col)
        for along in (-A / 2 + 0.05, A / 2 - 0.05):
            bar(env, along, 0, floor_t / 2, (0.04, W, floor_t), dark)

        # walls: long sides (+/- across) and short ends (+/- along)
        for side in (+1, -1):
            for z in (0.045, H - 0.02):                               # bottom and top rails
                bar(env, 0, side * (W - t) / 2, z, (A, t, 0.04), dark)
            for k in range(6):                                        # vertical slats
                along = -A / 2 + 0.06 + k * (A - 0.12) / 5
                bar(env, along, side * (W - t) / 2, H / 2, (0.03, t, H - 0.04), col)
        for end in (+1, -1):
            for z in (0.045, H - 0.02):
                bar(env, end * (A - t) / 2, 0, z, (t, W, 0.04), dark)
            for k in range(7):
                across = -W / 2 + 0.06 + k * (W - 0.12) / 6
                bar(env, end * (A - t) / 2, across, H / 2, (t, 0.03, H - 0.04), col)

        # corner posts
        for sa in (+1, -1):
            for sw in (+1, -1):
                bar(env, sa * (A - 0.04) / 2, sw * (W - 0.04) / 2, H / 2, (0.04, 0.04, H), dark)

        self.floor_z = floor_t
        self.inner_along = (self.mid - A / 2 + t, self.mid + A / 2 - t)
        self.inner_across = W / 2 - t

    # ---------- behaviour ----------
    def _arrived_organic(self):
        """Organic items that have reached the belt's end stop."""
        stop = self.belt.length - 0.1
        for shape in list(self.world.active_items):
            info = self.world.get_item_info(shape)
            if not info or info.get("material") != MATERIAL:
                continue
            p = np.array(shape.T)[:3, 3]
            if self.belt.contains(p) and self.belt.progress(p) >= stop - 0.005:
                yield shape, info

    def _release(self, shape, info):
        """Take the apple off the belt and give it a little push over the end."""
        self.world.remove_active(shape)
        r = float(info.get("radius", 0.04))
        self.falling.append(_Falling(shape, r,
                                     v_forward=self.rng.uniform(0.20, 0.32),
                                     v_side=self.rng.uniform(-0.05, 0.05)))

    def _advance(self, f):
        T = np.array(f.shape.T)
        p = T[:3, 3]
        along = self.belt.progress(p)
        across = (p - self.belt.start) @ self.belt.lateral
        z = p[2]

        # Still on the belt: keep rolling at the push speed until the apple's
        # centre is just past the end, then it drops.
        on_belt = along < self.belt.length + 0.01 and z >= BELT_HEIGHT
        if not on_belt:
            f.v[2] -= GRAVITY * DT
        along += f.v[0] * DT
        across += f.v[1] * DT
        z += f.v[2] * DT

        # Land on the basket floor, or on the apples already piled in it.
        layer = (self.in_basket // 5) * 0.07
        rest = self.floor_z + f.radius + layer
        if not on_belt and along > self.belt.length - 0.02 and z <= rest:
            lo, hi = self.inner_along
            along = float(np.clip(along, lo + f.radius, hi - f.radius))
            across = float(np.clip(across, -self.inner_across + f.radius,
                                   self.inner_across - f.radius))
            z = rest
            f.landed = True
            self.in_basket += 1

        T[:3, 3] = self._to_world(along, across, z)
        f.shape.T = T

    def step(self):
        for shape, info in self._arrived_organic():
            self._release(shape, info)
        for f in self.falling:
            self._advance(f)
        self.falling = [f for f in self.falling if not f.landed]
        self.world.tick()

    def run(self):
        while True:
            self.step()


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    OrganicBasketCell(World(env, dt=DT)).run()