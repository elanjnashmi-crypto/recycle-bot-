"""
Conveyor belts for the recycle-bot workcell (Swift, Python).

Layout follows the hand sketch: world origin at the FANUC base, x to the right
of the sketch, y towards the top of the sketch, z up. All units are metres.
Edit LAYOUT below to move or resize belts; everything else adapts.

Usage:
    import swift
    from environment.conveyors import ConveyorSystem

    env = swift.Swift(); env.launch(realtime=True)
    conveyors = ConveyorSystem(env)
    items = [...]                      # spatialgeometry shapes sitting on belts
    while True:
        conveyors.step(0.05, items)    # call once per sim step
        env.step(0.05)
"""

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid

BELT_HEIGHT = 0.75      # top of belt surface above floor
BELT_THICK = 0.04
DEFAULT_WIDTH = 0.40
DEFAULT_SPEED = 0.15    # m/s

# Ordered upstream -> downstream. At a transfer the receiving belt starts
# under the end of the feeding belt; where they overlap, the later
# (downstream) belt takes over the item.
# name: (start_xy, end_xy) -- items flow from start to end.
LAYOUT = {
    "main_line":    ((-0.30,  3.40), (-0.30,  0.75)),  # Kawasaki places items here
    "cross_line":   (( 0.60,  0.75), (-2.00,  0.75)),  # FANUC picks plastic/metal from here
    "general_line": ((-2.00,  0.95), (-2.00, -2.60)),  # whatever FANUC leaves -> general deposit
    "plastic_line": ((-0.90,  0.20), (-0.90, -2.60)),  # FANUC places plastic
    "metal_line":   (( 0.90,  0.20), ( 0.90, -2.90)),  # FANUC places metal
    "metal_bottom": (( 0.70, -2.90), ( 2.50, -2.90)),  # carries metal right
    "metal_return": (( 2.50, -3.10), ( 2.50,  1.80)),  # up to the Al / non-Al sorting robot
}

# Belts that stop items at their end (a deposit bin or a robot pick point)
# instead of letting them run off.
END_STOPS = {"general_line", "plastic_line", "metal_return"}


class Conveyor:
    """One straight belt: frame, legs, side rails and moving stripes."""

    def __init__(self, env, name, start, end, width=DEFAULT_WIDTH,
                 speed=DEFAULT_SPEED, stop_at_end=False):
        self.name = name
        self.start = np.array([start[0], start[1], BELT_HEIGHT], dtype=float)
        self.end = np.array([end[0], end[1], BELT_HEIGHT], dtype=float)
        self.length = float(np.linalg.norm(self.end[:2] - self.start[:2]))
        self.dir = (self.end - self.start) / self.length          # unit flow direction
        self.lateral = np.array([-self.dir[1], self.dir[0], 0.0])  # left of flow
        self.width = width
        self.speed = speed
        self.stop_at_end = stop_at_end
        self.yaw = np.arctan2(self.dir[1], self.dir[0])
        self._stripe_offset = 0.0
        self._build(env)

    # ---------- geometry ----------
    def _pose(self, s, lateral=0.0, z=BELT_HEIGHT):
        """SE3 at distance s along the belt centreline, aligned with flow."""
        p = self.start + s * self.dir + lateral * self.lateral
        return SE3(p[0], p[1], z) * SE3.Rz(self.yaw)

    def _build(self, env):
        L, w = self.length, self.width
        mid = L / 2

        # belt surface
        self.belt = Cuboid([L, w, BELT_THICK],
                           pose=self._pose(mid, z=BELT_HEIGHT - BELT_THICK / 2),
                           color=[0.15, 0.15, 0.15, 1])
        env.add(self.belt)

        # side rails (raised so items can't slide off sideways)
        for side in (+1, -1):
            rail = Cuboid([L, 0.03, 0.08],
                          pose=self._pose(mid, side * (w / 2 + 0.015), BELT_HEIGHT),
                          color=[0.6, 0.6, 0.65, 1])
            env.add(rail)

        # legs every ~1 m
        n_legs = max(2, int(np.ceil(L / 1.0)) + 1)
        for s in np.linspace(0.1, L - 0.1, n_legs):
            for side in (+1, -1):
                leg = Cuboid([0.05, 0.05, BELT_HEIGHT - BELT_THICK],
                             pose=self._pose(s, side * (w / 2 - 0.03),
                                             (BELT_HEIGHT - BELT_THICK) / 2),
                             color=[0.35, 0.35, 0.38, 1])
                env.add(leg)

        # stripes that move with the belt, so flow direction is visible
        self.stripe_spacing = 0.25
        self.stripes = []
        for i in range(int(L / self.stripe_spacing)):
            stripe = Cuboid([0.03, w * 0.9, 0.002],
                            pose=self._pose(i * self.stripe_spacing, z=BELT_HEIGHT + 0.001),
                            color=[0.95, 0.8, 0.1, 1])
            env.add(stripe)
            self.stripes.append(stripe)

    # ---------- behaviour ----------
    def contains(self, p, z_tol=0.15):
        """True if point p (xyz) is resting on this belt's footprint."""
        rel = np.asarray(p, dtype=float) - self.start
        s = rel @ self.dir
        lat = rel @ self.lateral
        return (0 <= s <= self.length and abs(lat) <= self.width / 2
                and abs(p[2] - BELT_HEIGHT) <= z_tol)

    def progress(self, p):
        """Distance along the belt (m)."""
        return (np.asarray(p, dtype=float) - self.start) @ self.dir

    def at_end(self, p, margin=0.15):
        """True when an item has reached the end (e.g. ready for a robot to pick)."""
        return self.contains(p) and self.progress(p) >= self.length - margin

    def move_item(self, item, dt):
        T = np.array(item.T, dtype=float)
        s = self.progress(T[:3, 3])
        step = self.speed * dt
        if self.stop_at_end:
            step = min(step, max(0.0, self.length - 0.1 - s))
        lat = (T[:3, 3] - self.start) @ self.lateral
        T[:3, 3] += self.dir * step - self.lateral * lat * min(1.0, 2.0 * dt)  # guides centre the item
        item.T = T

    def animate(self, dt):
        self._stripe_offset = (self._stripe_offset + self.speed * dt) % self.stripe_spacing
        for i, stripe in enumerate(self.stripes):
            s = (i * self.stripe_spacing + self._stripe_offset) % self.length
            stripe.T = self._pose(s, z=BELT_HEIGHT + 0.001).A


class ConveyorSystem:
    """All belts in LAYOUT, with one running flag for e-stop / light-curtain stops."""

    def __init__(self, env, layout=LAYOUT, speed=DEFAULT_SPEED):
        self.conveyors = [
            Conveyor(env, name, start, end, speed=speed, stop_at_end=name in END_STOPS)
            for name, (start, end) in layout.items()
        ]
        self.by_name = {c.name: c for c in self.conveyors}
        self.running = True

    def __getitem__(self, name):
        return self.by_name[name]

    def stop(self):
        self.running = False

    def start(self):
        self.running = True

    def belt_under(self, p):
        """Downstream-most belt under point p, or None."""
        for c in reversed(self.conveyors):
            if c.contains(p):
                return c
        return None

    def step(self, dt, items=()):
        """Advance belts and any items resting on them. Call once per sim step."""
        if not self.running:
            return
        for c in self.conveyors:
            c.animate(dt)
        for item in items:
            c = self.belt_under(np.array(item.T)[:3, 3])
            if c is not None:
                c.move_item(item, dt)