"""
Safety system for the whole recycle-bot workcell (all four robots and the belts).

GUARDING
--------
1. PERIMETER FENCE around the whole workcell, with ONE personnel gate on the
   bottom edge (in line with the walkway between plastic_line and metal_line).

2. LIGHT CURTAIN (light wall) across the gate. Breaking any beam trips the
   cell-wide e-stop (World.estop): every robot and every belt stops. Reset is
   refused while a beam is still blocked or while someone is still inside the
   fence (restart interlock); the cell only restarts on a separate Resume.
   Distance check, ISO 13855: S = K*T + C
       K = 1.6 m/s (walking approach), T = 0.25 s (curtain response + stop),
       C = 0.128 m (30 mm resolution: 8 * (30 - 14) mm)  ->  S = 0.53 m.
   The nearest hazard to the gate is the UR3e: 1.47 m from the gate to its
   base, minus its 0.50 m reach = 0.97 m > 0.53 m.

3. SPEED & SEPARATION MONITORING: an area scanner watches the floor zone in
   front of the gate. Anyone standing in it slows the WHOLE cell (every robot
   and belt) to WARN_SPEED, before they can reach the curtain.

INDICATION
----------
A stack light on a pole beside every robot (and one on the gate): GREEN while
the cell is running, RED while it is stopped (e-stop, or reset but not yet
resumed).

E-STOP STATIONS: a red mushroom e-stop button beside every robot and one
outside the gate (modelled equipment; the working e-stop is the GUI button).

A simulated person (keyboard keys on the control panel) walks up to and
through the gate to test the curtain and the speed zone.

Swift cannot recolour a shape after it is added, so lamps and beams that
change colour are drawn twice and the unused copy is parked below the floor.
"""
import time

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Sphere

# ---- perimeter fence (world frame; scene footprint is x -2.3..2.7, y -3.1..4.1) ----
FENCE_X = (-2.80, 3.30)
FENCE_Y = (-3.70, 5.00)
FENCE_H = 1.80
GATE_X = (-0.30, 0.30)          # gate in the bottom edge, in line with the FANUC walkway
GATE_Y = FENCE_Y[0]             # light curtain plane
POST_SPACING = 1.5

# ---- light curtain / area scanner ----
CURTAIN_BOTTOM, CURTAIN_TOP, BEAM_PITCH = 0.15, 1.65, 0.10
WARN_DEPTH = 0.85               # area-scanner zone depth in front of the gate
WARN_HALF_WIDTH = 0.80
WARN_SPEED = 0.25               # whole-cell speed while someone is in the zone

# ---- stack lights: (x, y) of a pole beside each robot, outside its reach ----
LIGHT_POLES = {
    "Kawasaki": (0.55, 4.45),
    "FANUC": (0.00, -0.95),
    "Yaskawa": (2.05, 2.70),
    "UR3e": (-1.55, -3.10),
    "gate": (GATE_X[1] + 0.08, GATE_Y - 0.08),
}
POLE_H = 2.10

# ---- e-stop stations: red mushroom buttons on yellow boxes, at hand height ----
# One beside each robot (next to its light pole) and one OUTSIDE the gate, so an
# operator can stop the cell without entering it.
ESTOP_STATIONS = {
    "Kawasaki": (0.80, 4.45),
    "FANUC": (0.00, -1.20),
    "Yaskawa": (1.80, 2.70),
    "UR3e": (-1.55, -3.40),
    "gate (outside)": (GATE_X[0] - 0.25, GATE_Y - 0.25),
}
ESTOP_HEIGHT = 1.05            # m, top of the button box (hand height)

# Person positions (y of the body centre); x is the middle of the gate
PERSON_STOPS = {
    "outside": GATE_Y - 1.30,
    "warning zone": GATE_Y - 0.45,
    "inside cell": GATE_Y + 0.55,
}
WALK_SPEED = 1.0               # m/s, wall-clock time (people keep walking during an e-stop)
BODY_HALF_DEPTH = 0.12         # front of the body is this far ahead of its centre

PARKED = SE3(0, 0, -10)        # where hidden duplicates wait
GREEN = (0.10, 0.85, 0.20, 1)
RED = (0.95, 0.05, 0.05, 1)
DARK = (0.15, 0.15, 0.15, 1)


def _dim(c):
    return (c[0] * 0.25, c[1] * 0.25, c[2] * 0.25, 1)


class Person:
    """Simple stand-in human: legs, torso, arms, head. Origin on the floor, faces +y."""

    def __init__(self, env, y):
        skin, shirt, pants = (0.90, 0.72, 0.58, 1), (1.0, 0.45, 0.05, 1), (0.12, 0.16, 0.35, 1)
        self.parts = [   # (shape, offset from the person's floor origin)
            (Cylinder(0.06, 0.85, color=pants), SE3(-0.07, 0, 0.425)),
            (Cylinder(0.06, 0.85, color=pants), SE3(0.07, 0, 0.425)),
            (Cuboid([0.32, 0.20, 0.60], color=shirt), SE3(0, 0, 1.15)),
            (Cylinder(0.04, 0.60, color=shirt), SE3(-0.20, 0, 1.13)),
            (Cylinder(0.04, 0.60, color=shirt), SE3(0.20, 0, 1.13)),
            (Sphere(0.11, color=skin), SE3(0, 0, 1.58)),
        ]
        self.x = (GATE_X[0] + GATE_X[1]) / 2
        self.y = y
        self.target_y = y
        self._last = time.time()
        for shape, _ in self.parts:
            env.add(shape)
        self._draw()

    @property
    def front(self):
        return self.y + BODY_HALF_DEPTH

    @property
    def back(self):
        return self.y - BODY_HALF_DEPTH

    def update(self):
        now = time.time()
        dt, self._last = min(now - self._last, 0.2), now
        dy = self.target_y - self.y
        if abs(dy) > 1e-4:
            self.y += float(np.clip(dy, -WALK_SPEED * dt, WALK_SPEED * dt))
            self._draw()

    def _draw(self):
        base = SE3(self.x, self.y, 0)
        for shape, off in self.parts:
            shape.T = (base * off).A


class SafetySystem:
    def __init__(self, world, fanuc=None):
        self.world = world
        self.fanuc = fanuc              # kept for compatibility; safety now covers the whole cell
        env = world.env
        self.curtain_blocked = False
        self.in_warning_zone = False
        self.trips = 0
        self._lamp_state = None

        self._build_fence(env)
        self._build_floor(env)
        self._build_curtain(env)
        self._build_stack_lights(env)
        self._build_estop_stations(env)
        self.person = Person(env, PERSON_STOPS["outside"])

        world.add_interlock(self._interlock)
        world.add_hook(self.update)
        self._show_beams(blocked=False)
        self._show_lamps(running=True)

    # ---------- scene ----------
    def _fence_run(self, env, a, b):
        """Straight fence from a to b (xy): posts, mesh panel, top rail."""
        a, b = np.array(a, float), np.array(b, float)
        L = float(np.linalg.norm(b - a))
        yaw = float(np.arctan2(*(b - a)[::-1]))
        n = max(1, int(np.ceil(L / POST_SPACING)))
        for s in np.linspace(0, 1, n + 1):
            p = a + s * (b - a)
            env.add(Cuboid([0.05, 0.05, FENCE_H], pose=SE3(p[0], p[1], FENCE_H / 2), color=DARK))
        mid = (a + b) / 2
        pose = SE3(mid[0], mid[1], 0) * SE3.Rz(yaw)
        env.add(Cuboid([L, 0.01, FENCE_H - 0.15], pose=pose * SE3(0, 0, 0.10 + (FENCE_H - 0.15) / 2),
                       color=(0.95, 0.80, 0.10, 0.25)))                 # wire-mesh panel
        env.add(Cuboid([L, 0.04, 0.04], pose=pose * SE3(0, 0, FENCE_H), color=(0.95, 0.80, 0.10, 1)))

    def _build_fence(self, env):
        (x0, x1), (y0, y1) = FENCE_X, FENCE_Y
        g0, g1 = GATE_X
        self._fence_run(env, (x0, y0), (g0, y0))       # bottom edge, left of the gate
        self._fence_run(env, (g1, y0), (x1, y0))       # bottom edge, right of the gate
        self._fence_run(env, (x1, y0), (x1, y1))
        self._fence_run(env, (x1, y1), (x0, y1))
        self._fence_run(env, (x0, y1), (x0, y0))

    def _build_floor(self, env):
        cx = (GATE_X[0] + GATE_X[1]) / 2
        env.add(Cuboid([2 * WARN_HALF_WIDTH, WARN_DEPTH, 0.004],
                       pose=SE3(cx, GATE_Y - WARN_DEPTH / 2, 0.002),
                       color=(1.0, 0.75, 0.0, 0.35)))                    # area-scanner zone
        x0, x1 = cx - WARN_HALF_WIDTH, cx + WARN_HALF_WIDTH
        for i, xs in enumerate(np.arange(x0, x1 - 1e-6, 0.08)):         # yellow / black tape
            env.add(Cuboid([0.08, 0.05, 0.006], pose=SE3(xs + 0.04, GATE_Y - WARN_DEPTH, 0.004),
                           color=(0.95, 0.80, 0.05, 1) if i % 2 == 0 else (0.05, 0.05, 0.05, 1)))

    def _build_curtain(self, env):
        h = CURTAIN_TOP + 0.10
        for x in GATE_X:                                  # emitter / receiver columns
            env.add(Cuboid([0.04, 0.06, h], pose=SE3(x, GATE_Y, h / 2), color=(0.10, 0.10, 0.10, 1)))
        span = GATE_X[1] - GATE_X[0] - 0.04
        cx = (GATE_X[0] + GATE_X[1]) / 2
        zs = np.arange(CURTAIN_BOTTOM, CURTAIN_TOP + 1e-6, BEAM_PITCH)
        self._beam_poses = [SE3(cx, GATE_Y, z) for z in zs]
        self._beams_ok = [Cuboid([span, 0.004, 0.004], color=(0.2, 1.0, 0.3, 0.30)) for _ in zs]
        self._beams_cut = [Cuboid([span, 0.008, 0.008], color=(1.0, 0.05, 0.05, 0.95)) for _ in zs]
        for b in self._beams_ok + self._beams_cut:
            env.add(b)

    def _build_stack_lights(self, env):
        """Pole + red (top) and green lamps. Unlit lamps are always drawn; the lit
        copies are moved into place or parked."""
        self._lamps = []                                  # (lit_shape, pose, "red"/"green")
        for x, y in LIGHT_POLES.values():
            env.add(Cylinder(0.025, POLE_H, pose=SE3(x, y, POLE_H / 2), color=DARK))
            env.add(Cylinder(0.08, 0.03, pose=SE3(x, y, 0.015), color=DARK))    # base plate
            for i, (name, c) in enumerate((("green", GREEN), ("red", RED))):
                pose = SE3(x, y, POLE_H + 0.05 + i * 0.10)
                env.add(Cylinder(0.05, 0.09, pose=pose, color=_dim(c)))
                lit = Cylinder(0.053, 0.092, color=c)
                env.add(lit)
                self._lamps.append((lit, pose, name))

    def _build_estop_stations(self, env):
        yellow = (0.95, 0.80, 0.05, 1)
        for x, y in ESTOP_STATIONS.values():
            env.add(Cuboid([0.05, 0.05, ESTOP_HEIGHT - 0.10], pose=SE3(x, y, (ESTOP_HEIGHT - 0.10) / 2),
                           color=DARK))                                          # post
            env.add(Cuboid([0.10, 0.10, 0.10], pose=SE3(x, y, ESTOP_HEIGHT - 0.05), color=yellow))
            env.add(Cylinder(0.018, 0.02, pose=SE3(x, y, ESTOP_HEIGHT + 0.01), color=DARK))   # stem
            env.add(Cylinder(0.04, 0.025, pose=SE3(x, y, ESTOP_HEIGHT + 0.03), color=RED))    # mushroom

    def _show_beams(self, blocked):
        for pose, ok, cut in zip(self._beam_poses, self._beams_ok, self._beams_cut):
            ok.T = (PARKED if blocked else pose).A
            cut.T = (pose if blocked else PARKED).A

    def _show_lamps(self, running):
        if running == self._lamp_state:
            return
        self._lamp_state = running
        on = "green" if running else "red"
        for lit, pose, name in self._lamps:
            lit.T = (pose if name == on else PARKED).A

    # ---------- GUI ----------
    def send_person(self, where):
        """Thread-safe: set where the person walks to (key of PERSON_STOPS)."""
        self.person.target_y = PERSON_STOPS[where]

    def status(self):
        return dict(curtain_blocked=self.curtain_blocked, in_warning_zone=self.in_warning_zone,
                    person_inside=self.person.back > GATE_Y, person_y=self.person.y,
                    trips=self.trips)

    # ---------- runs inside World (lock held) ----------
    def _interlock(self):
        if self.curtain_blocked:
            return "light curtain is still blocked"
        if self.person.back > GATE_Y:
            return "someone is still inside the fence"
        return ""

    def update(self):
        p = self.person
        p.update()
        blocked = p.front >= GATE_Y and p.back <= GATE_Y
        if blocked != self.curtain_blocked:
            self.curtain_blocked = blocked
            self._show_beams(blocked)
            if blocked:
                self.trips += 1
                self.world.estop("Light curtain broken at the cell gate")
        self.in_warning_zone = p.front >= GATE_Y - WARN_DEPTH and p.back <= GATE_Y
        self.world.slow_factor = WARN_SPEED if self.in_warning_zone else 1.0
        self._show_lamps(self.world.state() == "running")