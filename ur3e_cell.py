"""
UR3e plastic-collection cell at the end of plastic_line.

The belt carries plastic items to its end stop. The UR3e (mounted on a small
bench beside the belt end) picks each one and places it in a plastic crate
that sits on the same bench. This is the part of the workcell that is
reproduced on the REAL UR3e: same pick-from-belt-end / place-in-crate motion,
so the real-robot evidence maps directly onto this simulated station.

Motion:
  * big moves (wait -> hover -> crate) are joint-space moves to an IK solution
    chosen to be (a) elbow-up, (b) nearest to the current pose and (c) clear of
    the bench along the whole joint path (see UR3ePlanner);
  * the final lower / lift over the belt are straight-line Cartesian moves
    solved step by step from the previous joint state (resolved-rate style),
    so the tool goes straight down onto the item instead of arcing.

Run standalone for development:
    python ur3e_plastic_cell.py
or as part of the shared workcell via main_sim.py (shares one World).

REACH NOTE: a UR3e only reaches ~0.5 m, so it is mounted on a bench at belt
height ~0.4 m from the pick point. If a target cannot be reached the error
says which one -- move UR3E_BASE closer or use shorter items.
"""
from dataclasses import dataclass
from math import pi

import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

from conveyors import BELT_HEIGHT
from world import World

try:
    from ir_support import UR3e
except ImportError:                                  # model pack layout differs by version
    from ir_support_extra_robots import UR3e

DT = 0.05

# ---- layout (all metres, world frame; plastic_line ends at (-0.90, -2.60)) ----
BENCH_X = (-1.75, -1.20)        # bench x-extent; rail outer edge of plastic_line is ~-1.13
BENCH_Y = (-2.95, -2.00)
BENCH_TOP = BELT_HEIGHT         # bench top flush with belt height -> robot + crate in reach
UR3E_BASE = SE3(-1.30, -2.50, BENCH_TOP)
CRATE_XY = (-1.30, -2.15)       # crate beside the robot (~90 deg swing from the belt)
CRATE_SIZE = (0.22, 0.22, 0.12)

GRASP_CLEARANCE = 0.02          # flange stops this far above the item top
APPROACH = 0.08                 # hover height above item top / crate rim (small: UR3e reach)
MOVE_STEPS = 50                 # joint-move interpolation steps (more = slower)
LINE_STEPS = 24                 # Cartesian lower/lift steps (more = slower)

ITEM_GAP = 0.15
SPAWN_PERIOD = 10.0             # s between test items
SPAWN_BEFORE_END = 1.0          # m before the end stop that test items appear
TEST_SPAWN = True               # set False once the FANUC station feeds plastic_line

# name: (radius, height, colour)  -- kept short so the UR3e can reach the top
PLASTIC_ITEMS = {
    "water_bottle": (0.032, 0.14, (0.20, 0.70, 0.90, 1)),
    "yoghurt_tub":  (0.045, 0.08, (0.95, 0.95, 0.95, 1)),
    "detergent":    (0.040, 0.13, (0.90, 0.40, 0.70, 1)),
}


@dataclass
class Item:
    shape: object
    kind: str
    radius: float
    height: float

    @property
    def pos(self):
        return np.array(self.shape.T)[:3, 3]


class Crate:
    """Open-top crate on the bench that collects the plastic."""

    WALL = 0.01

    def __init__(self, env, xy, colour=(0.15, 0.35, 0.75, 1)):
        self.x, self.y = xy
        L, W, H = CRATE_SIZE
        t = self.WALL
        panels = [
            ([L, W, t], (0, 0, t / 2)),
            ([t, W, H], (+(L - t) / 2, 0, H / 2)),
            ([t, W, H], (-(L - t) / 2, 0, H / 2)),
            ([L, t, H], (0, +(W - t) / 2, H / 2)),
            ([L, t, H], (0, -(W - t) / 2, H / 2)),
        ]
        for size, (dx, dy, dz) in panels:
            env.add(Cuboid(size, pose=SE3(self.x + dx, self.y + dy, BENCH_TOP + dz), color=colour))
        self.count = 0
        self.rng = np.random.default_rng()

    def rim_z(self):
        return BENCH_TOP + CRATE_SIZE[2]

    def drop(self, item, step):
        """Release: item falls to the crate floor at a random spot (stacked in layers)."""
        inner = max(0.0, CRATE_SIZE[0] / 2 - self.WALL - item.radius)
        x, y = (self.x, self.y) + self.rng.uniform(-inner, inner, 2)
        layer = (self.count // 4) % 3
        z_rest = BENCH_TOP + self.WALL + item.height / 2 + layer * 0.03
        T = np.array(item.shape.T)
        z, v = T[2, 3], 0.0
        while z > z_rest:
            v += 9.81 * DT
            z = max(z_rest, z - v * DT)
            T[:3, 3] = (x, y, z)
            T[:3, :3] = np.eye(3)
            item.shape.T = T
            step()
        self.count += 1


class UR3ePlanner:
    """IK helper for a downward-pointing tool that keeps the arm elbow-up and
    above the bench. Works with any 6-joint DH robot (uses only ikine_LM and
    fkine_all), so it does not depend on the UTS model's joint conventions.

    Plain per-target IK was not enough: consecutive targets landed in different
    arm configurations (base angle flipping by ~150 deg, elbow near bench
    height), and interpolating between those sweeps the arm through the bench.
    """

    ELBOW_RISE = 0.03      # elbow must be at least this far above the shoulder
    MIN_CLEAR = 0.03       # every joint (except base) at least this far above the bench

    def __init__(self, robot, bench_top):
        self.robot = robot
        self.bench_top = bench_top
        self.rng = np.random.default_rng(1)          # fixed seed: repeatable behaviour

    def heights(self, q):
        fk = self.robot.fkine_all(q)
        z = [float(f.t[2]) for f in fk]
        if len(z) == len(q):                          # some toolbox versions omit the base frame
            z = [float(self.robot.base.t[2])] + z
        return z

    def clear_of_bench(self, q):
        return min(self.heights(q)[1:]) >= self.bench_top + self.MIN_CLEAR

    def pose_ok(self, q):
        """A usable target pose: clear of the bench AND elbow-up."""
        z = self.heights(q)
        return min(z[1:]) >= self.bench_top + self.MIN_CLEAR and z[2] >= z[1] + self.ELBOW_RISE

    def path_ok(self, q_a, q_b, n=20):
        """Joint-space path stays clear of the bench. (Elbow-up is only demanded of
        the target, so a robot starting from a flat 'zero' pose can still move.)"""
        return all(self.clear_of_bench(q) for q in rtb.jtraj(q_a, q_b, n).q)

    def solve(self, T, q_now, tries=40):
        """Valid IK solution nearest to q_now whose joint-space path from q_now
        stays clear of the bench; None if there is none."""
        q_now = np.array(q_now, dtype=float)
        yaw = np.arctan2(T.t[1] - self.robot.base.t[1], T.t[0] - self.robot.base.t[0])
        seeds = [q_now] + [np.array([yaw, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0.0])
                           + self.rng.normal(0, 0.9, 6) for _ in range(tries)]
        best, best_d = None, np.inf
        for q0 in seeds:
            sol = self.robot.ikine_LM(T, q0=q0, ilimit=40, slimit=1)
            if not sol.success:
                continue
            q = q_now + (sol.q - q_now + pi) % (2 * pi) - pi    # take the short way round each joint
            d = float(np.abs(q - q_now).sum())
            if d < best_d and self.pose_ok(q) and self.path_ok(q_now, q):
                best, best_d = q, d
        return best

    def line(self, xyz_from, xyz_to, yaw, q_now, steps):
        """Straight Cartesian line (tool down). Each step is solved from the
        previous joint state, so the motion is continuous. None if any step fails."""
        out, q = [], np.array(q_now, dtype=float)
        for s in np.linspace(0, 1, steps + 1)[1:]:
            p = (1 - s) * np.array(xyz_from) + s * np.array(xyz_to)
            sol = self.robot.ikine_LM(SE3(*p) * SE3.Rz(yaw) * SE3.Rx(pi), q0=q, ilimit=40, slimit=1)
            if not sol.success:
                return None
            q = q + (sol.q - q + pi) % (2 * pi) - pi
            if not self.pose_ok(q):
                return None
            out.append(q.copy())
        return out


def _make_ur3e(base):
    """Build the UTS UR3e; tolerate either constructor style."""
    try:
        return UR3e(base=base)
    except TypeError:
        robot = UR3e()
        robot.base = base
        return robot


class UR3eCell:
    def __init__(self, world: World):
        self.world = world
        self.env = world.env
        self.belt = world.conveyors["plastic_line"]

        # bench the robot and crate sit on
        bx, by = (BENCH_X[0] + BENCH_X[1]) / 2, (BENCH_Y[0] + BENCH_Y[1]) / 2
        self.env.add(Cuboid([BENCH_X[1] - BENCH_X[0], BENCH_Y[1] - BENCH_Y[0], BENCH_TOP],
                            pose=SE3(bx, by, BENCH_TOP / 2), color=(0.45, 0.45, 0.48, 1)))
        self.crate = Crate(self.env, CRATE_XY)

        self.robot = _make_ur3e(UR3E_BASE)
        if hasattr(self.robot, "add_to_env"):
            self.robot.add_to_env(self.env)
        else:
            self.env.add(self.robot)
        self.planner = UR3ePlanner(self.robot, BENCH_TOP)

        self.items = []
        self._held_item = None
        self.t = 0.0
        self._next_spawn = 0.0
        self.rng = np.random.default_rng()

        pick = self.belt.end - self.belt.dir * 0.1          # where items stop
        self._pick_xy = (float(pick[0]), float(pick[1]))
        self.q_wait = self._plan(self._pick_xy[0], self._pick_xy[1],
                                 BELT_HEIGHT + 0.14 + APPROACH + GRASP_CLEARANCE)

    # ---------- kinematics ----------
    def _yaw_to(self, x, y):
        b = self.robot.base.t
        return float(np.arctan2(y - b[1], x - b[0]))

    def _plan(self, x, y, z):
        """Joint target for a downward tool at (x, y, z), elbow-up and bench-safe."""
        T = SE3(x, y, z) * SE3.Rz(self._yaw_to(x, y)) * SE3.Rx(pi)
        q = self.planner.solve(T, self.robot.q)
        if q is None:
            raise RuntimeError(
                f"UR3e has no safe (elbow-up, above-bench) solution for ({x:.2f}, {y:.2f}, {z:.2f}) "
                f"from base {np.round(self.robot.base.t, 2)} -- move UR3E_BASE closer or lower the target")
        return q

    # ---------- world ----------
    def step(self):
        """Per-substep callback: carry the held item, spawn test items, advance the shared clock."""
        self.t += DT
        if self._held_item is not None:
            self._carry(self._held_item)
        if TEST_SPAWN and self.t >= self._next_spawn:
            self._spawn()
        self.world.tick()

    def _carry(self, item):
        """Glue the item under the flange (item top sits at the gripper tip)."""
        offset = SE3(0, 0, item.height / 2 + GRASP_CLEARANCE)
        item.shape.T = (self.robot.fkine(self.robot.q) * offset).A

    def _spawn(self):
        s = self.belt.length - SPAWN_BEFORE_END
        if any(self.belt.contains(i.pos) and abs(self.belt.progress(i.pos) - s) < ITEM_GAP
               for i in self.items):
            return
        kind = self.rng.choice(list(PLASTIC_ITEMS))
        r, h, colour = PLASTIC_ITEMS[kind]
        p = self.belt.start + self.belt.dir * s
        shape = Cylinder(r, h, pose=SE3(p[0], p[1], BELT_HEIGHT + h / 2), color=colour)
        self.world.add(shape)                              # thread-safe Swift add
        self.items.append(Item(shape, kind, r, h))
        self.world.add_active(shape)
        self._next_spawn = self.t + SPAWN_PERIOD

    def _item_at_pick(self):
        stop = self.belt.length - 0.1
        for item in self.items:
            if (self._held_item is not item and self.belt.contains(item.pos)
                    and self.belt.progress(item.pos) >= stop - 0.005):
                return item
        return None

    # ---------- motion ----------
    def _move_joint(self, q_goal, steps=MOVE_STEPS):
        for q in rtb.jtraj(self.robot.q, q_goal, steps).q:
            self.robot.q = q
            self.step()

    def _move_to(self, x, y, z, steps=MOVE_STEPS):
        self._move_joint(self._plan(x, y, z), steps)

    def _move_line(self, xyz_from, xyz_to, steps=LINE_STEPS):
        """Straight-line Cartesian move: the tool goes straight down / up."""
        yaw = self._yaw_to(xyz_to[0], xyz_to[1])
        path = self.planner.line(xyz_from, xyz_to, yaw, self.robot.q, steps)
        if path is None:
            raise RuntimeError(f"UR3e straight-line move {np.round(xyz_from, 2)} -> "
                               f"{np.round(xyz_to, 2)} left the reachable / bench-safe region")
        for q in path:
            self.robot.q = q
            self.step()

    # ---------- behaviour ----------
    def pick_and_store(self, item):
        x, y, _ = item.pos
        top = BELT_HEIGHT + item.height
        hover = (x, y, top + APPROACH)
        grasp = (x, y, top + GRASP_CLEARANCE)

        self._move_to(*hover)
        self._move_line(hover, grasp)                      # straight down onto the item

        self.items.remove(item)
        self.world.remove_active(item.shape)               # belt stops dragging it
        self._held_item = item                             # gripper closes

        self._move_line(grasp, hover)                      # straight back up

        cx, cy = CRATE_XY
        self._move_to(cx, cy, self.crate.rim_z() + item.height + APPROACH)

        self._held_item = None                             # gripper opens
        self.crate.drop(item, self.step)
        print(f"[{self.t:6.1f}s] UR3e stored {item.kind} (crate: {self.crate.count})")
        self._move_joint(self.q_wait)

    def run(self):
        self._move_joint(self.q_wait)
        while True:
            item = self._item_at_pick()
            if item is None:
                self.step()
            else:
                self.pick_and_store(item)


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    UR3eCell(World(env, dt=DT)).run()