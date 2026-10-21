from dataclasses import dataclass
from math import pi

import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

from conveyors import BELT_HEIGHT
from crate_model import build_milk_crate
from world import World

try:
    from ir_support import UR3e
except ImportError:                                  # model pack layout differs by version
    from ir_support_extra_robots import UR3e

DT = 0.05

# ---- layout (all metres, world frame; plastic_line ends at (-0.50, -2.60)) ----
BENCH_X = (-1.35, -0.80)        # bench x-extent; rail outer edge of plastic_line is ~-0.73
BENCH_Y = (-2.95, -2.00)
BENCH_TOP = BELT_HEIGHT         # bench top flush with belt height -> robot + crate in reach
UR3E_BASE = SE3(-0.90, -2.50, BENCH_TOP)
CRATE_XY = (-0.90, -2.15)       # crate beside the robot (~90 deg swing from the belt)
CRATE_SIZE = (0.30, 0.30, 0.22)  # milk-crate footprint and height
CRATE_COLOUR = (0.15, 0.35, 0.75, 1)
CRATE_DARK = (0.10, 0.22, 0.50, 1)

# The crate sits on a LOWER side table, not on the robot's bench: the arm only
# reaches ~0.5 m, so a lower rim means a lower, closer release point.
SIDE_TABLE_TOP = 0.42           # table-top height (belt is 0.75)
SIDE_TABLE_SIZE = 0.36
PEDESTAL_SIZE = 0.30            # the robot stands on a pedestal at belt height

GRASP_CLEARANCE = 0.02          # flange stops this far above the item top
APPROACH = 0.08                 # hover height above item top / crate rim (small: UR3e reach)
MOVE_STEPS = 28                 # joint-move interpolation steps (more = slower)
LINE_STEPS = 12                 # Cartesian lower/lift steps (more = slower)

ITEM_GAP = 0.15
SPAWN_PERIOD = 10.0             # s between test items
SPAWN_BEFORE_END = 1.0          # m before the end stop that test items appear
# Test items are only spawned when this cell runs on its own (python ur3e_cell.py).
# In main_sim.py the plastic comes from the FANUC station instead.

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
    """Slatted milk-crate on a low side table that collects the plastic."""

    def __init__(self, env, xy, colour=CRATE_COLOUR):
        self.x, self.y = xy
        sz = SIDE_TABLE_SIZE
        env.add(Cuboid([sz, sz, 0.03], pose=SE3(self.x, self.y, SIDE_TABLE_TOP - 0.015),
                       color=(0.55, 0.40, 0.25, 1)))                       # table top
        for sx in (+1, -1):                                                 # four legs
            for sy in (+1, -1):
                env.add(Cuboid([0.04, 0.04, SIDE_TABLE_TOP - 0.03],
                               pose=SE3(self.x + sx * (sz / 2 - 0.03), self.y + sy * (sz / 2 - 0.03),
                                        (SIDE_TABLE_TOP - 0.03) / 2), color=(0.35, 0.35, 0.38, 1)))
        A, W, H = CRATE_SIZE
        self.floor_z, half_a, half_w = build_milk_crate(env, self.x, self.y, SIDE_TABLE_TOP,
                                                        A, W, H, colour, CRATE_DARK)
        self.inner = min(half_a, half_w)
        self.count = 0
        self.rng = np.random.default_rng()

    def rim_z(self):
        return SIDE_TABLE_TOP + CRATE_SIZE[2]

    def drop(self, item, step):
        """Release: item falls to the crate floor at a random spot (stacked in layers)."""
        inner = max(0.0, self.inner - item.radius)
        x, y = (self.x, self.y) + self.rng.uniform(-inner, inner, 2)
        layer = (self.count // 4) % 3
        z_rest = self.floor_z + item.height / 2 + layer * 0.03
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

    def path_ok(self, q_a, q_b, n=10):
        """Joint-space path stays clear of the bench. (Elbow-up is only demanded of
        the target, so a robot starting from a flat 'zero' pose can still move.)"""
        return all(self.clear_of_bench(q) for q in rtb.jtraj(q_a, q_b, n).q)

    def solve(self, T, q_now, tries=12):
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
                if d < 1.0:             # close enough: stop searching, the shared clock waits on us
                    break
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
    def __init__(self, world: World, test_spawn=False):
        self.world = world
        self.env = world.env
        self.belt = world.conveyors["plastic_line"]

        # pedestal the robot stands on (belt height) + the low side table / crate
        bx, by = UR3E_BASE.t[0], UR3E_BASE.t[1]
        self.env.add(Cuboid([PEDESTAL_SIZE, PEDESTAL_SIZE, BENCH_TOP],
                            pose=SE3(bx, by, BENCH_TOP / 2), color=(0.45, 0.45, 0.48, 1)))
        self.crate = Crate(self.env, CRATE_XY)

        self.robot = _make_ur3e(UR3E_BASE)
        if hasattr(self.robot, "add_to_env"):
            self.robot.add_to_env(self.env)
        else:
            self.env.add(self.robot)
        self.planner = UR3ePlanner(self.robot, BENCH_TOP)

        self.items = []
        self.test_spawn = test_spawn
        self._adopted = set()                               # id(shape) of belt items already tracked
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
        self._adopt_from_belt()
        if self.test_spawn and self.t >= self._next_spawn:
            self._spawn()
        self.world.tick()

    def _carry(self, item):
        """Glue the item under the flange (item top sits at the gripper tip).
        Only the POSITION follows the tool: the tool points straight down, so
        copying its orientation would turn the item upside down."""
        offset = SE3(0, 0, item.height / 2 + GRASP_CLEARANCE)
        p = (self.robot.fkine(self.robot.q) * offset).t
        item.shape.T = SE3(p[0], p[1], p[2]).A

    def _adopt_from_belt(self):
        """Start tracking plastic another station (FANUC) has put on plastic_line.
        Size comes from the tag the spawner gave the item (world.tag_item)."""
        for shape in list(self.world.active_items):
            key = id(shape)
            if key in self._adopted or not self.belt.contains(np.array(shape.T)[:3, 3]):
                continue
            self._adopted.add(key)
            info = self.world.get_item_info(shape)
            if info is None or any(i.shape is shape for i in self.items):
                continue                                    # untagged, or one of our own test items
            self.items.append(Item(shape, info.get("kind", "plastic"),
                                   info.get("radius", 0.035), info.get("height", 0.14)))

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
    UR3eCell(World(env, dt=DT), test_spawn=True).run()