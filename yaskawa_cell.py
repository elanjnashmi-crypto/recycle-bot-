from dataclasses import dataclass

import numpy as np
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

from conveyors import BELT_HEIGHT
from world import World
from yaskawa_gp7.yaskawa import GP7, FINGER_OPEN, FINGER_LEN

DT = 0.05

PEDESTAL_HEIGHT = 0.45                  # floor-mounted, the GP7 can't reach down onto a 0.75 m belt
YASKAWA_BASE = SE3(1.90, 1.70, PEDESTAL_HEIGHT)   # 0.6 m left of the metal_return end stop
AL_BIN_XY = (1.30, 1.30)                # down-left of the robot (see sketch)
NON_AL_BIN_XY = (1.30, 2.00)            # left of the robot
BIN_SIZE = (0.40, 0.40, 0.35)           # open-top deposit box, sitting on the floor

AL_THRESHOLD_KG = 0.030                 # lighter than this -> aluminium
LOAD_CELL_NOISE_KG = 0.002

APPROACH = 0.15                         # clearance above items / bins for approach moves
ITEM_GAP = 0.15                         # items queue this far apart behind the end stop

# Used only if an item arrives without a tag (should not happen in the full sim).
DEFAULT_RADIUS, DEFAULT_HEIGHT, DEFAULT_MASS_KG = 0.033, 0.122, 0.055


@dataclass
class Item:
    shape: Cylinder
    kind: str
    radius: float
    height: float
    mass_kg: float
    is_aluminium: bool

    @property
    def pos(self):
        return np.array(self.shape.T)[:3, 3]


class DepositBin:
    """Open-top box built from five panels."""

    WALL = 0.02

    def __init__(self, env, xy, colour):
        self.x, self.y = xy
        L, W, H = BIN_SIZE
        t = self.WALL
        panels = [
            ([L, W, t], (0, 0, t / 2)),                 # floor
            ([t, W, H], (+(L - t) / 2, 0, H / 2)),
            ([t, W, H], (-(L - t) / 2, 0, H / 2)),
            ([L, t, H], (0, +(W - t) / 2, H / 2)),
            ([L, t, H], (0, -(W - t) / 2, H / 2)),
        ]
        for size, (dx, dy, dz) in panels:
            env.add(Cuboid(size, pose=SE3(self.x + dx, self.y + dy, dz), color=colour))
        self.count = 0
        self.rng = np.random.default_rng()

    def drop_point(self):
        return self.x, self.y, BIN_SIZE[2] + APPROACH

    def drop(self, item, step):
        """Let a released item fall to a random spot on the bin floor."""
        inner = BIN_SIZE[0] / 2 - self.WALL - item.radius
        x, y = (self.x, self.y) + self.rng.uniform(-inner, inner, 2)
        z_rest = self.WALL + item.height / 2
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


class LoadCell:
    """Wrist load cell: reads the payload mass with a little noise."""

    def __init__(self, noise_kg=LOAD_CELL_NOISE_KG):
        self.noise = noise_kg
        self.rng = np.random.default_rng()

    def read(self, item):
        return item.mass_kg + self.rng.normal(0, self.noise)


class SortingCell:
    def __init__(self, world: World):
        self.world = world
        self.env = world.env
        self.belt = world.conveyors["metal_return"]

        b = YASKAWA_BASE.t
        self.env.add(Cuboid([0.35, 0.35, PEDESTAL_HEIGHT], pose=SE3(b[0], b[1], PEDESTAL_HEIGHT / 2),
                            color=(0.35, 0.35, 0.38, 1)))
        self.robot = GP7(base=YASKAWA_BASE)
        self.robot.add_to_env(self.env)
        self.bins = {
            True: DepositBin(self.env, AL_BIN_XY, colour=(0.20, 0.45, 0.80, 1)),
            False: DepositBin(self.env, NON_AL_BIN_XY, colour=(0.85, 0.45, 0.10, 1)),
        }
        self.load_cell = LoadCell()

        self.t = 0.0
        self._warned_untagged = False

        # Waiting pose: gripper above the end stop. Every other pose is solved
        # from a seed facing the target, so the arm stays elbow-up throughout.
        pick = self.belt.end - self.belt.dir * 0.1
        self.q_wait = self._solve_above(pick[0], pick[1], BELT_HEIGHT + 0.17 + APPROACH)
        self.q_bins = {al: self._solve_above(*b.drop_point()) for al, b in self.bins.items()}

    # ---------- world ----------
    def step(self):
        """Per-substep callback handed to GP7's move_* helpers. Advances the
        SHARED world clock (conveyors + render) rather than owning its own,
        so this station stays in sync with whatever else is on the belts."""
        self.t += DT
        self.world.tick()

    def _as_item(self, shape):
        """Wrap a belt shape in an Item using the tag the infeed gave it."""
        info = self.world.get_item_info(shape)
        if info is None:
            if not self._warned_untagged:
                print("[yaskawa] item with no tag on metal_return -- using default size/mass")
                self._warned_untagged = True
            info = {}
        return Item(shape, info.get("kind", "untagged"),
                    info.get("radius", DEFAULT_RADIUS), info.get("height", DEFAULT_HEIGHT),
                    info.get("mass_kg", DEFAULT_MASS_KG), bool(info.get("is_aluminium", False)))

    def _item_at_pick(self):
        """First item that has reached the end stop of metal_return."""
        stop = self.belt.length - 0.1
        for shape in list(self.world.active_items):
            p = np.array(shape.T)[:3, 3]
            if self.belt.contains(p) and self.belt.progress(p) >= stop - 0.005:
                return self._as_item(shape)
        return None

    # ---------- robot ----------
    def _solve_above(self, x, y, z):
        b = self.robot.base.t
        yaw = np.arctan2(y - b[1], x - b[0])
        seed = [yaw, 0.3, 0.2, 0, -np.pi / 2 - 0.5, 0]
        return self.robot.solve(GP7.down_pose(x, y, z, yaw), seed)

    def pick_and_sort(self, item):
        r = self.robot
        x, y, _ = item.pos
        yaw = np.arctan2(y - r.base.t[1], x - r.base.t[0])
        top = BELT_HEIGHT + item.height
        grasp_z = top - (FINGER_LEN - 0.02) - 0.01   # palm stops 1 cm above the item
        grasp = GP7.down_pose(x, y, grasp_z, yaw)
        above = GP7.down_pose(x, y, top + APPROACH, yaw)

        r.move_line(above, self.step, DT)
        r.move_line(grasp, self.step, DT, speed=0.125)
        r.set_gripper(item.radius, self.step, DT)
        r.grasp(item.shape)
        self.world.remove_active(item.shape)   # now held by the gripper, not belt-driven
        r.move_line(above, self.step, DT)

        mass = self.load_cell.read(item)
        is_al = mass < AL_THRESHOLD_KG
        print(f"[{self.t:6.1f}s] {item.kind:18s} {mass * 1000:5.1f} g -> "
              f"{'Al' if is_al else 'Non-Al'}{'' if is_al == item.is_aluminium else '  (MIS-SORTED)'}")

        r.move_joint(self.q_bins[is_al], self.step, DT)
        r.set_gripper(FINGER_OPEN, self.step, DT, duration=0.16)
        r.release()
        self.bins[is_al].drop(item, self.step)
        r.move_joint(self.q_wait, self.step, DT)

    def run(self):
        self.robot.move_joint(self.q_wait, self.step, DT)
        while True:
            item = self._item_at_pick()
            if item is None:
                self.step()
            else:
                self.pick_and_sort(item)


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    SortingCell(World(env, dt=DT)).run()