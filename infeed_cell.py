"""
General-waste infeed station -- Max's station.

A pile of mixed rubbish sits in an open tub. The Kawasaki RS007L reaches in,
picks one item at a time, and places it at the start of main_line so it can
flow downstream to the next station (cross_line / FANUC).

Can run standalone for development:
    python infeed_cell.py
or be plugged into the shared workcell via main_sim.py by passing in a
shared World instead of letting it build its own.
"""
from dataclasses import dataclass
from math import pi

import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh, Sphere

from conveyors import BELT_HEIGHT, LAYOUT
from world import World
from rs007l.rs007l import RS007L
from item_meshes import BOTTLE_HEIGHT, BOTTLE_RADIUS, ensure_bottle_mesh

DT = 0.05

# ---- station layout -------------------------------------------------
BIN_XY = (-0.90, 3.60)                 # general rubbish tub, upstream of main_line start
BIN_SIZE = (0.55, 0.55, 0.30)          # open-top tub footprint + depth
BIN_FLOOR_Z = 0.10                     # tub sits on the floor, rim well below belt height

PEDESTAL_HEIGHT = 0.30                 # robot stands on a pedestal so it can reach belt height with tall items
PEDESTAL_SIZE = 0.36
KAWASAKI_BASE = SE3(-0.30, 3.90, PEDESTAL_HEIGHT)  # behind the bin, facing the belt+bin
MAX_PICK_RADIUS = 0.85                 # items are only dropped in the tub where the arm can reach them
PLACE_XY = LAYOUT["main_line"][0]      # belt start point from conveyors.py -- single source of truth

# parallel-jaw gripper on the flange (same layout as the FANUC's): a palm, then two
# fingers pointing along the tool axis. Distances are measured down the tool axis.
PALM_LEN = 0.05
FINGER_LEN = 0.05
FINGER_THICK = 0.012
FINGER_OPEN = 0.055                    # half-gap between the fingers when open
GRIP_TIME = 0.4                        # s to open / close
HOVER = 0.10                           # height above grasp/place height for approach moves
BIN_RIM_Z = BIN_FLOOR_Z + BIN_SIZE[2]  # top of the tub wall
CLEAR_RIM = 0.06                       # held item's bottom stays this far above the rim when leaving the tub
STAGE_BACK = 0.25                      # staging point this far behind the belt start (robot side: no belt there)
WAIT_Z = 0.95                          # flange height of the waiting pose over the tub
JOINT_SPEED = 0.8                      # rad/s, peak, for joint-space moves
LINE_SPEED = 0.15                      # m/s, straight-line lower / lift
TRANSIT_SPEED = 0.30                   # m/s, longer straight-line carry between tub and belt

# The four kinds of rubbish in the tub. Every item of a kind is identical.
# Each one is tagged in the World when it spawns (world.tag_item), so the
# downstream sensors (FANUC classifier, Yaskawa load cell) can read what it is.
#   shape    : which primitive/mesh to draw
#   material : "metal" | "plastic" | "organic"  (what the classifier decides on)
#   mass_kg / is_aluminium: used by the Yaskawa load cell (only meaningful for metal)
CAN_RADIUS, CAN_HEIGHT = 0.033, 0.122          # both metal cans are the SAME size
RUBBISH_ITEMS = {
    "aluminium_can": dict(shape="cyl", radius=CAN_RADIUS, height=CAN_HEIGHT, grip_radius=CAN_RADIUS,
                          colour=(0.85, 0.86, 0.90, 1), material="metal",
                          mass_kg=0.014, is_aluminium=True),
    "steel_can":     dict(shape="cyl", radius=CAN_RADIUS, height=CAN_HEIGHT, grip_radius=CAN_RADIUS,
                          colour=(0.62, 0.42, 0.18, 1), material="metal",
                          mass_kg=0.055, is_aluminium=False),
    "plastic_bottle": dict(shape="bottle", radius=BOTTLE_RADIUS, height=BOTTLE_HEIGHT, grip_radius=0.019,
                           colour=(0.45, 0.78, 0.95, 0.75), material="plastic",
                           mass_kg=0.025, is_aluminium=False),
    "apple":         dict(shape="sphere", radius=0.040, height=0.080, grip_radius=0.040,
                          colour=(0.80, 0.08, 0.08, 1), material="organic",
                          mass_kg=0.180, is_aluminium=False),
}


@dataclass
class Item:
    shape: object
    kind: str
    radius: float   # half-extent for boxes too, kept simple
    height: float
    grip_radius: float = 0.03   # half-gap the gripper fingers close to

    @property
    def pos(self):
        return np.array(self.shape.T)[:3, 3]


class GeneralBin:
    """Open-top tub holding a random pile of mixed rubbish."""

    WALL = 0.02

    def __init__(self, env, xy, max_items=5):
        self.x, self.y = xy
        self.max_items = max_items
        self.rng = np.random.default_rng()
        self._bag = []
        self._build(env)

    def _next_kind(self):
        """Shuffled bag: every 4 spawns contain one of each type, in random
        order, so the mix never runs dry of bottles or aluminium cans."""
        if not self._bag:
            self._bag = [str(k) for k in self.rng.permutation(list(RUBBISH_ITEMS))]
        return self._bag.pop()

    def _build(self, env):
        L, W, H = BIN_SIZE
        t = self.WALL
        panels = [
            ([L, W, t], (0, 0, t / 2)),
            ([t, W, H], (+(L - t) / 2, 0, H / 2)),
            ([t, W, H], (-(L - t) / 2, 0, H / 2)),
            ([L, t, H], (0, +(W - t) / 2, H / 2)),
            ([L, t, H], (0, -(W - t) / 2, H / 2)),
        ]
        for size, (dx, dy, dz) in panels:
            env.add(Cuboid(size, pose=SE3(self.x + dx, self.y + dy, BIN_FLOOR_Z + dz),
                           color=(0.5, 0.2, 0.2, 1)))

    def _random_point(self, radius, placed, margin=0.08):
        """A spot inside the tub that doesn't overlap items already in it."""
        L, W, _ = BIN_SIZE
        for _ in range(60):
            x = self.x + self.rng.uniform(-(L / 2 - margin), L / 2 - margin)
            y = self.y + self.rng.uniform(-(W / 2 - margin), W / 2 - margin)
            reachable = np.hypot(x - KAWASAKI_BASE.t[0], y - KAWASAKI_BASE.t[1]) <= MAX_PICK_RADIUS
            if reachable and all(np.hypot(x - it.pos[0], y - it.pos[1]) > radius + it.radius + 0.01
                                 for it in placed):
                break
        return x, y

    def spawn_item(self, world, placed=(), kind=None):
        kind = kind or self._next_kind()
        spec = RUBBISH_ITEMS[kind]
        r, h = spec["radius"], spec["height"]
        x, y = self._random_point(r, placed)
        pose = SE3(x, y, BIN_FLOOR_Z + self.WALL + h / 2)
        if spec["shape"] == "cyl":
            shape = Cylinder(r, h, pose=pose, color=spec["colour"])
        elif spec["shape"] == "sphere":
            shape = Sphere(r, pose=pose, color=spec["colour"])
        else:
            shape = Mesh(ensure_bottle_mesh(), pose=pose, color=spec["colour"])
        world.add(shape)          # thread-safe: other station may be stepping Swift
        world.tag_item(shape, kind=kind, material=spec["material"], radius=r, height=h,
                       mass_kg=spec["mass_kg"], is_aluminium=spec["is_aluminium"])
        return Item(shape, kind, r, h, spec["grip_radius"])


class InfeedCell:
    def __init__(self, world: World, robot=None):
        self.world = world
        self.env = world.env
        self.bin = GeneralBin(self.env, BIN_XY)

        # Low pedestal: on the floor the RS007L cannot lift a bottle up to belt height.
        px, py = KAWASAKI_BASE.t[0], KAWASAKI_BASE.t[1]
        self.env.add(Cuboid([PEDESTAL_SIZE, PEDESTAL_SIZE, PEDESTAL_HEIGHT],
                            pose=SE3(px, py, PEDESTAL_HEIGHT / 2), color=(0.35, 0.35, 0.38, 1)))
        self.robot = robot if robot is not None else RS007L(base=KAWASAKI_BASE)
        if hasattr(self.robot, "add_to_env"):
            self.robot.add_to_env(self.env)
        # Parallel-jaw gripper on the flange.
        self.palm = Cuboid([0.04, 0.11, PALM_LEN], color=(0.15, 0.15, 0.15, 1))
        self.fingers = [Cuboid([0.025, FINGER_THICK, FINGER_LEN], color=(0.75, 0.75, 0.78, 1))
                        for _ in range(2)]
        self._gap = FINGER_OPEN
        self.env.add(self.palm)
        for f in self.fingers:
            self.env.add(f)

        self.rng = np.random.default_rng(3)
        self._qlim = np.array(self.robot.qlim, dtype=float)       # shape (2, 6)
        self.items_in_bin = []
        self._held_item = None     # item currently held by the gripper, if any
        self._refill()

        self._set_q(self._home_q())
        self.q_wait = self._ik(self._down_pose(BIN_XY[0], BIN_XY[1], WAIT_Z, self._yaw_to(*BIN_XY)),
                               self.robot.q)
        self._set_q(self.q_wait)

    def _refill(self):
        while len(self.items_in_bin) < self.bin.max_items:
            self.items_in_bin.append(self.bin.spawn_item(self.world, self.items_in_bin))

    # ---------- kinematics ----------
    # RS007L has no .solve()/.move_line() helpers, so they are built here on
    # top of the RTB DHRobot it inherits through DHRobot3D.
    def _yaw_to(self, x, y):
        b = self.robot.base.t
        return float(np.arctan2(y - b[1], x - b[0]))

    def _down_pose(self, x, y, z, yaw):
        return SE3(x, y, z) * SE3.Rz(yaw) * SE3.Rx(pi)   # tool z pointing down

    def _home_q(self):
        """A sensible starting posture (arm reaching out, tool down -- the pose from
        RS007L.test) so IK never has to start from the all-zeros pose."""
        return np.array([self._yaw_to(*BIN_XY), -pi / 4, -pi / 4, 0.0, -pi / 2, 0.0])

    def _set_q(self, q):
        """Set the joints and redraw EVERYTHING for exactly this pose. DHRobot3D
        redraws its meshes *before* it stores the new q, so they lag one step
        behind; redrawing again after the store keeps arm, gripper and item together."""
        self.robot.q = q
        if hasattr(self.robot, "_update_3dmodel"):
            self.robot._update_3dmodel()
        self._update_tool()

    def _ik(self, T, q_seed):
        """IK for pose T that stays near q_seed: tries the seed first, then a few
        fixed random seeds; every result is wrapped to the equivalent angles
        nearest q_seed and must be inside the joint limits. Raises if the pose is
        out of reach instead of returning a half-converged junk answer."""
        q_seed = np.array(q_seed, dtype=float)
        best, best_d = None, np.inf
        seeds = [q_seed] + [q_seed + self.rng.normal(0, 0.5, 6) for _ in range(8)]
        for q0 in seeds:
            sol = self.robot.ikine_LM(T, q0=q0, ilimit=60, slimit=2, joint_limits=True)
            if not sol.success:
                continue
            q = q_seed + (sol.q - q_seed + pi) % (2 * pi) - pi
            if np.any(q < self._qlim[0] - 1e-6) or np.any(q > self._qlim[1] + 1e-6):
                continue
            d = float(np.abs(q - q_seed).sum())
            if d < best_d:
                best, best_d = q, d
            if d < 0.4:                       # close to where we are: take it
                break
        if best is None:
            p = np.round(T.t, 2)
            raise RuntimeError(f"Kawasaki cannot reach {p} (out of reach or joint limits)")
        return best

    # ---------- world ----------
    def _update_tool(self):
        """Gripper follows the flange; a held item is rigidly attached to it: its
        top face touches the palm, the fingers close on it, and it tilts with the tool."""
        T = self.robot.fkine(self.robot.q)
        self.palm.T = (T * SE3(0, 0, PALM_LEN / 2)).A
        for side, finger in zip((1, -1), self.fingers):
            y = side * (self._gap + FINGER_THICK / 2)
            finger.T = (T * SE3(0, y, PALM_LEN + FINGER_LEN / 2)).A
        if self._held_item is not None:
            # item's own +z (up) is the tool's -z, so flip it over after moving down the tool axis
            self._held_item.shape.T = (T * SE3(0, 0, PALM_LEN + self._held_item.height / 2)
                                       * SE3.Rx(pi)).A

    def _set_gripper(self, half_gap):
        """Open / close the fingers to `half_gap` over GRIP_TIME."""
        n = max(2, int(np.ceil(GRIP_TIME / DT)))
        for gap in np.linspace(self._gap, half_gap, n):
            self._gap = float(gap)
            self.step()

    def step(self):
        """Per-substep callback: advances the SHARED world, not a private one, so
        this station stays in sync with whatever else is on the belts. The gripper
        and any held item are re-attached EVERY step, including inside moves."""
        self._update_tool()
        self.world.tick()

    def _move_joint(self, q_goal):
        """Quintic joint move, timed so no joint exceeds JOINT_SPEED rad/s."""
        q0 = np.array(self.robot.q, dtype=float)
        n = max(12, int(np.ceil(1.875 * np.max(np.abs(q_goal - q0)) / (JOINT_SPEED * DT))))
        for q in rtb.jtraj(q0, q_goal, n).q:
            self._set_q(q)
            self.step()

    def _line(self, p_from, p_to, yaw, speed=LINE_SPEED):
        """Straight Cartesian line, tool pointing down. `yaw` is one heading, or a
        (start, end) pair to turn smoothly along the way. Each step is solved from
        the previous joint state, so the arm moves continuously (no flips or twists)."""
        p_from, p_to = np.array(p_from, dtype=float), np.array(p_to, dtype=float)
        y0, y1 = yaw if isinstance(yaw, (tuple, list)) else (yaw, yaw)
        y1 = y0 + (y1 - y0 + pi) % (2 * pi) - pi             # turn the short way round
        n = max(4, int(np.ceil(np.linalg.norm(p_to - p_from) / (speed * DT))))
        q = np.array(self.robot.q, dtype=float)
        for s in np.linspace(0, 1, n + 1)[1:]:
            p = (1 - s) * p_from + s * p_to
            q = self._ik(self._down_pose(p[0], p[1], p[2], (1 - s) * y0 + s * y1), q)
            self._set_q(q)
            self.step()

    # ---------- behaviour ----------
    def pick_and_place(self, item):
        x, y = float(item.pos[0]), float(item.pos[1])
        yaw = self._yaw_to(x, y)
        h = item.height
        top_z = BIN_FLOOR_Z + GeneralBin.WALL + h
        grasp_z = top_z + PALM_LEN                                   # palm underside touches the item top
        hover_z = grasp_z + HOVER
        travel_z = max(hover_z, BIN_RIM_Z + CLEAR_RIM + h + PALM_LEN)  # item bottom clears the tub rim

        # 1. over the item, straight down onto it
        self._move_joint(self._ik(self._down_pose(x, y, hover_z, yaw), self.robot.q))
        self._line((x, y, hover_z), (x, y, grasp_z), yaw)

        # 2. close the fingers on the item, then lift straight up out of the tub
        self._set_gripper(item.grip_radius)
        self.items_in_bin.remove(item)
        self._held_item = item
        self._line((x, y, grasp_z), (x, y, travel_z), yaw)
        self._refill()                  # keep the tub topped up (new item appears while the arm is clear)

        # 3. never swing sideways across the belt (its rail would cut the gripper / item):
        #    go to a staging point behind the belt start, where there is only open floor,
        #    slide forward over the belt start, then straight down until the item sits on it.
        px, py = PLACE_XY
        pyaw = self._yaw_to(px, py)
        place_hover = BELT_HEIGHT + h + PALM_LEN + HOVER
        place_z = BELT_HEIGHT + h + PALM_LEN
        back = np.array(PLACE_XY) + STAGE_BACK * (np.array(KAWASAKI_BASE.t[:2]) - np.array(PLACE_XY)) \
            / np.linalg.norm(np.array(KAWASAKI_BASE.t[:2]) - np.array(PLACE_XY))
        stage = (back[0], back[1], place_hover)
        self._line((x, y, travel_z), stage, (yaw, pyaw), speed=TRANSIT_SPEED)
        self._line(stage, (px, py, place_hover), pyaw)
        self._line((px, py, place_hover), (px, py, place_z), pyaw)

        # 4. open the fingers: item rests upright on the belt and the belt takes it
        self._held_item = None
        item.shape.T = SE3(px, py, BELT_HEIGHT + h / 2).A
        self.world.add_active(item.shape)
        self._set_gripper(FINGER_OPEN)
        self._line((px, py, place_z), (px, py, place_hover), pyaw)
        self._line((px, py, place_hover), stage, pyaw)      # back out the way we came in
        self._move_joint(self.q_wait)

    def _manual_pick(self):
        """Manual feed: pick the next requested kind from the tub. If none of that
        kind is in the tub, one is added to it first."""
        requests = getattr(self.world, "pick_requests", [])
        if not requests:
            self.world.infeed_status = "Manual: waiting for you to choose an item"
            self.step()
            return
        kind = requests.pop(0)
        item = next((it for it in self.items_in_bin if it.kind == kind), None)
        if item is None:
            item = self.bin.spawn_item(self.world, self.items_in_bin, kind=kind)
            self.items_in_bin.append(item)
        self.world.infeed_status = f"Kawasaki picking {kind}"
        self.pick_and_place(item)

    def run(self):
        self._move_joint(self.q_wait)
        while True:
            if not getattr(self.world, "auto_feed", True):   # manual feed: only pick what the GUI asks for
                self._manual_pick()
                continue
            if not self.items_in_bin:
                self._refill()
            self.pick_and_place(self.items_in_bin[0])


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    InfeedCell(World(env, dt=DT)).run()