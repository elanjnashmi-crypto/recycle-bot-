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
from spatialgeometry import Cuboid, Cylinder

from conveyors import BELT_HEIGHT, LAYOUT
from world import World
from rs007l.rs007l import RS007L

DT = 0.05

# ---- station layout -------------------------------------------------
BIN_XY = (-0.90, 3.60)                 # general rubbish tub, upstream of main_line start
BIN_SIZE = (0.55, 0.55, 0.30)          # open-top tub footprint + depth
BIN_FLOOR_Z = 0.10                     # tub sits on the floor, rim well below belt height

KAWASAKI_BASE = SE3(-0.30, 3.90, 0.0)  # floor-mounted, behind the bin, facing the belt+bin
PLACE_XY = LAYOUT["main_line"][0]      # belt start point from conveyors.py -- single source of truth

APPROACH = 0.18
GRASP_CLEARANCE = 0.02                 # how far above an item's top the tool stops

# name: (mesh, radius/half-extent, height, colour)
RUBBISH_ITEMS = {
    "plastic_bottle": ("cyl", 0.035, 0.20, (0.20, 0.70, 0.90, 1)),
    "cardboard_box":  ("box", 0.10,  0.10, (0.75, 0.55, 0.30, 1)),
    "glass_jar":      ("cyl", 0.04,  0.12, (0.30, 0.80, 0.40, 0.6)),
    "drink_can":      ("cyl", 0.033, 0.12, (0.80, 0.82, 0.88, 1)),
}


@dataclass
class Item:
    shape: object
    kind: str
    radius: float   # half-extent for boxes too, kept simple
    height: float

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
        self._build(env)

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

    def _random_point(self, margin=0.08):
        L, W, _ = BIN_SIZE
        x = self.x + self.rng.uniform(-(L / 2 - margin), L / 2 - margin)
        y = self.y + self.rng.uniform(-(W / 2 - margin), W / 2 - margin)
        return x, y

    def spawn_item(self, env):
        kind = self.rng.choice(list(RUBBISH_ITEMS))
        mesh, r, h, colour = RUBBISH_ITEMS[kind]
        x, y = self._random_point()
        z = BIN_FLOOR_Z + self.WALL + h / 2
        if mesh == "cyl":
            shape = Cylinder(r, h, pose=SE3(x, y, z), color=colour)
        else:
            shape = Cuboid([r * 2, r * 2, h], pose=SE3(x, y, z), color=colour)
        env.add(shape)
        return Item(shape, kind, r, h)


class InfeedCell:
    def __init__(self, world: World):
        self.world = world
        self.env = world.env
        self.bin = GeneralBin(self.env, BIN_XY)
        self.robot = RS007L(base=KAWASAKI_BASE)
        self.robot.add_to_env(self.env)

        self.items_in_bin = []
        self._refill()

        self.q_wait = self._solve_above(KAWASAKI_BASE.t[0], self.bin.y, 0.9, yaw=-pi / 2)

    def _refill(self):
        while len(self.items_in_bin) < self.bin.max_items:
            self.items_in_bin.append(self.bin.spawn_item(self.env))

    # ---------- kinematics (RS007L has no .solve()/.move_line() helpers of
    # its own, so these are built directly on top of the RTB DHRobot it
    # inherits through DHRobot3D) ----------
    def _down_pose(self, x, y, z, yaw):
        return SE3(x, y, z) * SE3.Rz(yaw) * SE3.Rx(pi)   # tool z pointing down

    def _solve_above(self, x, y, z, yaw=0.0):
        T = self._down_pose(x, y, z, yaw)
        sol = self.robot.ikine_LM(T, q0=self.robot.q)
        if not sol.success:
            raise RuntimeError(f"IK failed for target ({x:.2f}, {y:.2f}, {z:.2f})")
        return sol.q

    # ---------- world ----------
    def step(self):
        """Per-substep callback: advances the SHARED world, not a private
        one, so this station stays in sync with whatever else is on the
        belts (e.g. the Yaskawa sorting cell)."""
        self.world.tick()

    def _move_joint(self, q_goal, steps=60):
        for q in rtb.jtraj(self.robot.q, q_goal, steps).q:
            self.robot.q = q
            self.step()

    def _move_cartesian(self, T_goal, steps=40):
        """Simple resolved-motion step toward a Cartesian target: each
        sub-step re-solves IK for the fixed goal pose and interpolates a
        little closer in joint space. Not true RMRC (no Jacobian-based
        velocity control), but gives a smooth Cartesian-looking approach
        for the short grasp/place moves this station needs."""
        for _ in range(steps):
            sol = self.robot.ikine_LM(T_goal, q0=self.robot.q, ilimit=20)
            self.robot.q = rtb.jtraj(self.robot.q, sol.q, 2).q[-1]
            self.step()

    def _carry(self, item):
        """Keep a picked-up item glued to the tool flange each sim step."""
        tool_offset = SE3(0, 0, item.height / 2 + GRASP_CLEARANCE)
        item.shape.T = (self.robot.fkine(self.robot.q) * tool_offset).A

    # ---------- behaviour ----------
    def pick_and_place(self, item):
        x, y = item.pos[0], item.pos[1]
        yaw = np.arctan2(y - self.robot.base.t[1], x - self.robot.base.t[0])
        top_z = BIN_FLOOR_Z + GeneralBin.WALL + item.height
        grasp_z = top_z - item.height / 2 + GRASP_CLEARANCE

        self._move_joint(self._solve_above(x, y, top_z + APPROACH, yaw))
        self._move_cartesian(self._down_pose(x, y, grasp_z, yaw))

        self.items_in_bin.remove(item)
        for _ in range(20):               # lift with item attached
            self._carry(item)
            self.step()

        px, py = PLACE_XY
        place_yaw = np.arctan2(py - self.robot.base.t[1], px - self.robot.base.t[0])
        above_belt = self._down_pose(px, py, BELT_HEIGHT + item.height + APPROACH, place_yaw)
        place = self._down_pose(px, py, BELT_HEIGHT + item.height / 2 + GRASP_CLEARANCE, place_yaw)

        self._move_joint(self.robot.ikine_LM(above_belt, q0=self.robot.q).q)
        for _ in range(10):
            self._carry(item)
            self.step()

        self._move_joint(self.robot.ikine_LM(place, q0=self.robot.q).q)

        item.shape.T = SE3(px, py, BELT_HEIGHT + item.height / 2).A   # release onto belt
        self.world.add_active(item.shape)      # now riding main_line, world-driven
        self._move_joint(self.q_wait)

    def run(self):
        self._move_joint(self.q_wait)
        while True:
            if not self.items_in_bin:
                self._refill()
            self.pick_and_place(self.items_in_bin[0])


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    InfeedCell(World(env, dt=DT)).run()