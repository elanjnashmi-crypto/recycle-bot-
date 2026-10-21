import sys
import types

# macOS: the 'keyboard' package (imported by ir_support) crashes Python on
# import. This project never uses it, so give it a harmless stand-in.
if sys.platform == "darwin" and "keyboard" not in sys.modules:
    class _NoKeyboard(types.ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: False
    sys.modules["keyboard"] = _NoKeyboard("keyboard")

from math import pi

import numpy as np
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

from conveyors import BELT_HEIGHT
from world import World
from fanuc_lrmate200id.lrmate200id import LRMate200iD, FINGER_OPEN, FINGER_LEN, REACH

DT = 0.05

# ---- station layout (world frame, metres) ----------------------------------
# Stand between plastic_line and metal_line, just below the cross line (see
# sketch). conveyors.py puts the world origin at the FANUC base.
PEDESTAL_HEIGHT = 0.60      # best IK margin over all pick/place points (checked 0.45-0.75 m)
FANUC_BASE = SE3(0.0, 0.0, PEDESTAL_HEIGHT)
PEDESTAL_SIZE = 0.30

APPROACH = 0.12             # clearance above the tallest item for the waiting pose
TALLEST_ITEM = 0.20         # tallest item the infeed produces (plastic_bottle)
PICK_WINDOW_X = -0.18       # gripper waits above this x on the cross line
TRIGGER_AHEAD = 0.02        # start the tracking descent this far before the item is under the gripper
LAST_CHANCE_X = -0.24       # item already past this x when the robot is free -> let it go
DESCEND_SPEED = 0.30        # m/s, tracking descent onto a moving item
PLACE_PROGRESS = 0.15       # m along plastic/metal line from its start where items are placed
GRASP_BELOW_TOP = FINGER_LEN - 0.01   # TCP stops this far below the item's top

# Fallback only: the infeed now tags every item with its "material"
# (infeed_cell.RUBBISH_ITEMS), which the camera uses first. Anything not
# listed here and not tagged -> "general".
MATERIAL_OF = {
    "plastic_bottle": "plastic",
    "drink_can": "metal",
    "cardboard_box": "general",
    "glass_jar": "general",
}
TARGET_LINE = {"plastic": "plastic_line", "metal": "metal_line"}

# Standalone test items: name -> (shape, radius / half-extent, height, colour)
TEST_ITEMS = {
    "plastic_bottle": ("cyl", 0.035, 0.20, (0.20, 0.70, 0.90, 1)),
    "cardboard_box":  ("box", 0.10,  0.10, (0.75, 0.55, 0.30, 1)),
    "glass_jar":      ("cyl", 0.04,  0.12, (0.30, 0.80, 0.40, 0.6)),
    "drink_can":      ("cyl", 0.033, 0.12, (0.80, 0.82, 0.88, 1)),
}


class OverheadCamera:
    """Simulated vision sensor looking down at a stretch of the cross line.
    Reports the material of every item inside its field of view. (In the sim
    the 'image recognition' is the kind the spawner tagged the item with.)"""

    def __init__(self, world, belt, s_from, s_to):
        self.world = world
        self.belt = belt
        self.s_from, self.s_to = s_from, s_to
        self.seen = {}                      # id(shape) -> material (each item classified once)

    def scan(self):
        found = []
        for shape in list(self.world.active_items):
            p = np.array(shape.T)[:3, 3]
            if not self.belt.contains(p) or not (self.s_from <= self.belt.progress(p) <= self.s_to):
                continue
            key = id(shape)
            if key not in self.seen:
                info = self.world.get_item_info(shape) or {}
                # Prefer the material the infeed tagged the item with; fall back to the
                # kind lookup for items spawned by older test code. Anything that isn't
                # plastic or metal (organic, cardboard, glass...) is left as "general".
                material = info.get("material") or MATERIAL_OF.get(info.get("kind"), "general")
                self.seen[key] = material if material in TARGET_LINE else "general"
            found.append((shape, self.seen[key]))
        return found


class ClassifyCell:
    def __init__(self, world: World, test_spawn=False):
        self.world = world
        self.env = world.env
        self.cross = world.conveyors["cross_line"]
        self.lines = {m: world.conveyors[name] for m, name in TARGET_LINE.items()}

        b = FANUC_BASE.t
        self.env.add(Cuboid([PEDESTAL_SIZE, PEDESTAL_SIZE, PEDESTAL_HEIGHT],
                            pose=SE3(b[0], b[1], PEDESTAL_HEIGHT / 2), color=(0.35, 0.35, 0.38, 1)))
        self.robot = LRMate200iD(base=FANUC_BASE)
        self.robot.add_to_env(self.env)

        # Camera covers the cross line from where main_line drops items to just past the pick window
        y_belt = self.cross.start[1]
        s_drop = self.cross.progress(np.array([world.conveyors["main_line"].end[0], y_belt, BELT_HEIGHT]))
        s_last = self.cross.progress(np.array([LAST_CHANCE_X - 0.10, y_belt, BELT_HEIGHT]))
        self.camera = OverheadCamera(world, self.cross, s_drop - 0.25, s_last)
        self.known = set()                  # id(shape) of every item the camera has classified
        self.queue = []                     # (shape, material) plastic/metal items waiting to be picked

        self.t = 0.0
        self.counts = {"plastic": 0, "metal": 0, "general": 0, "missed": 0}
        self.test_spawn = test_spawn
        self._next_spawn = 1.0
        self.rng = np.random.default_rng()
        self._check_reach()

        y_belt = self.cross.start[1]
        self.hover_z = BELT_HEIGHT + TALLEST_ITEM + APPROACH
        self.q_wait = self._solve_above(PICK_WINDOW_X, y_belt, self.hover_z)
        self.q_place = {m: self._solve_above(*self._place_xy(m), self.hover_z) for m in self.lines}

    # ---------- layout helpers ----------
    def _place_xy(self, material):
        line = self.lines[material]
        p = line.start + line.dir * PLACE_PROGRESS
        return float(p[0]), float(p[1])

    def _check_reach(self):
        """Fail early with a clear message if a belt was moved out of reach."""
        b = FANUC_BASE.t[:2]
        y_belt = self.cross.start[1]
        points = {"pick window": (PICK_WINDOW_X, y_belt), "last-chance pick": (LAST_CHANCE_X - 0.12, y_belt)}
        points.update({f"{m} place": self._place_xy(m) for m in self.lines})
        for name, xy in points.items():
            d = float(np.linalg.norm(np.array(xy) - b))
            if d > 0.9 * REACH:
                raise RuntimeError(f"FANUC {name} at {np.round(xy, 2)} is {d:.2f} m from the base; "
                                   f"LR Mate 200iD reach is {REACH} m -- move the belt closer")

    # ---------- world ----------
    def step(self):
        """Per-substep callback: camera watches the belt (also while the arm is
        busy), spawn test items (standalone only), advance the shared clock."""
        self.t += DT
        self._observe()
        if self.test_spawn and self.t >= self._next_spawn:
            self._spawn_test_item()
        self.world.tick()

    def _observe(self):
        """Classify every new item in view; queue plastic/metal, let general go.
        Queued items that get past the last pick point while the arm is busy are missed."""
        for shape, material in self.camera.scan():
            if id(shape) in self.known:
                continue
            self.known.add(id(shape))
            if material == "general":
                self.counts["general"] += 1
                print(f"[{self.t:6.1f}s] FANUC camera: {self._kind(shape):15s} -> general, left on belt")
            else:
                self.queue.append((shape, material))
        for shape, material in list(self.queue):
            if float(np.array(shape.T)[0, 3]) < LAST_CHANCE_X:
                self.queue.remove((shape, material))
                self.counts["missed"] += 1
                print(f"[{self.t:6.1f}s] FANUC missed {self._kind(shape)} ({material}) -- arm was busy")

    def _spawn_test_item(self):
        kind = self.rng.choice(list(TEST_ITEMS))
        mesh, r, h, colour = TEST_ITEMS[kind]
        x, y = self.world.conveyors["main_line"].end[0], self.cross.start[1]
        pose = SE3(x, y, BELT_HEIGHT + h / 2)
        shape = (Cylinder(r, h, pose=pose, color=colour) if mesh == "cyl"
                 else Cuboid([2 * r, 2 * r, h], pose=pose, color=colour))
        self.world.add(shape)
        self.world.tag_item(shape, kind=kind, shape_type=mesh, radius=r, height=h)
        self.world.add_active(shape)
        self._next_spawn = self.t + 5.0

    # ---------- robot ----------
    def _yaw(self, x, y):
        b = self.robot.base.t
        return float(np.arctan2(y - b[1], x - b[0]))

    def _solve_above(self, x, y, z):
        yaw = self._yaw(x, y)
        seed = [yaw, 0.3, 0.2, 0, -pi / 2 - 0.5, 0]
        return self.robot.solve(LRMate200iD.down_pose(x, y, z, yaw), seed)

    def _kind(self, shape):
        return (self.world.get_item_info(shape) or {}).get("kind", "unknown")

    def _pick_tracking(self, shape, info):
        """Descend onto a moving item, following it along the belt, then grasp."""
        r = self.robot
        h = info.get("height", TALLEST_ITEM)
        yaw = float(r.fkine(r.q).rpy()[2])          # keep the current gripper heading
        z = self.hover_z
        z_grasp = BELT_HEIGHT + h - GRASP_BELOW_TOP
        while z > z_grasp:
            z = max(z_grasp, z - DESCEND_SPEED * DT)
            p = np.array(shape.T)[:3, 3]             # item keeps moving with the belt
            T = SE3(p[0], p[1], z) * SE3.Rz(yaw) * SE3.Rx(pi)
            r.q = r.solve(T, r.q)
            self.step()
        self.world.remove_active(shape)              # gripper has it: belt stops dragging it
        r.set_gripper(info.get("radius", 0.03), self.step, DT)
        r.grasp(shape)
        p = np.array(shape.T)[:3, 3]
        r.move_line(LRMate200iD.down_pose(p[0], p[1], self.hover_z, self._yaw(p[0], p[1])), self.step, DT)

    def _place(self, shape, info, material):
        r = self.robot
        h = info.get("height", TALLEST_ITEM)
        x, y = self._place_xy(material)
        yaw = self._yaw(x, y)
        r.move_joint(self.q_place[material], self.step, DT)
        z_release = BELT_HEIGHT + 0.01 + h - GRASP_BELOW_TOP
        r.move_line(LRMate200iD.down_pose(x, y, z_release, yaw), self.step, DT, speed=0.15)
        r.set_gripper(FINGER_OPEN, self.step, DT, duration=0.15)
        r.release()
        shape.T = SE3(x, y, BELT_HEIGHT + h / 2).A   # settle upright on the belt
        self.world.add_active(shape)                 # now carried by plastic/metal line
        r.move_line(LRMate200iD.down_pose(x, y, self.hover_z, yaw), self.step, DT)

    def sort(self, shape, material):
        info = self.world.get_item_info(shape) or {}
        try:
            self._pick_tracking(shape, info)
        except RuntimeError as err:                  # IK failed mid-track: give up on this one
            print(f"[{self.t:6.1f}s] FANUC could not pick {self._kind(shape)}: {err}".splitlines()[0])
            self.world.add_active(shape)
            self.robot.release()
            self.robot.move_joint(self.q_wait, self.step, DT)
            return
        self._place(shape, info, material)
        self.counts[material] += 1
        print(f"[{self.t:6.1f}s] FANUC: {self._kind(shape):15s} -> {TARGET_LINE[material]}   {self.counts}")
        self.robot.move_joint(self.q_wait, self.step, DT)

    def run(self):
        self.robot.move_joint(self.q_wait, self.step, DT)
        while True:
            if self.queue:
                shape, material = self.queue[0]      # oldest = furthest along the belt
                if float(np.array(shape.T)[0, 3]) <= PICK_WINDOW_X + TRIGGER_AHEAD:
                    self.queue.pop(0)
                    self.sort(shape, material)
                    continue
            self.step()                              # wait for the next item to reach the pick window


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    ClassifyCell(World(env, dt=DT), test_spawn=True).run()