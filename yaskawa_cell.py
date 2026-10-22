from dataclasses import dataclass
from math import pi

import numpy as np
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

from conveyors import BELT_HEIGHT
from fanuc_cell import OverheadCamera
from world import World
from yaskawa_gp7.yaskawa import GP7, FINGER_OPEN, FINGER_LEN, REACH

DT = 0.05

PEDESTAL_HEIGHT = 0.45                  # floor-mounted, the GP7 can't reach down onto a 0.75 m belt
YASKAWA_BASE = SE3(1.90, 1.70, PEDESTAL_HEIGHT)   # 0.6 m left of the metal_return end stop
AL_BIN_XY = (1.30, 1.30)                # down-left of the robot (see sketch)
NON_AL_BIN_XY = (1.30, 2.00)            # left of the robot
BIN_SIZE = (0.40, 0.40, 0.35)           # open-top deposit box, sitting on the floor

AL_THRESHOLD_KG = 0.030                 # lighter than this -> aluminium
LOAD_CELL_NOISE_KG = 0.002

APPROACH = 0.15                         # clearance above items / bins for approach moves
TALLEST_ITEM = 0.17
LIFT_SPEED = 0.35                       # m/s peak, lift off the belt
GRASP_BELOW_TOP = FINGER_LEN - 0.01     # TCP stops this far below the item's top

DEFAULT_RADIUS, DEFAULT_HEIGHT, DEFAULT_MASS_KG = 0.033, 0.122, 0.055

# ---- camera: the same OverheadCamera as the FANUC, watching the end of metal_return ----
CAM_WINDOW = 0.60                       # m of belt before the end stop the camera can see
STOPPED_SPEED = 0.04                    # m/s, item counts as stopped below this (belt runs 0.15)

# ---- visual servoing + RMRC (the final descent onto the item) -----------------
KP_XY = 8.0                             # 1/s, gain on the camera's gripper-to-item position error
KP_Z = 12.0
KR = 3.0                                # 1/s, gain that keeps the gripper pointing straight down
DESCEND_SPEED = 0.30                    # m/s, max descent speed while servoing
SERVO_TOL_XY = 0.006                    # m, grasp when the gripper is within 6 mm of the item ...
SERVO_TOL_Z = 0.004                     # m, ... and within 4 mm of grasp height
SERVO_TIMEOUT = 4.0                     # s, give up on an item if the servo hasn't converged

# ---- collision avoidance -----------------------------------------------------
PATH_SAMPLES = 30                       # poses checked along every planned joint move
VIA_MAX_Z = 1.45                        # m, highest TCP height tried when going over an obstacle
REPLAN_EVERY = 0.5                      # s, while blocked, re-check this often
DROP_OPTIONS = ((0, 0), (0.13, 0), (-0.13, 0), (0, 0.13), (0, -0.13))   # offsets inside a bin


class PathBlocked(Exception):
    pass


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

    def obstacle(self):
        return dict(lo=np.array([self.x - BIN_SIZE[0] / 2, self.y - BIN_SIZE[1] / 2, 0.0]),
                    hi=np.array([self.x + BIN_SIZE[0] / 2, self.y + BIN_SIZE[1] / 2, BIN_SIZE[2]]))

    def drop(self, item, step, x, y):
        """Let a released item fall straight down from (x, y) onto the bin floor."""
        inner = BIN_SIZE[0] / 2 - self.WALL - item.radius
        x = float(np.clip(x + self.rng.normal(0, 0.01), self.x - inner, self.x + inner))
        y = float(np.clip(y + self.rng.normal(0, 0.01), self.y - inner, self.y + inner))
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
        stop = self.belt.length - 0.1                     # where the end stop holds items
        self.camera = OverheadCamera(world, self.belt, stop - CAM_WINDOW, stop + 0.05)
        self.s_stop = stop
        self.ready = []                     # shapes the camera has seen stopped at the end stop

        self.t = 0.0
        self.counts = {"Al": 0, "Non-Al": 0}
        self._warned_untagged = False

        # Speed factor (1.0 = normal).
        self.zone_factor = 1.0
        self.robot.speed_fn = lambda: self.zone_factor

        # Collision avoidance: test obstacles (world.obstacles) + the two bins
        self.fixed_obstacles = [bn.obstacle() for bn in self.bins.values()]
        self.robot.collision_fn = lambda q: self._hits(q) is not None
        self._obstacle_requests = []
        self._clear_obstacles = False
        self.collision_msg = ""
        self._in_go = False                 # True while _go() is executing a planned move
        self._blocked_time = 0.0
        self._check_reach()

        # Waiting pose: gripper above the end stop
        pick = self.belt.end - self.belt.dir * 0.1
        self.hover_z = BELT_HEIGHT + TALLEST_ITEM + APPROACH
        self.q_wait = self._solve_above(pick[0], pick[1], self.hover_z)
        self._q_drop_cache = {}
        self.obstacle_spots = self._make_obstacle_spots()

    # ---------- layout helpers ----------
    def _check_reach(self):
        """Fail early with a clear message if the belt or a bin was moved out of reach."""
        b = YASKAWA_BASE.t[:2]
        pick = self.belt.end[:2] - self.belt.dir[:2] * 0.1
        points = {"pick point": pick, "Al bin": AL_BIN_XY, "Non-Al bin": NON_AL_BIN_XY}
        for name, xy in points.items():
            d = float(np.linalg.norm(np.array(xy) - b))
            if d > 0.9 * REACH:
                raise RuntimeError(f"Yaskawa {name} at {np.round(xy, 2)} is {d:.2f} m from the base; "
                                   f"GP7 reach is {REACH} m -- move it closer")

    # ---------- world ----------
    def step(self):
        self.t += DT
        self._handle_obstacle_requests()
        self._observe()
        if self.robot.blocked:                       # guard is holding the arm (obstacle in the way)
            self._blocked_time += DT
            if self._in_go and self._blocked_time >= REPLAN_EVERY:
                self._blocked_time = 0.0
                self.world.tick()
                raise PathBlocked()
        else:
            self._blocked_time = 0.0
        self.world.tick()

    def _observe(self):
        for shape, _material in self.camera.scan():
            p, v = self.camera.estimate(shape)
            stopped = self.belt.progress(p) >= self.s_stop - 0.01 and np.linalg.norm(v[:2]) < STOPPED_SPEED
            if stopped and shape not in self.ready:
                self.ready.append(shape)
        self.ready = [s for s in self.ready if s in self.world.active_items]

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

    # ---------- robot ----------
    def _yaw(self, x, y):
        b = self.robot.base.t
        return float(np.arctan2(y - b[1], x - b[0]))

    def _solve_above(self, x, y, z):
        yaw = self._yaw(x, y)
        seed = [yaw, 0.3, 0.2, 0, -pi / 2 - 0.5, 0]
        return self.robot.solve(GP7.down_pose(x, y, z, yaw), seed)

    @staticmethod
    def _orientation_error(R, R_goal):
        """Small rotation (world frame) that turns R towards R_goal."""
        return 0.5 * sum(np.cross(R[:, k], R_goal[:, k]) for k in range(3))

    def _pick_servo(self, item):
        """Camera-guided descent: RMRC drives the TCP onto the camera's item estimate."""
        r = self.robot
        z_grasp = BELT_HEIGHT + item.height - GRASP_BELOW_TOP
        yaw = float(r.fkine(r.q).rpy()[2])          # keep the current gripper heading
        R_goal = (SE3.Rz(yaw) * SE3.Rx(pi)).R
        t = 0.0
        while True:
            if r.speed_factor() <= 0:                # held by a safety stop: don't move at all
                self.step()
                continue
            est = self.camera.estimate(item.shape)
            if est is None:
                raise RuntimeError("camera lost the item")
            p_item, v_item = est
            T = r.fkine(r.q)
            e_xy = p_item[:2] - T.t[:2]              # visual error: gripper -> item
            e_z = z_grasp - T.t[2]
            if np.linalg.norm(e_xy) < SERVO_TOL_XY and abs(e_z) < SERVO_TOL_Z:
                break
            v = np.zeros(6)
            v[:2] = KP_XY * e_xy
            v[2] = np.clip(KP_Z * e_z, -DESCEND_SPEED, DESCEND_SPEED)
            v[3:] = KR * self._orientation_error(T.R, R_goal)
            r.rmrc_step(v, DT)
            self.step()
            t += DT
            if t > SERVO_TIMEOUT:
                raise RuntimeError("visual servo did not converge")
        self.world.remove_active(item.shape)         # gripper has it: belt stops dragging it
        r.set_gripper(item.radius, self.step, DT)
        r.grasp(item.shape, item.radius, item.height)
        p = item.pos
        r.move_line(GP7.down_pose(p[0], p[1], self.hover_z, self._yaw(p[0], p[1])), self.step, DT,
                    speed=LIFT_SPEED)

    def pick_and_sort(self, item):
        r = self.robot
        try:
            est = self.camera.estimate(item.shape)
            if est is None:
                raise RuntimeError("camera lost the item")
            x, y = est[0][:2]
            r.move_line(GP7.down_pose(x, y, self.hover_z, self._yaw(x, y)), self.step, DT)
            self._pick_servo(item)
        except RuntimeError as err:                  # IK failed or servo lost the item: give up on it
            print(f"[{self.t:6.1f}s] Yaskawa could not pick {item.kind}: {err}".splitlines()[0])
            if r.release() is not None:
                self.world.add_active(item.shape)
            self._go(self.q_wait)
            return

        mass = self.load_cell.read(item)
        is_al = mass < AL_THRESHOLD_KG
        label = "Al" if is_al else "Non-Al"
        self.counts[label] += 1
        print(f"[{self.t:6.1f}s] {item.kind:18s} {mass * 1000:5.1f} g -> "
              f"{label}{'' if is_al == item.is_aluminium else '  (MIS-SORTED)'}   {self.counts}")

        x, y, q_drop = self._choose_drop_spot(is_al)
        self._go(q_drop)
        r.set_gripper(FINGER_OPEN, self.step, DT, duration=0.16)
        r.release()
        self.bins[is_al].drop(item, self.step, x, y)
        self._go(self.q_wait)

    # ---------- collision detection / avoidance ----------
    def _make_obstacle_spots(self):
        r = self.robot
        q_mid = 0.5 * (np.array(self.q_wait) + np.array(self._q_drop_for(True, (0, 0))))
        mx, my, mz = r.fkine(q_mid).t                                       # half-way through the swing
        top = mz - 0.10
        bn = self.bins[True]
        return {
            "Yaskawa swing path (box on a stand)": dict(centre=(mx, my, top / 2), size=(0.16, 0.16, top)),
            "Al bin drop spot (box in the bin)": dict(centre=(bn.x, bn.y, 0.225), size=(0.14, 0.14, 0.45)),
        }

    def request_obstacle(self, spot):
        """GUI thread: drop a test obstacle at a named spot (added on the next step)."""
        self._obstacle_requests.append(spot)

    def request_clear_obstacles(self):
        self._clear_obstacles = True

    def _handle_obstacle_requests(self):
        if self._clear_obstacles:
            self._clear_obstacles = False
            for ob in self.world.obstacles:
                ob["shape"].T = SE3(0, 0, -10).A                 # parked out of sight
            self.world.obstacles.clear()
            self.collision_msg = "Obstacles cleared"
            print(f"[{self.t:6.1f}s] Yaskawa: obstacles cleared")
        while self._obstacle_requests:
            spot = self.obstacle_spots[self._obstacle_requests.pop(0)]
            c, size = np.array(spot["centre"]), np.array(spot["size"])
            ob = dict(lo=c - size / 2, hi=c + size / 2)
            if self._point_hits(self.robot.collision_points(self.robot.q), ob):
                self.collision_msg = "Can't drop it there: the arm is in the way"
                continue
            ob["shape"] = Cuboid(list(size), pose=SE3(*c), color=(0.85, 0.35, 0.05, 1))
            self.world.add(ob["shape"])
            self.world.obstacles.append(ob)
            print(f"[{self.t:6.1f}s] Yaskawa: obstacle added at {np.round(c[:2], 2)}")

    @staticmethod
    def _point_hits(points, ob):
        lo, hi = ob["lo"], ob["hi"]
        return any(np.all(p >= lo - rad) and np.all(p <= hi + rad) for p, rad in points)

    def _hits(self, q):
        """The obstacle that pose q would touch (arm, gripper or held item), or None."""
        pts = self.robot.collision_points(q)
        for ob in self.world.obstacles + self.fixed_obstacles:
            if self._point_hits(pts, ob):
                return ob
        return None

    def _path_hits(self, q_from, q_to):
        q_from, q_to = np.array(q_from, float), np.array(q_to, float)
        for u in np.linspace(0, 1, PATH_SAMPLES):
            ob = self._hits(q_from + u * (q_to - q_from))
            if ob is not None:
                return ob
        return None

    def _plan(self, q_goal):
        q0 = np.array(self.robot.q, float)
        ob = self._path_hits(q0, q_goal)
        if ob is None:
            return [q_goal]
        x0, y0, _ = self.robot.fkine(q0).t
        x1, y1, _ = self.robot.fkine(q_goal).t
        for z in np.arange(ob["hi"][2] + 0.05, VIA_MAX_Z + 1e-6, 0.05):
            try:
                qa = self.robot.solve(GP7.down_pose(x0, y0, z, self._yaw(x0, y0)), q0)
                qb = self.robot.solve(GP7.down_pose(x1, y1, z, self._yaw(x1, y1)), q_goal)
            except RuntimeError:
                break                                        # can't go any higher
            legs = [(q0, qa), (qa, qb), (qb, q_goal)]
            if all(self._path_hits(a, b) is None for a, b in legs):
                self.collision_msg = f"Collision predicted: going over the obstacle at z = {z:.2f} m"
                print(f"[{self.t:6.1f}s] Yaskawa: {self.collision_msg}")
                return [qa, qb, q_goal]
        return None

    def _go(self, q_goal):
        while True:
            plan = self._plan(q_goal)
            if plan is None:
                self.collision_msg = "Collision predicted: no way round - stopped until the obstacle is removed"
                print(f"[{self.t:6.1f}s] Yaskawa: {self.collision_msg}")
                while plan is None:
                    for _ in range(int(REPLAN_EVERY / DT)):
                        self.step()
                    plan = self._plan(q_goal)
                self.collision_msg = "Path clear again - continuing"
            try:
                self._in_go = True
                for q in plan:
                    self.robot.move_joint(q, self.step, DT)
                return
            except PathBlocked:
                self.collision_msg = "Obstacle appeared in the path - stopped and replanning"
                print(f"[{self.t:6.1f}s] Yaskawa: {self.collision_msg}")
            finally:
                self._in_go = False

    def _q_drop_for(self, is_al, offset):
        key = (is_al, offset)
        if key not in self._q_drop_cache:
            bn = self.bins[is_al]
            self._q_drop_cache[key] = self._solve_above(bn.x + offset[0], bn.y + offset[1],
                                                        BIN_SIZE[2] + APPROACH)
        return self._q_drop_cache[key]

    def _choose_drop_spot(self, is_al):
        """Centre of the bin if it's clear, otherwise another spot inside it; wait if all blocked."""
        bn = self.bins[is_al]
        waited = False
        while True:
            for offset in DROP_OPTIONS:
                try:
                    q = self._q_drop_for(is_al, offset)
                except RuntimeError:
                    continue
                if self._hits(q) is None:
                    if offset != DROP_OPTIONS[0]:
                        self.collision_msg = "Bin drop spot blocked: dropping beside it in the same bin"
                        print(f"[{self.t:6.1f}s] Yaskawa: {self.collision_msg}")
                    elif waited:
                        self.collision_msg = "Bin drop spot clear again - continuing"
                    return bn.x + offset[0], bn.y + offset[1], q
            if not waited:
                self.collision_msg = "Every drop spot in the bin is blocked - stopped until an obstacle is removed"
                print(f"[{self.t:6.1f}s] Yaskawa: {self.collision_msg}")
                waited = True
            for _ in range(int(REPLAN_EVERY / DT)):
                self.step()

    def status(self):
        return dict(counts=dict(self.counts), collision_msg=self.collision_msg)

    # ---------- main loop ----------
    def run(self):
        self._go(self.q_wait)
        while True:
            if self.ready:
                self.pick_and_sort(self._as_item(self.ready.pop(0)))
                continue
            self.step()                              # wait for the camera to see an item stop


if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    SortingCell(World(env, dt=DT)).run()
