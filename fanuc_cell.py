import queue
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
from fanuc_lrmate200id.lrmate200id import LRMate200iD, FINGER_OPEN, FINGER_LEN, PALM_LEN, REACH, TCP_OFFSET

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
DESCEND_SPEED = 0.30        # m/s, max descent speed while servoing onto a moving item
BELT_SPEED_REF = 0.15       # m/s, belt speed DESCEND_SPEED was tuned for (scaled up for faster belts)

# ---- visual servoing + RMRC (the moving-item pick) -------------------------
KP_XY = 8.0                 # 1/s, gain on the camera's gripper-to-item position error
KP_Z = 12.0                 # 1/s, gain on the height error (high: come down at full speed, settle fast)
KR = 3.0                    # 1/s, gain that keeps the gripper pointing straight down
CAM_NOISE = 0.001           # m, camera position noise (1 mm, one standard deviation)
CAM_VEL_FILTER = 0.5        # weight of each new frame in the camera's velocity estimate
SERVO_TOL_XY = 0.006        # m, grasp when the gripper is within 6 mm of the item ...
SERVO_TOL_Z = 0.004         # m, ... and within 4 mm of grasp height
SERVO_TIMEOUT = 4.0         # s, give up on an item if the servo hasn't converged

# ---- collision avoidance -----------------------------------------------------
PATH_SAMPLES = 30           # poses checked along every planned joint move
VIA_MAX_Z = 1.45            # m, highest TCP height tried when going over an obstacle
REPLAN_EVERY = 0.5          # s, while blocked, re-check this often
PLACE_OPTIONS = (0.15, 0.40)  # m along the output line: normal drop spot, then the backup spot
# ---- teach / jog (manual mode) ----------------------------------------------
T1_TCP_SPEED = 0.25         # m/s, manual-mode TCP speed limit (ISO 10218-1 reduced speed, "T1")
T1_JOINT_SCALE = 0.10       # manual joint speed, fraction of rated
SAFE_X = (-0.80, 0.80)      # manual-mode safe zone for the TCP (like FANUC DCS position check)
SAFE_Y = (-0.70, 0.85)
SAFE_Z = (BELT_HEIGHT + 0.03, 1.60)
FINGERTIP_BELOW_TCP = PALM_LEN + FINGER_LEN - TCP_OFFSET
LIFT_SPEED = 0.25           # m/s peak, lift off the belt (at 100 %)
PLACE_SPEED = 0.18          # m/s peak, lower onto the plastic/metal line (at 100 %)
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


class PathBlocked(Exception):
    """Raised inside a planned move when an obstacle has appeared in the way and
    the guard has held the arm for REPLAN_EVERY seconds: _go() then replans."""


class OverheadCamera:
    """Simulated vision sensor looking down at a stretch of the cross line.
    Reports the material of every item inside its field of view. (In the sim
    the 'image recognition' is the kind the spawner tagged the item with.)"""

    def __init__(self, world, belt, s_from, s_to):
        self.world = world
        self.belt = belt
        self.s_from, self.s_to = s_from, s_to
        self.seen = {}                      # id(shape) -> material (each item classified once)
        self.track = {}                     # id(shape) -> (measured position, estimated velocity, frame)
        self.frame = 0
        self.rng = np.random.default_rng(1)

    def estimate(self, shape):
        """Camera's current measurement of an item: (position, velocity), or None
        if the item is not in view this frame."""
        tr = self.track.get(id(shape))
        if tr is None or tr[2] < self.frame:
            return None
        return tr[0], tr[1]

    def scan(self):
        """One camera frame (called every 0.05 s): classify new items and update
        each visible item's measured position and filtered velocity."""
        found = []
        self.frame += 1
        for shape in list(self.world.active_items):
            p = np.array(shape.T)[:3, 3]
            if not self.belt.contains(p) or not (self.s_from <= self.belt.progress(p) <= self.s_to):
                continue
            key = id(shape)
            meas = p + self.rng.normal(0, CAM_NOISE, 3)           # what the camera "sees"
            if key in self.track:
                p_old, v_old, _ = self.track[key]
                v = (1 - CAM_VEL_FILTER) * v_old + CAM_VEL_FILTER * (meas - p_old) / DT
            else:
                v = np.zeros(3)
            self.track[key] = (meas, v, self.frame)
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
        s_last = self.cross.progress(np.array([LAST_CHANCE_X - 0.40, y_belt, BELT_HEIGHT]))
        self.camera = OverheadCamera(world, self.cross, s_drop - 0.25, s_last)
        self.known = set()                  # id(shape) of every item the camera has classified
        self.queue = []                     # (shape, material) plastic/metal items waiting to be picked

        self.t = 0.0
        self.counts = {"plastic": 0, "metal": 0, "general": 0, "missed": 0}

        # Speed factor (1.0 = normal). The safety speed zone slows the whole cell via World instead.
        self.zone_factor = 1.0
        self.robot.speed_fn = lambda: self.zone_factor

        # Collision avoidance: obstacles live in world.obstacles; the GUI asks for
        # them through request_obstacle() and this thread adds them in step().
        self.robot.collision_fn = lambda q: self._hits(q) is not None
        self._obstacle_requests = []
        self._clear_obstacles = False
        self.collision_msg = ""
        self._q_place_cache = {}
        self._in_go = False                 # True while _go() is executing a planned move
        self._blocked_time = 0.0

        # Teach / jog: the GUI sets requested_mode and queues jog commands; this
        # thread switches modes between cycles, so it never drops a held item.
        self.mode = "auto"                  # auto | manual | to_auto
        self.requested_mode = "auto"
        self.jog_target = None              # joint target the manual loop moves towards
        self._jog_cmds = queue.Queue()
        self.jog_msg = ""

        self.test_spawn = test_spawn
        self._next_spawn = 1.0
        self.rng = np.random.default_rng()
        self._check_reach()

        y_belt = self.cross.start[1]
        self.hover_z = BELT_HEIGHT + TALLEST_ITEM + APPROACH
        self.q_wait = self._solve_above(PICK_WINDOW_X, y_belt, self.hover_z)
        self.q_place = {m: self._solve_above(*self._place_xy(m), self.hover_z) for m in self.lines}
        self.obstacle_spots = self._make_obstacle_spots()

    # ---------- layout helpers ----------
    def _place_xy(self, material, progress=PLACE_PROGRESS):
        line = self.lines[material]
        p = line.start + line.dir * progress
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
        if self.test_spawn and self.world.auto_feed and self.t >= self._next_spawn:
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
            elif self.requested_mode == "manual" or self.mode != "auto":
                self.counts["missed"] += 1
                print(f"[{self.t:6.1f}s] FANUC in manual: {self._kind(shape)} ({material}) left on belt")
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

    @staticmethod
    def _orientation_error(R, R_goal):
        """Small rotation (world frame) that turns R towards R_goal."""
        return 0.5 * sum(np.cross(R[:, k], R_goal[:, k]) for k in range(3))

    def _pick_tracking(self, shape, info):
        """Visual servoing + RMRC onto a moving item.
        Every 0.05 s the camera gives the item's position and velocity. The TCP
        velocity command is: the item's velocity (feed-forward, so the gripper keeps
        pace with the belt) + KP_XY x the camera's position error (closes the gap),
        a capped descent towards grasp height, and an angular velocity that keeps
        the gripper pointing down. RMRC turns that into joint velocities."""
        r = self.robot
        h = info.get("height", TALLEST_ITEM)
        z_grasp = BELT_HEIGHT + h - GRASP_BELOW_TOP
        yaw = float(r.fkine(r.q).rpy()[2])          # keep the current gripper heading
        R_goal = (SE3.Rz(yaw) * SE3.Rx(pi)).R
        t = 0.0
        while True:
            if r.speed_factor() <= 0:                # held by a safety stop: don't move at all
                self.step()
                continue
            est = self.camera.estimate(shape)
            if est is None:
                raise RuntimeError("camera lost the item")
            p_item, v_item = est
            T = r.fkine(r.q)
            e_xy = p_item[:2] - T.t[:2]              # visual error: gripper -> item
            e_z = z_grasp - T.t[2]
            if np.linalg.norm(e_xy) < SERVO_TOL_XY and abs(e_z) < SERVO_TOL_Z:
                break
            descend = DESCEND_SPEED * max(1.0, self.cross.speed / BELT_SPEED_REF)
            v = np.zeros(6)
            v[:2] = v_item[:2] + KP_XY * e_xy
            v[2] = np.clip(KP_Z * e_z, -descend, descend)
            v[3:] = KR * self._orientation_error(T.R, R_goal)
            r.rmrc_step(v, DT)
            self.step()
            t += DT
            if t > SERVO_TIMEOUT:
                raise RuntimeError("visual servo did not converge")
        self.world.remove_active(shape)              # gripper has it: belt stops dragging it
        r.set_gripper(info.get("radius", 0.03), self.step, DT)
        r.grasp(shape, info.get("radius", 0.03), h)
        p = np.array(shape.T)[:3, 3]
        r.move_line(LRMate200iD.down_pose(p[0], p[1], self.hover_z, self._yaw(p[0], p[1])), self.step, DT,
                    speed=LIFT_SPEED)

    def _place(self, shape, info, material):
        r = self.robot
        h = info.get("height", TALLEST_ITEM)
        z_release = BELT_HEIGHT + 0.01 + h - GRASP_BELOW_TOP
        x, y, q_place = self._choose_place_spot(material, z_release)
        yaw = self._yaw(x, y)
        self._go(q_place)
        r.move_line(LRMate200iD.down_pose(x, y, z_release, yaw), self.step, DT, speed=PLACE_SPEED)
        r.set_gripper(FINGER_OPEN, self.step, DT, duration=0.15)
        r.release()
        shape.T = SE3(x, y, BELT_HEIGHT + h / 2).A   # settle upright on the belt
        self.world.add_active(shape)                 # now carried by plastic/metal line
        r.move_line(LRMate200iD.down_pose(x, y, self.hover_z, yaw), self.step, DT, speed=LIFT_SPEED)

    def sort(self, shape, material):
        info = self.world.get_item_info(shape) or {}
        try:
            self._pick_tracking(shape, info)
        except RuntimeError as err:                  # IK failed or servo lost the item: give up on it
            print(f"[{self.t:6.1f}s] FANUC could not pick {self._kind(shape)}: {err}".splitlines()[0])
            self.world.add_active(shape)
            self.robot.release()
            self._go(self.q_wait)
            return
        self._place(shape, info, material)
        self.counts[material] += 1
        print(f"[{self.t:6.1f}s] FANUC: {self._kind(shape):15s} -> {TARGET_LINE[material]}   {self.counts}")
        self._go(self.q_wait)

    # ---------- collision detection / avoidance ----------
    def _make_obstacle_spots(self):
        """Two places the GUI can drop a test obstacle, both on the FANUC's real path:
        a tall box on the cross line under the swing from the pick to the metal line,
        and a box sitting on the metal line's drop spot."""
        r = self.robot
        q_mid = 0.5 * (np.array(self.q_wait) + np.array(self.q_place["metal"]))   # half-way through the swing
        mx, my = r.fkine(q_mid).t[:2]
        swing_top = self.hover_z - 0.03                                          # just above the fingertips
        x, y = self._place_xy("metal")
        return {
            "swing path (box on cross line)": dict(centre=(mx, my, (BELT_HEIGHT + swing_top) / 2),
                                                   size=(0.16, 0.16, swing_top - BELT_HEIGHT)),
            "metal line drop spot": dict(centre=(x, y, BELT_HEIGHT + 0.09), size=(0.14, 0.14, 0.18)),
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
            print(f"[{self.t:6.1f}s] FANUC: obstacles cleared")
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
            print(f"[{self.t:6.1f}s] FANUC: obstacle added at {np.round(c[:2], 2)}")

    @staticmethod
    def _point_hits(points, ob):
        lo, hi = ob["lo"], ob["hi"]
        return any(np.all(p >= lo - rad) and np.all(p <= hi + rad) for p, rad in points)

    def _hits(self, q):
        """The obstacle that pose q would touch (arm, gripper or held item), or None."""
        if not self.world.obstacles:
            return None
        pts = self.robot.collision_points(q)
        for ob in self.world.obstacles:
            if self._point_hits(pts, ob):
                return ob
        return None

    def _path_hits(self, q_from, q_to):
        """Check the straight joint-space path from q_from to q_to (the same path
        move_joint follows) at PATH_SAMPLES poses."""
        q_from, q_to = np.array(q_from, float), np.array(q_to, float)
        for u in np.linspace(0, 1, PATH_SAMPLES):
            ob = self._hits(q_from + u * (q_to - q_from))
            if ob is not None:
                return ob
        return None

    def _plan(self, q_goal):
        """List of joint waypoints to q_goal that avoids every obstacle, or None.
        Straight there if the path is clear; otherwise go over the obstacle: up to
        a via height, across, and back down, trying higher via heights until clear."""
        q0 = np.array(self.robot.q, float)
        ob = self._path_hits(q0, q_goal)
        if ob is None:
            return [q_goal]
        x0, y0, _ = self.robot.fkine(q0).t
        x1, y1, _ = self.robot.fkine(q_goal).t
        for z in np.arange(ob["hi"][2] + 0.05, VIA_MAX_Z + 1e-6, 0.05):
            try:
                qa = self.robot.solve(LRMate200iD.down_pose(x0, y0, z, self._yaw(x0, y0)), q0)
                qb = self.robot.solve(LRMate200iD.down_pose(x1, y1, z, self._yaw(x1, y1)), q_goal)
            except RuntimeError:
                break                                        # can't go any higher
            legs = [(q0, qa), (qa, qb), (qb, q_goal)]
            if all(self._path_hits(a, b) is None for a, b in legs):
                self.collision_msg = f"Collision predicted: going over the obstacle at z = {z:.2f} m"
                print(f"[{self.t:6.1f}s] FANUC: {self.collision_msg}")
                return [qa, qb, q_goal]
        return None

    def _go(self, q_goal):
        """Collision-checked joint move. Plans around obstacles; if there's no way
        round, stops and waits until there is. If an obstacle appears part-way
        through, the per-step guard stops the arm, and after REPLAN_EVERY seconds
        it replans from where it stopped."""
        while True:
            plan = self._plan(q_goal)
            if plan is None:
                self.collision_msg = "Collision predicted: no way round - stopped until the obstacle is removed"
                print(f"[{self.t:6.1f}s] FANUC: {self.collision_msg}")
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
                print(f"[{self.t:6.1f}s] FANUC: {self.collision_msg}")
            finally:
                self._in_go = False

    def _q_place_for(self, material, progress):
        key = (material, progress)
        if key not in self._q_place_cache:
            self._q_place_cache[key] = self._solve_above(*self._place_xy(material, progress), self.hover_z)
        return self._q_place_cache[key]

    def _choose_place_spot(self, material, z_release):
        """First drop spot along the output line where lowering the item would not
        hit an obstacle. If every spot is blocked, stop and wait."""
        waited = False
        while True:
            for progress in PLACE_OPTIONS:
                x, y = self._place_xy(material, progress)
                q_hover = self._q_place_for(material, progress)
                try:
                    q_rel = self.robot.solve(LRMate200iD.down_pose(x, y, z_release, self._yaw(x, y)), q_hover)
                except RuntimeError:
                    continue
                if self._path_hits(q_hover, q_rel) is None:
                    if progress != PLACE_OPTIONS[0]:
                        self.collision_msg = f"Drop spot blocked: placing {progress - PLACE_OPTIONS[0]:.2f} m further along"
                        print(f"[{self.t:6.1f}s] FANUC: {self.collision_msg}")
                    elif waited:
                        self.collision_msg = "Drop spot clear again - continuing"
                    return x, y, q_hover
            if not waited:
                self.collision_msg = "Every drop spot is blocked - stopped until an obstacle is removed"
                print(f"[{self.t:6.1f}s] FANUC: {self.collision_msg}")
                waited = True
            for _ in range(int(REPLAN_EVERY / DT)):
                self.step()

    # ---------- teach / jog: GUI interface (called from the GUI thread) ----------
    def request_mode(self, mode):
        if mode in ("auto", "manual"):
            self.requested_mode = mode

    def jog_joint(self, i, angle):
        """Manual: move joint i (0-5) towards `angle` (rad)."""
        self._jog_cmds.put(("joint", int(i), float(angle)))

    def jog_cartesian(self, dx, dy, dz):
        """Manual: move the TCP by (dx, dy, dz) m in the world frame, gripper still pointing down."""
        self._jog_cmds.put(("cart", float(dx), float(dy), float(dz)))

    def goto_joints(self, q):
        """Manual: planned, collision-checked move to joint angles q (rad), e.g. a
        final joint state supplied during marking."""
        self._jog_cmds.put(("goto", np.array(q, dtype=float)))

    def jog_home(self):
        self._jog_cmds.put(("goto", np.array(self.q_wait, dtype=float)))

    def jog_gripper(self, close):
        self._jog_cmds.put(("grip", bool(close)))

    def status(self):
        r = self.robot
        return dict(mode=self.mode, requested=self.requested_mode, q=np.array(r.q, dtype=float),
                    target=None if self.jog_target is None else np.array(self.jog_target, dtype=float),
                    tcp=r.fkine(r.q).t, counts=dict(self.counts), jog_msg=self.jog_msg,
                    collision_msg=self.collision_msg, qlim=np.array(r.qlim, dtype=float))

    # ---------- teach / jog: runs in the FANUC thread ----------
    def _zone_violation(self, q):
        """'' if pose q is allowed in manual mode, else the reason: TCP inside the
        safe box, nothing below the belt surface, and no obstacle hit."""
        r = self.robot
        tcp = r.fkine(q).t
        for v, (lo, hi), axis in zip(tcp, (SAFE_X, SAFE_Y, SAFE_Z), "xyz"):
            if not lo <= v <= hi:
                return f"TCP {axis} = {v:.2f} m is outside the safe zone ({lo:.2f} to {hi:.2f})"
        if tcp[2] - FINGERTIP_BELOW_TCP < BELT_HEIGHT + 0.005:
            return "fingers would hit the belt"
        for T in list(r.fkine_all(q))[2:]:          # elbow, wrist and flange frames
            if T.t[2] < SAFE_Z[0]:
                return "arm would go below the belt surface"
        if self._hits(q) is not None:
            return "would hit an obstacle"
        return ""

    def _handle_jog_command(self, cmd):
        r = self.robot
        if cmd[0] == "joint":
            _, i, angle = cmd
            target = np.array(self.jog_target, dtype=float)
            target[i] = float(np.clip(angle, r.qlim[0, i], r.qlim[1, i]))
            self.jog_target = target
            self.jog_msg = ""
        elif cmd[0] == "cart":
            _, dx, dy, dz = cmd
            goal = SE3(dx, dy, dz) * r.fkine(self.jog_target)    # world-frame translation, same orientation
            try:
                q = r.solve(goal, self.jog_target)
            except RuntimeError:
                self.jog_msg = "Jog refused: out of reach"
                return
            why = self._zone_violation(q)
            if why:
                self.jog_msg = f"Jog refused: {why}"
                return
            self.jog_target = q
            self.jog_msg = ""
        elif cmd[0] == "goto":
            q = cmd[1]
            if np.any(q < r.qlim[0]) or np.any(q > r.qlim[1]):
                self.jog_msg = "Refused: outside the joint limits"
                return
            why = self._zone_violation(q)
            if why:
                self.jog_msg = f"Refused: {why}"
                return
            self.jog_msg = "Planning a collision-checked move to the joint state..."
            self._go(q)                              # same planner as auto: goes over obstacles or waits
            self.jog_target = np.array(r.q, dtype=float)
            self.jog_msg = "Reached the requested joint state"
        elif cmd[0] == "grip":
            r.set_gripper(0.02 if cmd[1] else FINGER_OPEN, self.step, DT)

    def _jog_step(self):
        """One manual step towards jog_target at reduced (T1) speed, stopping at the
        safe-zone boundary or before an obstacle."""
        r = self.robot
        q = np.array(r.q, dtype=float)
        dq = np.array(self.jog_target, dtype=float) - q
        if np.max(np.abs(dq)) < 1e-5:
            return
        dq = np.clip(dq, -r.qd_max * T1_JOINT_SCALE * DT, r.qd_max * T1_JOINT_SCALE * DT)
        moved = float(np.linalg.norm(r.fkine(q + dq).t - r.fkine(q).t))
        if moved > T1_TCP_SPEED * DT:                # keep the TCP under 250 mm/s
            dq *= T1_TCP_SPEED * DT / moved
        why = self._zone_violation(q + dq)
        if why:
            self.jog_target = q                      # stop where we are
            self.jog_msg = f"Stopped: {why}"
            return
        r.q = q + dq

    def _manual_loop(self):
        self.jog_target = np.array(self.robot.q, dtype=float)
        while not self._jog_cmds.empty():            # drop jogs sent before we got here
            self._jog_cmds.get_nowait()
        self.mode = "manual"
        self.jog_msg = "Manual mode: jog enabled (250 mm/s max)"
        print(f"[{self.t:6.1f}s] FANUC: manual (teach) mode")
        while self.requested_mode == "manual":
            while not self._jog_cmds.empty():
                self._handle_jog_command(self._jog_cmds.get_nowait())
            self._jog_step()
            self.step()
        self.mode = "to_auto"
        self.jog_msg = "Returning to the wait pose"
        self.robot.set_gripper(FINGER_OPEN, self.step, DT)
        self._go(self.q_wait)
        self.jog_target = None
        self.mode = "auto"
        self.jog_msg = ""
        print(f"[{self.t:6.1f}s] FANUC: auto mode")

    # ---------- main loop ----------
    def run(self):
        self._go(self.q_wait)
        while True:
            if self.requested_mode == "manual":
                self._manual_loop()
                continue
            if self.queue:
                shape, material = self.queue[0]      # oldest = furthest along the belt
                if float(np.array(shape.T)[0, 3]) <= PICK_WINDOW_X + TRIGGER_AHEAD:
                    self.queue.pop(0)
                    self.sort(shape, material)
                    continue
            self.step()                              # wait for the next item to reach the pick window


if __name__ == "__main__":
    # Standalone: FANUC + belts + safety + control panel, no other robots.
    import threading
    from control_panel import run_panel
    from safety import SafetySystem

    env = swift.Swift()
    env.launch(realtime=True)
    world = World(env, dt=DT)
    cell = ClassifyCell(world, test_spawn=True)
    safety = SafetySystem(world, cell)
    threading.Thread(target=cell.run, daemon=True, name="ClassifyCell").start()
    run_panel(world, cell, safety)