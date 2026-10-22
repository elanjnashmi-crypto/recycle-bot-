"""
FANUC LR Mate 200iD (standard arm) with a parallel-jaw gripper (Swift, Python).

Sources -- nothing here is guessed:
  * Kinematics, joint limits and visual meshes: ROS-Industrial
    fanuc_lrmate200id_support (github.com/ros-industrial/fanuc, BSD licence,
    see meshes/LICENSE_ros-industrial_fanuc.txt), file
    urdf/lrmate200id_macro.xacro:
        J1 0.330 m above the base, J2 0.050 m forward of J1,
        upper arm (J2->J3) 0.330 m, J3->J4 0.035 m, forearm (J4->J5) 0.335 m,
        J5->flange 0.080 m.
  * Joint speeds: FANUC LR Mate 200iD datasheet (450, 380, 520, 550, 545,
    1000 deg/s), which match the URDF's velocity limits.
  * Reach 717 mm, payload 7 kg: FANUC datasheet.
    (0.050 + 0.330 + sqrt(0.335^2 + 0.035^2) = 0.717 m, so the URDF agrees.)

The URDF has the same joint layout as the Yaskawa GP7 used elsewhere in this
project (both are ROS-Industrial models), so the DH table below has the same
form as yaskawa_gp7/yaskawa.py, just with the LR Mate's lengths. The DH model
is checked against the URDF chain in verify_against_urdf().

Usage:
    robot = LRMate200iD(base=SE3(0, 0, 0.55))
    robot.add_to_env(env)
    robot.move_joint(q_goal, step, dt)    # step() advances the world one dt
"""
import sys
import types

# macOS: the 'keyboard' package (imported by ir_support) crashes Python on
# import (bus error). This project never uses it, so give it a harmless stand-in.
if sys.platform == "darwin" and "keyboard" not in sys.modules:
    class _NoKeyboard(types.ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: False
    sys.modules["keyboard"] = _NoKeyboard("keyboard")

import os
from math import pi

import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid
from ir_support.robots.DHRobot3D import DHRobot3D

# Colours from fanuc_resources/urdf/common_colours.xacro
FANUC_YELLOW = (0.96, 0.76, 0.13, 1.0)
FANUC_GRAY28 = (71 / 255, 71 / 255, 71 / 255, 1.0)
FANUC_BLACK = (0.15, 0.15, 0.15, 1.0)
LIGHT_GREY = (0.75, 0.75, 0.78, 1.0)

# From the URDF (lrmate200id_macro.xacro), degrees
QLIM_DEG = [(-170, 170), (-100, 145), (-70, 205), (-190, 190), (-125, 125), (-360, 360)]
QD_MAX_DEG = [450, 380, 520, 550, 545, 1000]   # rated joint speeds, deg/s (datasheet = URDF)
SPEED_SCALE = 0.30                             # fraction of rated speed at 100 % override (was 0.50)
ROT_SPEED = 1.5                                # rad/s, tool re-orientation during straight-line moves
RMRC_W0 = 0.02                                 # manipulability below which RMRC adds damping
RMRC_LAMBDA = 0.05                             # maximum damping (damped least squares)
LINK_RADIUS = 0.05                             # m, arm links as capsules for collision checks
GRIPPER_RADIUS = 0.05                          # m, palm + fingers
REACH = 0.717                                  # m, datasheet

# Link lengths (m), URDF joint origins
D1, A1, A2, A3, D4, D6 = 0.330, 0.050, 0.330, 0.035, 0.335, 0.080

# Upper arm vertical, forearm forward, tool pointing straight down
HOME_Q = [0, 0, 0, 0, -pi / 2, 0]

# Gripper, measured out along the tool axis from the flange
PALM_LEN = 0.05
FINGER_LEN = 0.07
FINGER_THICK = 0.012
FINGER_OPEN = 0.055                       # half-gap between fingers when open
TCP_OFFSET = PALM_LEN + FINGER_LEN - 0.02  # tool centre point, 2 cm above the fingertips


class LRMate200iD(DHRobot3D):
    def __init__(self, base=None):
        links = self._create_DH()

        link3D_names = dict(
            link0="base_link", color0=FANUC_GRAY28,
            link1="link_1", color1=FANUC_YELLOW,
            link2="link_2", color2=FANUC_YELLOW,
            link3="link_3", color3=FANUC_YELLOW,
            link4="link_4", color4=FANUC_YELLOW,
            link5="link_5", color5=FANUC_YELLOW,
            link6="link_6", color6=FANUC_BLACK,
        )

        # Pose of each mesh (in the robot's own frame) when q = qtest. The ROS
        # meshes are drawn in their URDF link frames, which at q = 0 are all
        # aligned with the base (upper arm vertical, forearm along +x), so
        # each is just the URDF joint origins summed up the chain.
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [spb.transl(x, 0, z)
                            for x, z in ((0, 0),
                                         (0, D1),
                                         (A1, D1),
                                         (A1, D1 + A2),
                                         (A1, D1 + A2 + A3),
                                         (A1 + D4, D1 + A2 + A3),
                                         (A1 + D4 + D6, D1 + A2 + A3))]

        mesh_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "meshes", "original_stl")
        super().__init__(links, link3D_names, name="LRMate200iD", link3d_dir=mesh_dir,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        # The last DH frame's z points back into the wrist; flip it so tool z
        # points out of the flange, then move out to the gripper's TCP.
        self.tool = SE3.Rx(pi) * SE3(0, 0, TCP_OFFSET)
        self.qd_max = np.radians(QD_MAX_DEG)

        self._palm = Cuboid([0.04, 0.11, PALM_LEN], color=FANUC_BLACK)
        self._fingers = [Cuboid([0.025, FINGER_THICK, FINGER_LEN], color=LIGHT_GREY)
                         for _ in range(2)]
        self._finger_gap = FINGER_OPEN
        self._held = None
        self._held_offset = None
        # Live speed factor 0..1, read every motion step, so a move can be slowed
        # or held part-way. None -> always full speed. 0 -> hold position.
        self.speed_fn = None
        # Collision guard: fn(q) -> True if pose q would hit something. Checked
        # every step of move_joint / move_line; a blocked step is held, not taken.
        self.collision_fn = None
        self.blocked = False
        self._held_size = (0.0, 0.0)              # (radius, height) of the held item

        if base is not None:
            self.base = base
        self.q = HOME_Q

    @staticmethod
    def _create_DH():
        # Joint signs and zeros match the URDF, so QLIM_DEG applies directly.
        d      = [D1,    0,     0,     -D4,   0,     -D6]
        a      = [A1,    A2,    A3,     0,    0,      0]
        alpha  = [-pi/2, pi,   -pi/2,   pi/2, -pi/2,  0]
        offset = [0,    -pi/2,  0,      0,    0,      0]
        return [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i],
                               qlim=np.radians(QLIM_DEG[i]))
                for i in range(6)]

    # ---------- scene ----------
    def __setattr__(self, name, value):
        # DHRobot3D redraws the meshes *before* storing the new q, so they lag
        # one step behind. Store first, then redraw arm, gripper and payload.
        rtb.DHRobot.__setattr__(self, name, value)
        if name in ("q", "base") and "_relation_matrices" in self.__dict__:
            self._update_3dmodel()
            self._update_gripper()

    def add_to_env(self, env):
        super().add_to_env(env)
        env.add(self._palm)
        for finger in self._fingers:
            env.add(finger)
        self._update_gripper()

    def _update_gripper(self):
        if "_fingers" not in self.__dict__:
            return
        tcp = self.fkine(self.q)
        flange = tcp * SE3(0, 0, -TCP_OFFSET)
        self._palm.T = (flange * SE3(0, 0, PALM_LEN / 2)).A
        for side, finger in zip((1, -1), self._fingers):
            y = side * (self._finger_gap + FINGER_THICK / 2)
            finger.T = (flange * SE3(0, y, PALM_LEN + FINGER_LEN / 2)).A
        if self._held is not None:
            self._held.T = (tcp * self._held_offset).A

    # ---------- gripper ----------
    def set_gripper(self, half_gap, step, dt, duration=0.3):
        """Open/close the fingers to `half_gap` (m from the TCP) over `duration` s
        (at 100 % speed)."""
        start = self._finger_gap

        def apply(u):
            self._finger_gap = start + u * (half_gap - start)
            self._update_gripper()
        self._run_profile(duration, apply, step, dt)

    def grasp(self, obj, radius=0.0, height=0.0):
        """Rigidly attach a scene object to the gripper where it is right now.
        radius/height let collision checks include the held item."""
        self._held = obj
        self._held_offset = self.fkine(self.q).inv() * SE3(np.array(obj.T), check=False)
        self._held_size = (radius, height)

    def release(self):
        obj, self._held = self._held, None
        self._held_size = (0.0, 0.0)
        return obj

    # ---------- collision model ----------
    def collision_points(self, q):
        """The arm as a set of spheres (centre, radius) for pose q: points along
        each link (between consecutive DH frame origins), the gripper, and the
        held item if there is one."""
        frames = [T.t for T in self.fkine_all(q)]
        tcp = self.fkine(q)
        flange = (tcp * SE3(0, 0, -TCP_OFFSET)).t
        tip = (tcp * SE3(0, 0, 0.02)).t                  # fingertips, 2 cm past the TCP
        pts = []
        for a, b in zip(frames[1:], frames[2:]):         # links 2..6 (link 1 is inside the pedestal top)
            for s in np.linspace(0, 1, 4):
                pts.append((a + s * (b - a), LINK_RADIUS))
        for s in np.linspace(0, 1, 4):
            pts.append((flange + s * (tip - flange), GRIPPER_RADIUS))
        if self._held is not None:
            r, h = self._held_size
            c = (tcp * self._held_offset).t
            for dz in np.linspace(-h / 2, h / 2, 3):
                pts.append((c + np.array([0, 0, dz]), r))
        return pts

    # ---------- motion ----------
    def solve(self, T, q_seed):
        """IK for TCP pose T, starting from q_seed so the arm keeps the same posture."""
        sol = self.ikine_LM(T, q0=q_seed, joint_limits=True)
        if not sol.success:
            raise RuntimeError(f"LR Mate 200iD cannot reach\n{T}")
        return sol.q

    def speed_factor(self):
        """Current speed factor 0..1 from speed_fn; 1 when nothing is set."""
        if self.speed_fn is None:
            return 1.0
        return min(1.0, max(0.0, float(self.speed_fn())))

    @staticmethod
    def _quintic(s):
        """Smooth 0->1 progress (zero velocity and acceleration at both ends)."""
        return s ** 3 * (10 - 15 * s + 6 * s * s)

    def _run_profile(self, duration, apply, step, dt):
        """Drive a move along a quintic profile that takes `duration` s at 100 %.
        The speed factor is read EVERY step, so the speed slider and the safety
        slow-down act in the middle of a move; at 0 the arm holds still but keeps
        stepping the world (so the other stations are not held up)."""
        duration = max(float(duration), 2 * dt)
        s = 0.0
        while s < 1.0:
            s_next = min(1.0, s + dt * self.speed_factor() / duration)
            if apply(self._quintic(s_next)) is False:   # collision guard: hold this step
                self.blocked = True
            else:
                self.blocked = False
                s = s_next
            step()

    def _guarded_set(self, q):
        """Move to q unless the collision guard says q hits something."""
        if self.collision_fn is not None and self.collision_fn(q):
            return False
        self.q = q
        return True

    def move_joint(self, q_goal, step, dt):
        """Quintic joint-space move, timed so no joint exceeds SPEED_SCALE of its
        rated speed at 100 % (a quintic peaks at 1.875 * travel / duration)."""
        q0 = np.array(self.q, dtype=float)
        q_goal = np.array(q_goal, dtype=float)
        T = np.max(1.875 * np.abs(q_goal - q0) / (self.qd_max * SPEED_SCALE))

        def apply(u):
            return self._guarded_set(q0 + u * (q_goal - q0))
        self._run_profile(T, apply, step, dt)

    def move_line(self, T_goal, step, dt, speed=0.15):
        """Straight-line TCP move, peak `speed` m/s at 100 % (approach, descend, lift)."""
        T0 = self.fkine(self.q)
        dist = float(np.linalg.norm(T_goal.t - T0.t))
        turn = float(abs((T0.inv() * T_goal).angvec()[0]))
        T = 1.875 * max(dist / speed, turn / ROT_SPEED)

        def apply(u):
            return self._guarded_set(self.solve(T0.interp(T_goal, u), self.q))
        self._run_profile(T, apply, step, dt)

    def rmrc_step(self, v, dt):
        """Resolved-rate motion control, one step: turn a desired TCP velocity
        v = [vx, vy, vz, wx, wy, wz] (world frame) into joint velocities with the
        Jacobian, q_dot = J^T (J J^T + lambda^2 I)^-1 v (damped least squares:
        damping only switches on near a singularity), then integrate one dt.
        Joint speeds are capped at SPEED_SCALE of rated and joints stay in limits."""
        q = np.array(self.q, dtype=float)
        J = self.jacob0(q)
        w = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))           # manipulability
        lam2 = 0.0 if w >= RMRC_W0 else (1 - (w / RMRC_W0) ** 2) * RMRC_LAMBDA ** 2
        qd = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), np.asarray(v, dtype=float))
        qd_lim = self.qd_max * SPEED_SCALE
        scale = np.max(np.abs(qd) / qd_lim)
        if scale > 1.0:                                           # keep the direction, slow it down
            qd = qd / scale
        q_new = np.clip(q + qd * dt, self.qlim[0] + 1e-3, self.qlim[1] - 1e-3)
        self.q = q_new
        return w

    @staticmethod
    def down_pose(x, y, z, yaw=0.0):
        """TCP pose at (x, y, z) with the gripper pointing straight down."""
        return SE3(x, y, z) * SE3.Rz(yaw) * SE3.Rx(pi)

    # ---------- checks ----------
    def verify_against_urdf(self, n=200, seed=0):
        """Compare the DH model's flange pose with the URDF joint chain for
        random joint angles. Returns the worst position error (m)."""
        def urdf_flange(q):
            def rot(axis, ang):
                R = {"x": spb.trotx, "y": spb.troty, "z": spb.trotz}[axis[-1]]
                return R(-ang if axis[0] == "-" else ang)
            chain = [((0, 0, D1), "z", q[0]), ((A1, 0, 0), "y", q[1]), ((0, 0, A2), "-y", q[2]),
                     ((0, 0, A3), "-x", q[3]), ((D4, 0, 0), "-y", q[4]), ((D6, 0, 0), "-x", q[5])]
            T = np.eye(4)
            for xyz, axis, ang in chain:
                T = T @ spb.transl(*xyz) @ rot(axis, ang)
            return T

        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n):
            q = rng.uniform(self.qlim[0], self.qlim[1])
            dh = np.eye(4)
            for link, qi in zip(self.links, q):
                dh = dh @ link.A(qi).A
            worst = max(worst, float(np.linalg.norm(dh[:3, 3] - urdf_flange(q)[:3, 3])))
        return worst

    def test(self):
        """Show the robot in Swift and swing through its base rotation."""
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)
        dt = 0.05

        def step():
            env.step(dt)

        print(f"DH vs URDF worst flange error: {self.verify_against_urdf() * 1000:.3f} mm")
        lo, hi = self.qlim[:, 0]
        for q1 in (lo, hi, 0):
            self.move_joint([q1, 0.3, 0.2, 0, -pi / 2 - 0.5, 0], step, dt)
        self.set_gripper(0.02, step, dt)
        self.set_gripper(FINGER_OPEN, step, dt)
        print("TCP pose:\n", self.fkine(self.q))
        env.hold()


if __name__ == "__main__":
    LRMate200iD().test()