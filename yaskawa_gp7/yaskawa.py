#Dom's Bot

import os
from math import pi

import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid
from ir_support.robots.DHRobot3D import DHRobot3D

YASKAWA_BLUE = (0.10, 0.32, 0.68, 1.0)
DARK_GREY = (0.20, 0.20, 0.22, 1.0)
LIGHT_GREY = (0.75, 0.75, 0.78, 1.0)

# From the GP7 URDF.
QLIM_DEG = [(-170, 170), (-65, 145), (-70, 190), (-190, 190), (-135, 135), (-360, 360)]
QD_MAX_DEG = [375, 315, 410, 550, 550, 1000]   # rated joint speeds, deg/s
SPEED_SCALE = 0.3125                           # fraction of rated speed used for moves
ROT_SPEED = 1.5                                # rad/s, tool re-orientation during straight-line moves
RMRC_W0 = 0.02                                 # manipulability below which RMRC adds damping
RMRC_LAMBDA = 0.05                             # maximum damping (damped least squares)
LINK_RADIUS = 0.07                             # m, arm links as capsules for collision checks
GRIPPER_RADIUS = 0.06                          # m, palm + fingers
REACH = 0.927                                  # m, datasheet

# Arm up, forearm forward, tool pointing straight down
HOME_Q = [0, 0, 0, 0, -pi / 2, 0]

# Gripper, measured out along the tool axis from the flange
PALM_LEN = 0.06
FINGER_LEN = 0.08
FINGER_THICK = 0.015
FINGER_OPEN = 0.06                      # half-gap between fingers when open
TCP_OFFSET = PALM_LEN + FINGER_LEN - 0.02  # tool centre point, 2 cm above the fingertips


class GP7(DHRobot3D):
    def __init__(self, base=None):
        links = self._create_DH()

        link3D_names = dict(
            link0="gp7_base_link", color0=DARK_GREY,
            link1="gp7_link_1_s", color1=YASKAWA_BLUE,
            link2="gp7_link_2_l", color2=YASKAWA_BLUE,
            link3="gp7_link_3_u", color3=YASKAWA_BLUE,
            link4="gp7_link_4_r", color4=YASKAWA_BLUE,
            link5="gp7_link_5_b", color5=YASKAWA_BLUE,
            link6="gp7_link_6_t", color6=DARK_GREY,
        )

        # Pose of each mesh in the world when q = qtest.
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [spb.transl(x, 0, z)
                            for x, z in ((0, 0), (0, 0.330), (0.040, 0.330), (0.040, 0.775),
                                         (0.480, 0.815), (0.480, 0.815), (0.480, 0.815))]

        mesh_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "meshes", "GP7")
        super().__init__(links, link3D_names, name="GP7", link3d_dir=mesh_dir,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        # The last DH frame's z points back into the wrist
        self.tool = SE3.Rx(pi) * SE3(0, 0, TCP_OFFSET)
        self.qd_max = np.radians(QD_MAX_DEG)

        self._palm = Cuboid([0.05, 0.12, PALM_LEN], color=DARK_GREY)
        self._fingers = [Cuboid([0.03, FINGER_THICK, FINGER_LEN], color=LIGHT_GREY)
                         for _ in range(2)]
        self._finger_gap = FINGER_OPEN
        self._held = None
        self._held_offset = None
        self.speed_fn = None
        # Collision guard
        self.collision_fn = None
        self.blocked = False
        self._held_size = (0.0, 0.0)              # (radius, height) of the held item

        if base is not None:
            self.base = base
        self.q = HOME_Q

    def _create_DH(self):
        # Joint signs and zeros match the URDF, so QLIM_DEG applies directly.
        d      = [0.330, 0,      0,     -0.440, 0,     -0.080]
        a      = [0.040, 0.445,  0.040,  0,     0,      0]
        alpha  = [-pi/2, pi,    -pi/2,   pi/2, -pi/2,   0]
        offset = [0,    -pi/2,   0,      0,     0,      0]
        return [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i],
                               qlim=np.radians(QLIM_DEG[i]))
                for i in range(6)]

    # ---------- scene ----------
    def __setattr__(self, name, value):
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
    def set_gripper(self, half_gap, step, dt, duration=0.32):
        """Open/close the fingers to `half_gap` (m from the TCP) over `duration` s."""
        start = self._finger_gap

        def apply(u):
            self._finger_gap = start + u * (half_gap - start)
            self._update_gripper()
        self._run_profile(duration, apply, step, dt)

    def grasp(self, obj, radius=0.0, height=0.0):
        """Rigidly attach a scene object to the gripper where it is right now."""
        self._held = obj
        self._held_offset = self.fkine(self.q).inv() * SE3(np.array(obj.T), check=False)
        self._held_size = (radius, height)

    def release(self):
        obj, self._held = self._held, None
        self._held_size = (0.0, 0.0)
        return obj

    # ---------- collision model ----------
    def collision_points(self, q):
        frames = [T.t for T in self.fkine_all(q)]
        tcp = self.fkine(q)
        flange = (tcp * SE3(0, 0, -TCP_OFFSET)).t
        tip = (tcp * SE3(0, 0, 0.02)).t                  # fingertips, 2 cm past the TCP
        pts = []
        for a, b in zip(frames[1:], frames[2:]):         # links 2..6 (link 1 sits on the pedestal)
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
            raise RuntimeError(f"GP7 cannot reach\n{T}")
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
        q0 = np.array(self.q, dtype=float)
        q_goal = np.array(q_goal, dtype=float)
        T = np.max(1.875 * np.abs(q_goal - q0) / (self.qd_max * SPEED_SCALE))

        def apply(u):
            return self._guarded_set(q0 + u * (q_goal - q0))
        self._run_profile(T, apply, step, dt)

    def move_line(self, T_goal, step, dt, speed=0.35):
        """Straight-line TCP move, `speed` m/s peak (for approach, descend and lift)."""
        T0 = self.fkine(self.q)
        dist = float(np.linalg.norm(T_goal.t - T0.t))
        turn = float(abs((T0.inv() * T_goal).angvec()[0]))
        T = 1.875 * max(dist / speed, turn / ROT_SPEED)

        def apply(u):
            return self._guarded_set(self.solve(T0.interp(T_goal, u), self.q))
        self._run_profile(T, apply, step, dt)

    def rmrc_step(self, v, dt):
        """One resolved-rate step: TCP twist v (vx, vy, vz, wx, wy, wz) -> joint rates."""
        q = np.array(self.q, dtype=float)
        J = self.jacob0(q)
        w = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))           # manipulability
        lam2 = 0.0 if w >= RMRC_W0 else (1 - (w / RMRC_W0) ** 2) * RMRC_LAMBDA ** 2
        qd = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), np.asarray(v, dtype=float))
        qd_lim = self.qd_max * SPEED_SCALE
        scale = np.max(np.abs(qd) / qd_lim)
        if scale > 1.0:                                           # keep the direction, slow it down
            qd = qd / scale
        self.q = np.clip(q + qd * dt, self.qlim[0] + 1e-3, self.qlim[1] - 1e-3)
        return w

    @staticmethod
    def down_pose(x, y, z, yaw=0.0):
        """TCP pose at (x, y, z) with the gripper pointing straight down."""
        return SE3(x, y, z) * SE3.Rz(yaw) * SE3.Rx(pi)

    def test(self):
        """Show the robot in Swift and swing through its full base rotation."""
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)
        dt = 0.05

        def step():
            env.step(dt)

        lo, hi = self.qlim[:, 0]
        for q1 in (lo, hi, 0):
            self.move_joint([q1, 0.3, 0.2, 0, -pi / 2 - 0.5, 0], step, dt)
        self.set_gripper(0.02, step, dt)
        self.set_gripper(FINGER_OPEN, step, dt)
        print("TCP pose:\n", self.fkine(self.q))
        env.hold()


if __name__ == "__main__":
    GP7().test()
