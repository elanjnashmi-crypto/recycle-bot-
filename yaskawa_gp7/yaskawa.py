"""
Yaskawa Motoman GP7 with a parallel-jaw gripper (Swift, Python).

Kinematics, joint limits and meshes come from ROS-Industrial's
motoman_gp7_support package (github.com/ros-industrial/motoman, kinetic-devel):
shoulder 0.330 m above the floor, upper arm 0.445 m, forearm 0.440 m,
wrist-to-flange 0.080 m, 927 mm reach, 7 kg payload.

Usage:
    robot = GP7(base=SE3(1.9, 1.7, 0))
    robot.add_to_env(env)
    robot.move_joint(q_goal, step)    # step() advances the world one dt
"""
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

# From the GP7 URDF. J1 is +/-170 deg on the real robot (340 deg total);
# raise it to 180 here if you want a full turn in simulation.
QLIM_DEG = [(-170, 170), (-65, 145), (-70, 190), (-190, 190), (-135, 135), (-360, 360)]
QD_MAX_DEG = [375, 315, 410, 550, 550, 1000]   # rated joint speeds, deg/s
SPEED_SCALE = 0.3125                           # fraction of rated speed used for moves

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

        # Pose of each mesh in the world when q = qtest. The ROS meshes are drawn
        # in their URDF link frames, which at q = 0 are all world-aligned
        # (upper arm vertical, forearm horizontal along +x), so only translations.
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [spb.transl(x, 0, z)
                            for x, z in ((0, 0), (0, 0.330), (0.040, 0.330), (0.040, 0.775),
                                         (0.480, 0.815), (0.480, 0.815), (0.480, 0.815))]

        mesh_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "meshes", "GP7")
        super().__init__(links, link3D_names, name="GP7", link3d_dir=mesh_dir,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        # The last DH frame's z points back into the wrist; flip it so tool z
        # points out of the flange, then move out to the gripper's TCP.
        self.tool = SE3.Rx(pi) * SE3(0, 0, TCP_OFFSET)
        self.qd_max = np.radians(QD_MAX_DEG)

        self._palm = Cuboid([0.05, 0.12, PALM_LEN], color=DARK_GREY)
        self._fingers = [Cuboid([0.03, FINGER_THICK, FINGER_LEN], color=LIGHT_GREY)
                         for _ in range(2)]
        self._finger_gap = FINGER_OPEN
        self._held = None
        self._held_offset = None

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
    def set_gripper(self, half_gap, step, dt, duration=0.32):
        """Open/close the fingers to `half_gap` (m from the TCP) over `duration` s."""
        start = self._finger_gap
        n = max(2, int(np.ceil(duration / dt)))
        for gap in np.linspace(start, half_gap, n):
            self._finger_gap = gap
            self._update_gripper()
            step()

    def grasp(self, obj):
        """Rigidly attach a scene object to the gripper where it is right now."""
        self._held = obj
        self._held_offset = self.fkine(self.q).inv() * SE3(np.array(obj.T), check=False)

    def release(self):
        obj, self._held = self._held, None
        return obj

    # ---------- motion ----------
    def solve(self, T, q_seed):
        """IK for TCP pose T, starting from q_seed so the arm keeps the same posture."""
        sol = self.ikine_LM(T, q0=q_seed, joint_limits=True)
        if not sol.success:
            raise RuntimeError(f"GP7 cannot reach\n{T}")
        return sol.q

    def move_joint(self, q_goal, step, dt):
        """Quintic joint-space move, timed so no joint exceeds SPEED_SCALE of its
        rated speed (a quintic peaks at 1.875 * travel / duration)."""
        q0 = np.array(self.q, dtype=float)
        q_goal = np.array(q_goal, dtype=float)
        T = max(np.max(1.875 * np.abs(q_goal - q0) / (self.qd_max * SPEED_SCALE)), 2 * dt)
        for q in rtb.jtraj(q0, q_goal, int(np.ceil(T / dt)) + 1).q:
            self.q = q
            step()

    def move_line(self, T_goal, step, dt, speed=0.19):
        """Straight-line TCP move at `speed` m/s (for approach, descend and lift)."""
        T0 = self.fkine(self.q)
        dist = np.linalg.norm(T_goal.t - T0.t)
        n = max(2, int(np.ceil(dist / (speed * dt))) + 1)
        for T in rtb.ctraj(T0, T_goal, n):
            self.q = self.solve(T, self.q)
            step()

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
