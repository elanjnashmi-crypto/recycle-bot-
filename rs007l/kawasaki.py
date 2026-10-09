import os
import time
from math import pi

import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as spb
import swift
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D

WHITE = (0.95, 0.95, 0.95, 1.0)
BLACK = (0.10, 0.10, 0.10, 1.0)


class RS007L(DHRobot3D):
    def __init__(self, base=None):
        links = self._create_DH()

        link3D_names = dict(
            link0="RS007L_J0", color0=WHITE,   # base
            link1="RS007L_J1", color1=GREEN,   # waist
            link2="RS007L_J2", color2=WHITE,   # upper arm
            link3="RS007L_J3", color3=GREEN,   # elbow housing
            link4="RS007L_J4", color4=WHITE,   # forearm
            link5="RS007L_J5", color5=GREEN,   # wrist
            link6="RS007L_J6", color6=BLACK,   # tool flange
        )

        # Pose of each mesh in the world when q = qtest (arm straight up).
        # Every mesh is rotated 90 deg about z and lifted to its joint height.
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [spb.transl(0, 0, z) @ spb.trotz(pi / 2)
                            for z in (0.0, 0.360, 0.360, 0.815, 1.290, 1.290, 1.368)]

        mesh_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "meshes", "RS007L")
        super().__init__(links, link3D_names, name="RS007L", link3d_dir=mesh_dir,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        if base is not None:
            self.base = base
        self.q = qtest

    def _create_DH(self):
        d     = [0.360, 0,     0,      0.475,  0,      0.078]
        a     = [0,     0.455, 0,      0,      0,      0]
        alpha = [pi/2,  0,     pi/2,  -pi/2,   pi/2,   0]
        offset = [0,    pi/2,  pi/2,   0,      0,      0]
        qlim_deg = [180, 135, 157, 200, 125, 360]
        return [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i],
                               qlim=np.radians([-qlim_deg[i], qlim_deg[i]]))
                for i in range(6)]

    def test(self):
        """Show the robot in Swift and run a short joint-space move."""
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)

        q_goal = [0, -pi/4, -pi/4, 0, -pi/2, 0]   # reach out along +x, tool pointing down
        for q in rtb.jtraj(self.q, q_goal, 80).q:
            self.q = q
            env.step(0.03)
        print("End-effector pose:\n", self.fkine(self.q))
        env.hold()


if __name__ == "__main__":
    RS007L().test()
