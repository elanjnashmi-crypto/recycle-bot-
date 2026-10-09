"""Quick visual test of the conveyor system. Run from the Assignment 2 folder:
    python test_conveyors.py
"""
import time
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder

from conveyors import ConveyorSystem, BELT_HEIGHT

env = swift.Swift()
env.launch(realtime=True)

conveyors = ConveyorSystem(env)

# A few test items: one at the top of each input line
items = [
    Cuboid([0.12, 0.12, 0.12], pose=SE3(-0.30, 3.30, BELT_HEIGHT + 0.06),
           color=[0.8, 0.6, 0.3, 1]),                           # cardboard box -> general line
    Cylinder(0.04, 0.20, pose=SE3(-0.90, 0.10, BELT_HEIGHT + 0.10),
             color=[0.2, 0.6, 1.0, 1]),                         # plastic bottle
    Cylinder(0.035, 0.12, pose=SE3(0.90, 0.10, BELT_HEIGHT + 0.06),
             color=[0.75, 0.75, 0.8, 1]),                       # metal can
]
for item in items:
    env.add(item)

dt = 0.05
t = 0.0
while True:
    conveyors.step(dt, items)
    env.step(dt)
    t += dt

    # E-stop test: belts freeze between 10 s and 13 s
    if 10.0 <= t < 10.0 + dt:
        print("Conveyors stopped")
        conveyors.stop()
    if 13.0 <= t < 13.0 + dt:
        print("Conveyors restarted")
        conveyors.start()