"""
User-chosen item spawning for the control panel.

The GUI calls request(kind, where) from its own thread. Nothing touches Swift
there: the request is queued and created inside World's per-round hook (world
lock held), the same way every other shape is added while the stations run.

Items use the infeed's RUBBISH_ITEMS table and are tagged exactly like the
Kawasaki tub's items, so the FANUC camera, the Yaskawa load cell and the
organic basket treat them the same as normal rubbish.
"""
from spatialmath import SE3
from spatialgeometry import Cylinder, Mesh, Sphere
import numpy as np

from conveyors import BELT_HEIGHT
from infeed_cell import RUBBISH_ITEMS
from item_meshes import ensure_bottle_mesh

CLEARANCE = 0.30        # m: wait until no other item is this close to the drop point
MIN_GAP = 14.0          # s (sim time) between spawns at the same point, so the FANUC
                        # (about 10-12 s per pick at 100 %) can pick each spawned item

# where: (belt, distance along it from its start, m)
SPAWN_POINTS = {
    "main line (start of the whole line)": ("main_line", 0.40),
    "cross line (straight to the FANUC)": ("cross_line", 0.45),
}


class ItemSpawner:
    def __init__(self, world):
        self.world = world
        self.pending = []           # (kind, where)
        self.spawned = 0
        self._last = {}             # where -> sim time of the last spawn there
        world.add_hook(self._update)

    @property
    def kinds(self):
        return list(RUBBISH_ITEMS)

    @property
    def places(self):
        return list(SPAWN_POINTS)

    def request(self, kind, where):
        """Thread-safe: queue an item; it appears on the next round with space for it."""
        if kind not in RUBBISH_ITEMS or where not in SPAWN_POINTS:
            raise ValueError(f"unknown item/place: {kind!r}, {where!r}")
        self.pending.append((kind, where))

    # ---------- runs inside World (lock held) ----------
    def _drop_point(self, where):
        belt, s = SPAWN_POINTS[where]
        c = self.world.conveyors[belt]
        p = c.start + c.dir * s
        return float(p[0]), float(p[1])

    def _clear(self, x, y):
        for shape in self.world.active_items:
            p = np.array(shape.T)[:3, 3]
            if np.hypot(p[0] - x, p[1] - y) < CLEARANCE:
                return False
        return True

    def _update(self):
        if not self.pending:
            return
        kind, where = self.pending[0]
        x, y = self._drop_point(where)
        if self.world.t - self._last.get(where, -MIN_GAP) < MIN_GAP or not self._clear(x, y):
            return                          # something is passing: try again next round
        self.pending.pop(0)
        self._last[where] = self.world.t
        spec = RUBBISH_ITEMS[kind]
        r, h = spec["radius"], spec["height"]
        pose = SE3(x, y, BELT_HEIGHT + h / 2)
        if spec["shape"] == "cyl":
            shape = Cylinder(r, h, pose=pose, color=spec["colour"])
        elif spec["shape"] == "sphere":
            shape = Sphere(r, pose=pose, color=spec["colour"])
        else:
            shape = Mesh(ensure_bottle_mesh(), pose=pose, color=spec["colour"])
        self.world.add(shape)
        self.world.tag_item(shape, kind=kind, material=spec["material"], radius=r, height=h,
                            mass_kg=spec["mass_kg"], is_aluminium=spec["is_aluminium"])
        self.world.add_active(shape)
        self.spawned += 1
        print(f"[{self.world.t:6.1f}s] spawned {kind} on {where.split(' (')[0]}")