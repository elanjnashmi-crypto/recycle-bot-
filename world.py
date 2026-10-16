"""
The single shared simulation world: one Swift environment, one
ConveyorSystem, and one list of "active" items (things currently riding a
belt rather than held by a gripper).

Every station (InfeedCell, SortingCell, and whatever the FANUC station ends
up being called) should be built around a shared World instance instead of
each creating its own env/ConveyorSystem. That's what actually makes this
"one workcell" rather than three separate demos running side by side.

Each robot driver's blocking move helpers (GP7.move_line, RS007L moves, etc.)
take a per-substep callback. Point that callback at a station's own step()
method, and have that station's step() call world.tick() -- see
infeed_cell.py / yaskawa_cell.py for the pattern.
"""
import threading

from conveyors import ConveyorSystem


class World:
    def __init__(self, env, dt=0.05):
        self.env = env
        self.dt = dt
        self.conveyors = ConveyorSystem(env)
        self.t = 0.0
        self.running = True          # single e-stop flag for the whole workcell
        self.active_items = []       # shapes currently belt-driven (not held by any gripper)
        self._lock = threading.Lock()

    # ---------- item registry ----------
    def add_active(self, shape):
        """Call once an item is released onto a belt, so the shared
        ConveyorSystem starts carrying it."""
        with self._lock:
            if shape not in self.active_items:
                self.active_items.append(shape)

    def remove_active(self, shape):
        """Call once a robot grasps an item off a belt, so the belt stops
        trying to drag it (its pose is now controlled by the gripper)."""
        with self._lock:
            if shape in self.active_items:
                self.active_items.remove(shape)

    # ---------- e-stop ----------
    def estop(self):
        """Simulated / physical e-stop trip: halt belts and stop advancing
        the world. Does NOT clear on its own -- see reset()/resume()."""
        self.running = False
        self.conveyors.stop()

    def reset(self):
        """Deliberate acknowledge/reset action after an e-stop. Clears the
        fault on the conveyors but does not itself resume motion -- call
        resume() as a separate action, per the assignment's requirement
        that reset and resume are distinct steps."""
        self.conveyors.start()

    def resume(self):
        """Separate start/resume action after reset()."""
        self.running = True

    # ---------- stepping ----------
    def tick(self):
        """Advance the shared world by one dt: animate/move belt items,
        render one Swift frame. Thread-safe -- every station's motion loop
        calls this every interpolation sub-step, so calls are serialised
        to keep env.step() single-threaded.

        Caveat: if two stations are both mid-move at the same wall-clock
        moment, both will call tick() independently, so belts/spawners can
        advance faster than `dt` alone suggests. Fine for a visual demo;
        swap for a fixed-rate scheduler thread if you need exact timing.
        """
        if not self.running:
            return
        with self._lock:
            self.t += self.dt
            self.conveyors.step(self.dt, list(self.active_items))
            self.env.step(self.dt)