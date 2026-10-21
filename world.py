"""
The single shared simulation world: one Swift environment, one
ConveyorSystem, and one list of "active" items (things currently riding a
belt rather than held by a gripper).

Every station (InfeedCell, SortingCell, UR3eCell, FANUC...) is built around a
shared World instead of creating its own env/ConveyorSystem. That is what
makes this "one workcell" rather than separate demos side by side.

SHARED CLOCK (important)
------------------------
Each station runs its own blocking motion loop in its own thread and calls
world.tick() once per motion sub-step. If every call advanced the world, N
stations would advance it N times faster than each robot moves (belts race
ahead and every robot crawls, because it only gets 1 of every N ticks). So
tick() is a lock-step barrier: the world advances ONE dt per round, once all
registered stations have asked for their next step. Idle stations keep
calling tick() while they wait, so they stay in the round.

SWIFT IS NOT THREAD-SAFE
------------------------
Swift talks to the browser over one request/reply socket. Anything that
touches it after the stations start (env.add for spawned items, env.step)
must go through World.add()/World.tick(), which share one lock.

Usage from main_sim.py:
    world.register()  # once per station, BEFORE its thread starts
    ... thread runs station.run() ...
    world.unregister()  # in the thread's finally block (station finished/crashed)
A standalone station (python some_cell.py) needs no registration.
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
        self.item_info = {}          # id(shape) -> dict(kind=..., radius=..., height=...), set by whoever spawns it
        self._cond = threading.Condition(threading.RLock())
        self._n_stations = 0         # registered stations (0 -> standalone, treated as 1)
        self._waiting = 0            # stations waiting in the current round
        self._round = 0              # round counter (lets waiters detect a finished round)

    # ---------- station registration ----------
    def register(self):
        with self._cond:
            self._n_stations += 1

    def unregister(self):
        """A station finished or crashed: drop it so the others are not left
        waiting for it forever."""
        with self._cond:
            self._n_stations = max(0, self._n_stations - 1)
            if self._waiting >= max(1, self._n_stations) and self._waiting > 0:
                self._finish_round()
            self._cond.notify_all()

    # ---------- thread-safe Swift access ----------
    def add(self, shape):
        """Add a shape to Swift from any thread (use this for anything
        spawned while the stations are running)."""
        with self._cond:
            self.env.add(shape)

    # ---------- item registry ----------
    def tag_item(self, shape, **info):
        """Record what an item is (kind, radius, height, ...) when it is spawned,
        so downstream stations' sensors can identify it."""
        with self._cond:
            self.item_info[id(shape)] = dict(info)

    def get_item_info(self, shape):
        """What the spawner recorded for this item, or None if it was never tagged."""
        with self._cond:
            return self.item_info.get(id(shape))

    def add_active(self, shape):
        """Call once an item is released onto a belt, so the shared
        ConveyorSystem starts carrying it."""
        with self._cond:
            if shape not in self.active_items:
                self.active_items.append(shape)

    def remove_active(self, shape):
        """Call once a robot grasps an item off a belt, so the belt stops
        trying to drag it (its pose is now controlled by the gripper)."""
        with self._cond:
            if shape in self.active_items:
                self.active_items.remove(shape)

    # ---------- e-stop ----------
    def estop(self):
        """Simulated / physical e-stop trip: halt belts and freeze the clock.
        Station threads block inside tick() (no busy loop, no motion) until
        reset() AND resume() have both been called. Does NOT clear on its own."""
        with self._cond:
            self.running = False
            self.conveyors.stop()

    def reset(self):
        """Deliberate acknowledge/reset after an e-stop. Clears the conveyor
        fault but does not itself resume motion -- resume() is a separate
        action, per the brief's reset-then-start requirement."""
        with self._cond:
            self.conveyors.start()

    def resume(self):
        """Separate start/resume action after reset()."""
        with self._cond:
            self.running = True
            self._cond.notify_all()

    # ---------- stepping ----------
    def _finish_round(self):
        """Advance the world by one dt. Caller holds self._cond."""
        self.t += self.dt
        self.conveyors.step(self.dt, list(self.active_items))
        self.env.step(self.dt)
        self._waiting = 0
        self._round += 1
        self._cond.notify_all()

    def tick(self):
        """Called by a station once per motion sub-step. Blocks until every
        registered station has called it, then the world advances one dt
        (conveyors + one Swift frame) and all stations are released together."""
        with self._cond:
            while not self.running:                  # e-stopped: wait, don't spin
                self._cond.wait(0.1)
            self._waiting += 1
            my_round = self._round
            if self._waiting >= max(1, self._n_stations):
                self._finish_round()
                return
            while self._round == my_round:
                self._cond.wait(0.5)                 # timeout so a stalled round can't hang forever
                if self._round == my_round and self._waiting >= max(1, self._n_stations):
                    self._finish_round()