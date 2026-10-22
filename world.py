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
import time

from conveyors import ConveyorSystem


class World:
    def __init__(self, env, dt=0.05):
        self.env = env
        self.dt = dt
        self.conveyors = ConveyorSystem(env)
        self.t = 0.0
        self.running = True          # single e-stop flag for the whole workcell
        self.auto_feed = True        # False = manual spawning: the infeed stops feeding new items
        self.slow_factor = 1.0       # <1 slows the WHOLE cell in real time (safety speed zone)
        self.active_items = []       # shapes currently belt-driven (not held by any gripper)
        self.item_info = {}          # id(shape) -> dict(kind=..., radius=..., height=...), set by whoever spawns it
        self._cond = threading.Condition(threading.RLock())
        self._n_stations = 0         # registered stations (0 -> standalone, treated as 1)
        self._waiting = 0            # stations waiting in the current round
        self._round = 0              # round counter (lets waiters detect a finished round)
        # e-stop state: running=False freezes every station inside tick();
        # latched=True means it has not been reset yet (reset, THEN resume).
        self.latched = False
        self.stop_reason = ""
        self._interlocks = []        # fn() -> "" if OK, else why reset/resume is refused
        self._hooks = []             # fn() run once per round AND while e-stopped (lock held)
        self._last_pump = 0.0

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

    # ---------- belt speed ----------
    def set_belt_speed(self, speed):
        """Set every belt's speed (m/s). Safe to call from the GUI thread."""
        with self._cond:
            for c in self.conveyors.conveyors:
                c.speed = float(speed)

    def belt_speed(self):
        return self.conveyors.conveyors[0].speed

    # ---------- hooks / interlocks ----------
    def add_hook(self, fn):
        """fn() is called with the world lock held once per round, and about
        10x a second while e-stopped (so safety sensors, lights and spawning
        keep working when the clock is frozen). Use it for anything that
        touches Swift shapes from outside a station thread."""
        with self._cond:
            self._hooks.append(fn)

    def add_interlock(self, fn):
        """fn() returns "" when it is safe to reset/resume, or the reason it is not
        (e.g. "light curtain still blocked")."""
        with self._cond:
            self._interlocks.append(fn)

    def _blocked_by(self):
        for fn in self._interlocks:
            why = fn()
            if why:
                return why
        return ""

    # ---------- e-stop ----------
    def estop(self, reason="E-stop button pressed"):
        """E-stop trip (GUI button, keyboard, light curtain...): halt belts and
        freeze the clock. Station threads block inside tick() until reset()
        AND resume() have both been called. It never clears on its own."""
        with self._cond:
            self.running = False
            self.conveyors.stop()
            if not self.latched:
                self.latched = True
                self.stop_reason = reason
                print(f"[{self.t:6.1f}s] *** E-STOP: {reason} ***")

    def reset(self):
        """Acknowledge the e-stop. Refused while an interlock is still active
        (e.g. someone inside the cell). Does NOT restart anything."""
        with self._cond:
            if not self.latched:
                return True, "Nothing to reset" if self.running else "Already reset - press Resume"
            why = self._blocked_by()
            if why:
                return False, f"Reset refused: {why}"
            self.latched = False
            print(f"[{self.t:6.1f}s] e-stop reset ({self.stop_reason}) - waiting for Resume")
            return True, "Reset OK - press Resume to restart"

    def resume(self):
        """Separate start action after reset(), per the reset-then-start requirement."""
        with self._cond:
            if self.running:
                return True, "Already running"
            if self.latched:
                return False, "Reset the e-stop first"
            why = self._blocked_by()
            if why:
                return False, f"Resume refused: {why}"
            self.running = True
            self.stop_reason = ""
            self.conveyors.start()
            self._cond.notify_all()
            print(f"[{self.t:6.1f}s] resumed")
            return True, "Running"

    def state(self):
        """'running' | 'estop' (latched) | 'reset' (waiting for resume)."""
        if self.running:
            return "running"
        return "estop" if self.latched else "reset"

    def _run_hooks(self):
        for fn in list(self._hooks):
            fn()

    def _pump_while_stopped(self):
        """While e-stopped nothing advances the world, but the safety hooks
        (person walking, stack light) still need to run and Swift still needs
        frames, so redraw ~10x a second without moving the clock. Lock held."""
        now = time.time()
        if now - self._last_pump < 0.1:
            return
        self._last_pump = now
        self._run_hooks()
        self.env.step(0)

    # ---------- stepping ----------
    def _finish_round(self):
        """Advance the world by one dt. Caller holds self._cond."""
        self.t += self.dt
        self._run_hooks()
        self.conveyors.step(self.dt, list(self.active_items))
        self.env.step(self.dt)
        if self.slow_factor < 1.0:   # every robot and belt advances once per round -> all slow down
            time.sleep(self.dt * (1.0 / max(self.slow_factor, 0.05) - 1.0))
        self._waiting = 0
        self._round += 1
        self._cond.notify_all()

    def tick(self):
        """Called by a station once per motion sub-step. Blocks until every
        registered station has called it, then the world advances one dt
        (conveyors + one Swift frame) and all stations are released together."""
        with self._cond:
            while not self.running:                  # e-stopped: wait, don't spin
                self._cond.wait(0.05)
                if not self.running:
                    self._pump_while_stopped()
            self._waiting += 1
            my_round = self._round
            if self._waiting >= max(1, self._n_stations):
                self._finish_round()
                return
            while self._round == my_round:
                self._cond.wait(0.5)                 # timeout so a stalled round can't hang forever
                if self._round == my_round and self._waiting >= max(1, self._n_stations):
                    self._finish_round()