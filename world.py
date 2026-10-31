import os
import sys
import threading
import time

import swift

from conveyors import ConveyorSystem


def _one_swift_tab():

    if route is None or getattr(route.SwiftSocket, "_one_tab", False):
        return
    serve = route.SwiftSocket.serve

    async def one_tab_serve(self, websocket, path=None):
        if getattr(self, "_client", None) is not None:
            print("[world] a second Swift tab tried to connect - closed it (keep only the first tab)")
            await websocket.close()
            return
        self._client = websocket
        try:
            return await serve(self, websocket, path)
        except Exception:
            pass
        finally:
            print("[world] Swift window closed - simulation stopped.", flush=True)
            os._exit(0)

    route.SwiftSocket.serve = one_tab_serve
    route.SwiftSocket._one_tab = True


_one_swift_tab()


class World:
    def __init__(self, env, dt=0.05):
        self.env = env
        self.dt = dt
        self.conveyors = ConveyorSystem(env)
        self.t = 0.0
        self.running = True #single e-stop flag for the whole workcell
        self.auto_feed = True
        self.pick_requests = []
        self.infeed_status = ""      # what the Kawasaki is doing, shown on the control panel
        self.slow_factor = 1.0       #<1 slows the WHOLE cell in real time (standing in front of light barrier)
        self.sim_speed = 1.0         # GUI simulation speed setter, 1 = real time, 5 = speed*5
        self._pace_wall = None       # wall clock /sim time the pacing is measured from
        self._pace_sim = 0.0
        self._pace_speed = 1.0
        # World paces the clock itself
        if not getattr(env, "headless", False):
            env.realtime = False
            env.rate = 30
        self.obstacles = []
        self.active_items = [] #shapes currently on conveyors, used to detect collisions with the robot
        self.item_info = {}
        self._cond = threading.Condition(threading.RLock())
        self._n_stations = 0         
        self._waiting = 0            
        self._round = 0              
        # e-stop state
        self.latched = False
        self.stop_reason = ""
        self._interlocks = []   
        self._hooks = []   #run once per round AND while e-stopped (lock held)
        self._last_pump = 0.0

    
    def register(self):
        with self._cond:
            self._n_stations += 1

    def unregister(self):
        with self._cond:
            self._n_stations = max(0, self._n_stations - 1)
            if self._waiting >= max(1, self._n_stations) and self._waiting > 0:
                self._finish_round()
            self._cond.notify_all()

    
    def add(self, shape):
        with self._cond:
            self.env.add(shape)

    
    def tag_item(self, shape, **info):
        with self._cond:
            self.item_info[id(shape)] = dict(info)

    def get_item_info(self, shape):
        # what the spawner knows about the item
        with self._cond:
            return self.item_info.get(id(shape))

    def add_active(self, shape):
        with self._cond:
            if shape not in self.active_items:
                self.active_items.append(shape)

    def remove_active(self, shape):
        with self._cond:
            if shape in self.active_items:
                self.active_items.remove(shape)

    #conveyors speed
    def set_belt_speed(self, speed):
        with self._cond:
            for c in self.conveyors.conveyors:
                c.speed = float(speed)

    def set_sim_speed(self, speed):
        #sim speed setter to allow the GUI to change the speed of the simulation
        with self._cond:
            self.sim_speed = max(0.1, float(speed))

    def belt_speed(self):
        return self.conveyors.conveyors[0].speed

    #manual item spawning
    def request_pick(self, kind):
        self.pick_requests.append(kind)

    
    def add_hook(self, fn):
        with self._cond:
            self._hooks.append(fn)

    def add_interlock(self, fn):
        with self._cond:
            self._interlocks.append(fn)

    def _blocked_by(self):
        for fn in self._interlocks:
            why = fn()
            if why:
                return why
        return ""

    
    def estop(self, reason="E-stop button pressed"):
        with self._cond:
            self.running = False
            self.conveyors.stop()
            if not self.latched:
                self.latched = True
                self.stop_reason = reason
                print(f"[{self.t:6.1f}s] *** E-STOP: {reason} ***")

    def reset(self):
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
        #seperate from reset, so that a station can refuse to resume if it is not ready yet
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
        #running, estop, reset, resume
        if self.running:
            return "running"
        return "estop" if self.latched else "reset"

    def _run_hooks(self):
        for fn in list(self._hooks):
            fn()

    def _pump_while_stopped(self):
        now = time.time()
        if now - self._last_pump < 0.1:
            return
        self._last_pump = now
        self._run_hooks()
        self.env.step(0)

    
    def _finish_round(self):
        """Advance the world by one dt. Caller holds self._cond."""
        self.t += self.dt
        self._run_hooks()
        self.conveyors.step(self.dt, list(self.active_items))
        self.env.step(self.dt)
        self._pace()
        self._waiting = 0
        self._round += 1
        self._cond.notify_all()

    def _pace(self):
        # pace the simulation wall clock to match the sim time, scaled by self.sim_speed * self.slow_factor
        if getattr(self.env, "headless", False):
            return
        speed = self.sim_speed * max(self.slow_factor, 0.05)
        now = time.time()
        if self._pace_wall is None or speed != self._pace_speed:
            self._pace_wall, self._pace_sim, self._pace_speed = now, self.t, speed
            return
        target = self._pace_wall + (self.t - self._pace_sim) / speed
        if target > now:
            time.sleep(target - now)
        elif now - target > 0.25:                
            self._pace_wall, self._pace_sim = now, self.t

    def tick(self):
        with self._cond:
            while not self.running: # e-stopped dont spin
                self._cond.wait(0.05)
                if not self.running:
                    self._pump_while_stopped()
            self._waiting += 1
            my_round = self._round
            if self._waiting >= max(1, self._n_stations):
                self._finish_round()
                return
            while self._round == my_round:
                self._cond.wait(0.5)                 
                if self._round == my_round and self._waiting >= max(1, self._n_stations):
                    self._finish_round()