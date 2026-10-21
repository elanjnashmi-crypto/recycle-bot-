"""
Single entry point for the shared Kerbside workcell.

Creates ONE Swift environment and ONE World (shared conveyors + render
clock), then runs every station concurrently, each in its own thread.
Stations stay independent (each still has its own blocking run() loop,
inherited from how GP7/RS007L's move_* helpers work -- they repeatedly
call a per-substep callback), but all of them now tick the SAME world
instead of each building a private one, so items genuinely flow from the
Kawasaki infeed, along main_line, to wherever the next station is.

Run:
    python main_sim.py

Known limitation (see world.py's tick() docstring): because each station's
blocking move loop calls world.tick() independently, belts/spawners can
advance a bit faster than DT alone suggests when two robots are mid-move at
the same wall-clock moment. Fine for a visual demo; replace with a
fixed-rate scheduler thread if exact timing ever matters.
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

import threading
import traceback

import swift

from world import World
from infeed_cell import InfeedCell
from yaskawa_cell import SortingCell
from ur3e_cell import UR3eCell
from fanuc_cell import ClassifyCell

DT = 0.05


def run_station(station, world):
    try:
        station.run()
    except Exception:                 # keep one station's crash from killing the whole sim
        print(f"[{type(station).__name__}] stopped with an exception:")
        traceback.print_exc()
    finally:
        world.unregister()            # don't leave the other stations waiting on a dead one


def main():
    env = swift.Swift()
    env.launch(realtime=True)

    world = World(env, dt=DT)

    infeed = InfeedCell(world)
    sorter = SortingCell(world)
    ur3e = UR3eCell(world)
    classifier = ClassifyCell(world)

    stations = [infeed, classifier, sorter, ur3e]
    for _ in stations:
        world.register()              # all registered BEFORE any thread starts, so rounds wait for everyone
    threads = [threading.Thread(target=run_station, args=(s, world), daemon=True) for s in stations]
    for t in threads:
        t.start()

    # Main thread just keeps the process alive -- all world.tick() calls
    # happen from the station threads, serialised by World's internal lock.
    # Hook a GUI e-stop button here later: world.estop() / world.reset() /
    # world.resume(), called from whatever thread the GUI runs on.
    threading.Event().wait()


if __name__ == "__main__":
    main()