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
import threading
import traceback

import swift

from world import World
from infeed_cell import InfeedCell
from yaskawa_cell import SortingCell
# from fanuc_cell import ClassifyCell   # add once that teammate's station exists

DT = 0.05


def run_station(station):
    try:
        station.run()
    except Exception:                 # keep one station's crash from killing the whole sim
        print(f"[{type(station).__name__}] stopped with an exception:")
        traceback.print_exc()


def main():
    env = swift.Swift()
    env.launch(realtime=True)

    world = World(env, dt=DT)

    infeed = InfeedCell(world)
    sorter = SortingCell(world)
    # classifier = ClassifyCell(world)

    stations = [infeed, sorter]
    threads = [threading.Thread(target=run_station, args=(s,), daemon=True) for s in stations]
    for t in threads:
        t.start()

    # Main thread just keeps the process alive -- all world.tick() calls
    # happen from the station threads, serialised by World's internal lock.
    # Hook a GUI e-stop button here later: world.estop() / world.reset() /
    # world.resume(), called from whatever thread the GUI runs on.
    threading.Event().wait()


if __name__ == "__main__":
    main()