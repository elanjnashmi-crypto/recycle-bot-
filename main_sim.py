import sys
import types

# macOS: the 'keyboard' package (imported by ir_support) crashes Python on
# import (bus error). This project never uses it, so give it a harmless stand-in.
if sys.platform == "darwin" and "keyboard" not in sys.modules:
    class _NoKeyboard(types.ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: False
    sys.modules["keyboard"] = _NoKeyboard("keyboard")

import importlib
import os
import signal
import threading
import time
import traceback

import swift

DT = 0.05
WATCHDOG_PERIOD = 4.0          # s of wall time with no clock progress before a stack dump

# Which UR3e station sits at the end of plastic_line. Only ONE of these can be
# used (they occupy the same spot):
#   "team" -> ur3e_cell.UR3eCell          (the group's version)
#   "mine" -> ur3e_plastic_cell.UR3eCell  (bench + crate version, elbow-up planner)
UR3E_IMPL = "team"

# (display name, module, class). Order = thread start order.
STATIONS = [
    ("Kawasaki infeed", "infeed_cell", "InfeedCell"),
    ("FANUC classifier", "fanuc_cell", "ClassifyCell"),
    ("Yaskawa sorter", "yaskawa_cell", "SortingCell"),
    ("Organic basket", "organic_cell", "OrganicBasketCell"),
    ("UR3e", "ur3e_cell" if UR3E_IMPL == "team" else "ur3e_plastic_cell", "UR3eCell"),
]


def build_stations(world):
    """Import + construct every station; skip (and report) any that fail."""
    stations = []
    for name, module, cls_name in STATIONS:
        try:
            cls = getattr(importlib.import_module(module), cls_name)
            stations.append(cls(world))
            print(f"[main] built: {name}")
        except Exception:
            print(f"[main] FAILED to build {name} ({module}.{cls_name}) -- skipping it. Traceback:")
            traceback.print_exc()
    return stations


def run_station(station, world):
    name = type(station).__name__
    print(f"[main] started: {name}")
    try:
        station.run()
    except Exception:
        print(f"[main] {name} stopped with an exception:")
        traceback.print_exc()
    finally:
        world.unregister()            # don't leave the other stations waiting on a dead one


def watchdog(world, threads):
    """Main-thread loop: if the clock stops advancing, show where each thread is stuck."""
    last_t, last_change = world.t, time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)
        if world.t != last_t:
            last_t, last_change = world.t, time.time()
        elif world.running and time.time() - last_change > WATCHDOG_PERIOD:
            print(f"\n[main] WATCHDOG: world clock stuck at t={world.t:.2f}s for "
                  f">{WATCHDOG_PERIOD:.0f}s. Where each thread is:")
            frames = sys._current_frames()
            for t in threads:
                if t.is_alive() and t.ident in frames:
                    print(f"--- {t.name}")
                    traceback.print_stack(frames[t.ident], limit=6)
            last_change = time.time()      # report again only after another full period
    print("[main] all stations have stopped.")


def main():
    from world import World

    env = swift.Swift()
    try:
        # Building four robots takes a while with no Swift traffic. Newer Swift
        # versions close the browser tab after a few seconds of silence, which
        # makes the next env.add() time out -- so turn that timer off.
        env.launch(realtime=True, browser_timeout=None)
    except TypeError:                 # older Swift without that option
        env.launch(realtime=True)
    world = World(env, dt=DT)
    try:
        from factory import build_factory
        build_factory(env)                # warehouse building around the cell (static)
    except Exception:
        print("[main] factory building failed to build:")
        traceback.print_exc()

    stations = build_stations(world)
    if not stations:
        print("[main] no stations could be built -- nothing to run. See tracebacks above.")
        return

    for _ in stations:
        world.register()              # all registered BEFORE any thread starts, so rounds wait for everyone
    # Safety (light curtain, speed zone, stack light) and the item spawner are
    # added before any station thread starts, so they can still touch Swift directly.
    fanuc = next((s for s in stations if type(s).__name__ == "ClassifyCell"), None)
    safety = spawner = None
    try:
        from safety import SafetySystem
        safety = SafetySystem(world, fanuc)
    except Exception:
        print("[main] safety system failed to build:")
        traceback.print_exc()
    try:
        from item_spawner import ItemSpawner
        spawner = ItemSpawner(world)
    except Exception:
        print("[main] item spawner failed to build:")
        traceback.print_exc()

    threads = [threading.Thread(target=run_station, args=(s, world), daemon=True,
                                name=type(s).__name__) for s in stations]
    for t in threads:
        t.start()

    # Ctrl+C in the terminal quits straight away (Tk would otherwise swallow it).
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    # Control panel on the main thread (Tk needs it on macOS); the watchdog
    # moves to a background thread. Without a display, fall back to the old loop.
    threading.Thread(target=watchdog, args=(world, threads), daemon=True, name="watchdog").start()
    from control_panel import run_panel
    if not run_panel(world, fanuc, safety, spawner):
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()