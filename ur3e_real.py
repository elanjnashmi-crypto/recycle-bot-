import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb

ROS_HOST, ROS_PORT = "192.168.27.1", 9090              # Raspberry Pi running ROS (lab handout)
JOINT_STATE_TOPIC = "/ur/joint_states"
COMMAND_TOPIC = "/ur/scaled_pos_joint_traj_controller/command"
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

WAYPOINT_FILE = Path("ur3e_waypoints.json")

# ---- limits (edit only if the demonstrator agrees) -------------------------
JOINT_SPEED = 0.30        # rad/s  peak-ish speed of joint moves (slow on purpose)
MAX_JOINT_SPEED = 0.8     # hard cap: --speed cannot exceed this
MIN_MOVE_TIME = 1.5       # s, no joint move is ever faster than this
SAMPLE_DT = 0.1           # s between points of a trajectory
STALE_AFTER = 1.0         # s without a joint_states message -> abort
BIG_MOVE = 2.6           # rad (~150 deg): bigger single-joint moves need a typed 'yes'
ARRIVE_TOL = 0.03         # rad: how close counts as "arrived"
STALL_TIME = 3.0          # s with no progress towards the target -> abort

# The five taught waypoints, in the order they are recorded. The arm moves joint-to-joint
# between them exactly as taught -- no straight lines, no position checks.
POSES = [
    ("wait",        "safe waiting pose, clear of the belt and the crate"),
    ("hover",       "a few cm above the item at the belt end"),
    ("grasp",       "gripper around the item"),
    ("crate_hover", "above the crate, item clear of the rim"),
    ("crate_drop",  "low enough to let go inside the crate"),
]


# =====================================================================
# Gripper -- OnRobot gripper through two std_srvs/Trigger services
# (taken from simple_gripper_control.py)
# =====================================================================
GRIPPER_OPEN_SERVICE = "/onrobot/open"
GRIPPER_CLOSE_SERVICE = "/onrobot/close"
GRIPPER_TIME = 2.0        # s to let the fingers finish (the lab script waits 2 s)


def _gripper_call(client, service_name):
    """Call a Trigger service; return True if the gripper reported success."""
    if client is None:                                      # --mock
        print(f"    (mock) gripper service {service_name}")
        return True
    import roslibpy
    service = roslibpy.Service(client, service_name, "std_srvs/Trigger")
    result = service.call(roslibpy.ServiceRequest({}))
    print(f"    gripper {service_name}: success={result['success']} '{result['message']}'")
    time.sleep(GRIPPER_TIME)
    return bool(result["success"])


def gripper_close(client):
    if not _gripper_call(client, GRIPPER_CLOSE_SERVICE):
        raise RuntimeError("Gripper did not close -- stopping before lifting.")


def gripper_open(client):
    if not _gripper_call(client, GRIPPER_OPEN_SERVICE):
        raise RuntimeError("Gripper did not open -- stopping.")


# =====================================================================
# Robot back-ends
# =====================================================================
class RosRobot:
    """UR3e through rosbridge (roslibpy)."""

    def __init__(self, host, port, topic=JOINT_STATE_TOPIC):
        import roslibpy
        self._rl = roslibpy
        self.topic = topic
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        if not self.client.is_connected:
            raise RuntimeError(f"Could not connect to rosbridge at {host}:{port} "
                               "(Pi booted? same network? firewall?)")
        self._q = None              # latest real (not all-zero) reading
        self._stamp = 0.0
        self._zero_stamp = 0.0      # last all-zero message (a placeholder publisher, not the arm)
        self._msgs = self._zero_msgs = self._skipped = 0
        self._names_seen = None
        self._sub = roslibpy.Topic(self.client, topic, "sensor_msgs/JointState")
        self._sub.subscribe(self._on_state)
        self._pub = roslibpy.Topic(self.client, COMMAND_TOPIC, "trajectory_msgs/JointTrajectory")
        self._pub.advertise()
        t0 = time.monotonic()
        while self._q is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        if self._q is None:
            self.diagnose()
            raise RuntimeError(f"No real joint angles on {topic} after 5 s (see the report above)")

    @staticmethod
    def _match_names(names):
        """Index of each of JOINT_NAMES in the message, allowing a prefix such as
        'ur3e_shoulder_pan_joint' or 'ur/shoulder_pan_joint'. None if any joint is missing."""
        idx = []
        for want in JOINT_NAMES:
            hits = [i for i, n in enumerate(names) if n == want or n.endswith("_" + want) or n.endswith("/" + want)]
            if len(hits) != 1:
                return None
            idx.append(hits[0])
        return idx

    def _on_state(self, msg):
        self._msgs += 1
        self._names_seen = msg["name"]
        idx = self._match_names(msg["name"])
        if idx is None or len(msg["position"]) < len(msg["name"]):
            self._skipped += 1
            return
        q = [float(msg["position"][i]) for i in idx]
        if max(abs(v) for v in q) < 1e-9:
            self._zero_msgs += 1
            self._zero_stamp = time.monotonic()
            return                  # never let a placeholder overwrite the arm's real angles
        self._q = q
        self._stamp = time.monotonic()

    def q(self):
        if self._q is None:
            raise RuntimeError("No real joint angles received from the arm")
        return list(self._q)

    def stale(self):
        return time.monotonic() - self._stamp > STALE_AFTER

    def fresh_q(self, timeout=1.0):
        """Joint angles from a message that arrived after this call (not a cached one)."""
        t0 = time.monotonic()
        while self._stamp <= t0:
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"No new real joint angles within {timeout:.0f} s")
            time.sleep(0.01)
        return self.q()

    def diagnose(self):
        """What is actually on the topic -- printed when the readings look wrong."""
        print(f"\n--- joint_states report ({self.topic}) ---")
        print(f"  messages received: {self._msgs}   all-zero: {self._zero_msgs}   "
              f"skipped (joint names didn't match): {self._skipped}")
        print(f"  joint names in the last message: {self._names_seen}")
        try:
            srv = self._rl.Service(self.client, "/rosapi/topics", "rosapi/Topics")
            res = srv.call(self._rl.ServiceRequest({}), timeout=3)
            js = [t for t, ty in zip(res["topics"], res["types"]) if ty == "sensor_msgs/JointState"]
            print(f"  JointState topics on the Pi: {js}")
            for t in js:
                pubs = self._rl.Service(self.client, "/rosapi/publishers", "rosapi/Publishers").call(
                    self._rl.ServiceRequest({"topic": t}), timeout=3)["publishers"]
                print(f"    {t}: published by {pubs}")
        except Exception as exc:
            print(f"  (couldn't list topics through rosapi: {exc})")
        if self._zero_msgs and self._q is None:
            print("  Only all-zero messages: the UR driver isn't streaming the arm's angles. Check the\n"
                  "  driver is running on the Pi and connected to the robot (robot ON, brakes released,\n"
                  "  External Control program playing), or try another topic with --joint-topic.")
        print("---------------------------------------------\n")

    # ---- starting the External Control program without the pendant ----
    def _find_service(self, suffix):
        """Full name of the driver service ending in `suffix` (the namespace differs per setup)."""
        if not hasattr(self, "_services"):
            try:
                srv = self._rl.Service(self.client, "/rosapi/services", "rosapi/Services")
                self._services = srv.call(self._rl.ServiceRequest({}), timeout=3)["services"]
            except Exception as exc:
                print(f"    (couldn't list services through rosapi: {exc})")
                self._services = []
        hits = [s for s in self._services if s.endswith(suffix)]
        return hits[0] if hits else None

    def _call(self, name, srv_type):
        return self._rl.Service(self.client, name, srv_type).call(self._rl.ServiceRequest({}), timeout=5)

    def _program_running(self):
        name = self._find_service("dashboard/program_running")
        if name is None:
            return None                                     # unknown
        try:
            return bool(self._call(name, "ur_dashboard_msgs/IsProgramRunning")["program_running"])
        except Exception:
            return None

    def start_program(self):
        """Press 'play' on the External Control program through the UR driver (what the pendant's
        play button does), so the arm accepts trajectories. Needs the robot in Remote Control mode."""
        if self._program_running():
            print("External Control program is running.")
            return
        print("External Control program not running -- starting it from here...")
        tried = []
        for suffix, srv_type in (("dashboard/play", "std_srvs/Trigger"),
                                 ("resend_robot_program", "std_srvs/Trigger")):
            name = self._find_service(suffix)
            if name is None:
                continue
            tried.append(name)
            try:
                res = self._call(name, srv_type)
                print(f"    {name}: success={res['success']} '{res['message']}'")
                connect = self._find_service("dashboard/connect")
                if not res["success"] and "reconnect" in res["message"].lower() and connect:
                    # the driver's dashboard link was opened in Local mode; the robot only honours
                    # a link opened after switching to Remote, so reopen it and try again
                    print(f"    reconnecting the dashboard ({connect}) and retrying...")
                    self._call(connect, "std_srvs/Trigger")
                    time.sleep(1.0)
                    res = self._call(name, srv_type)
                    print(f"    {name}: success={res['success']} '{res['message']}'")
            except Exception as exc:
                print(f"    {name}: failed ({exc})")
                continue
            for _ in range(50):                             # up to 5 s for the program to come up
                if self._program_running():
                    time.sleep(1.0)                         # let the driver reconnect to the arm
                    print("External Control program is running.")
                    return
                time.sleep(0.1)
            if self._program_running() is None and res.get("success"):
                time.sleep(2.0)                             # can't check; trust the driver's answer
                return
        if not tried:
            raise RuntimeError("The UR driver on the Pi offers no dashboard/play or resend_robot_program "
                               "service, so the program has to be started on the pendant.")
        raise RuntimeError("Couldn't start the External Control program remotely. Put the robot in "
                           "Remote Control mode on the pendant (top-right corner, once) and clear any "
                           "protective stop, then run again.")

    def check(self):
        """Read-only health report of the robot -> driver -> controller chain. Moves nothing."""
        def ask(suffix, srv_type):
            name = self._find_service(suffix)
            if name is None:
                return "(service not offered by the driver)"
            try:
                res = self._rl.Service(self.client, name, srv_type).call(self._rl.ServiceRequest({}), timeout=5)
            except Exception as exc:
                return f"(call failed: {exc})"
            return res

        print("\n=== UR3e chain check (read-only, moves nothing) ===")
        print(f"1. Joint angles: {np.round(np.degrees(self.q()), 1).tolist()} deg, "
              f"{'STALE' if self.stale() else 'live'}")
        print("2. Robot, as its dashboard reports it:")
        for suffix, srv_type in (("dashboard/get_robot_mode", "ur_dashboard_msgs/GetRobotMode"),
                                 ("dashboard/get_safety_mode", "ur_dashboard_msgs/GetSafetyMode"),
                                 ("dashboard/is_in_remote_control", "ur_dashboard_msgs/IsInRemoteControl"),
                                 ("dashboard/get_loaded_program", "ur_dashboard_msgs/GetLoadedProgram"),
                                 ("dashboard/program_state", "ur_dashboard_msgs/GetProgramState"),
                                 ("dashboard/program_running", "ur_dashboard_msgs/IsProgramRunning")):
            res = ask(suffix, srv_type)
            print(f"   {suffix.split('/')[-1]:22s} {res.get('answer', res) if isinstance(res, dict) else res}")
        topics = self._rl.Service(self.client, "/rosapi/topics", "rosapi/Topics").call(
            self._rl.ServiceRequest({}), timeout=3)["topics"]
        print("3. External Control link (driver's robot_program_running topic, must be True to move):")
        rp = [t for t in topics if t.endswith("robot_program_running")]
        if rp:
            got = []
            sub = self._rl.Topic(self.client, rp[0], "std_msgs/Bool")
            sub.subscribe(lambda m: got.append(m["data"]))
            time.sleep(1.5)
            sub.unsubscribe()
            print(f"   {rp[0]}: {got[-1] if got else '(no message)'}")
        else:
            print("   (topic not found)")
        print("4. Controllers on the Pi:")
        res = ask("controller_manager/list_controllers", "controller_manager_msgs/ListControllers")
        if isinstance(res, dict):
            for c in res.get("controller", []):
                print(f"   {c['name']:45s} {c['state']}")
        else:
            print(f"   {res}")
        print(f"5. Who listens on {COMMAND_TOPIC}:")
        try:
            subs = self._rl.Service(self.client, "/rosapi/subscribers", "rosapi/Subscribers").call(
                self._rl.ServiceRequest({"topic": COMMAND_TOPIC}), timeout=3)["subscribers"]
            print(f"   {subs or 'NOBODY -- commands sent there go nowhere'}")
        except Exception as exc:
            print(f"   (couldn't ask: {exc})")
        print(f"   trajectory command topics on the Pi: "
              f"{[t for t in topics if t.endswith('/command') and 'traj' in t]}")
        print("===================================================\n")

    def send(self, points, times):
        """points: list of 6-joint lists; times: seconds from now for each."""
        msg = {"joint_names": JOINT_NAMES,
               "points": [{"positions": [float(v) for v in p],
                           "time_from_start": {"secs": int(t), "nsecs": int((t % 1) * 1e9)}}
                          for p, t in zip(points, times)]}
        self._pub.publish(self._rl.Message(msg))

    def hold(self):
        """Stop where we are."""
        try:
            self.send([self.q()], [0.5])
        except Exception:
            pass

    def close(self):
        self.hold()
        try:
            self._sub.unsubscribe()
            self._pub.unadvertise()
            self.client.terminate()
        except Exception:
            pass


class MockRobot:
    """Pretends to be a robot so the whole sequence can be rehearsed at a desk."""

    def __init__(self):
        self._q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        self.client = None

    def q(self):
        return list(self._q)

    def fresh_q(self, timeout=1.0):
        return self.q()

    def start_program(self):
        print("    (mock) External Control program started")

    def diagnose(self):
        print("    (mock) no topic to report on")

    def stale(self):
        return False

    def send(self, points, times):
        d = float(np.max(np.abs(np.array(points[-1]) - np.array(self._q))))
        print(f"    (mock) trajectory: {len(points)} points, {times[-1]:.1f} s, largest joint change {d:.2f} rad")
        self._q = list(points[-1])

    def hold(self):
        pass

    def close(self):
        pass


# =====================================================================
# Teaching
# =====================================================================
def _live_key(robot):
    """Show the arm's joint angles live (so free-driving visibly changes them) until a key is
    pressed; return that key: '' for Enter, 's', 'q' or 'd'. Windows console; elsewhere input()."""
    try:
        import msvcrt
    except ImportError:
        return input("    Enter = record, s = skip, q = finish, d = report > ").strip().lower()
    while True:
        try:
            deg = np.degrees(robot.q())
            line = "    live q (deg): " + " ".join(f"{v:+7.1f}" for v in deg)
            if robot.stale():
                line += "   [NO FRESH DATA]"
        except RuntimeError:
            line = "    live q: waiting for real joint angles..."
        print("\r" + line.ljust(78), end="", flush=True)
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            print()
            return "" if ch in "\r\n" else ch.lower()
        time.sleep(0.1)


def teach(robot):
    """Free-drive / jog the arm to each pose, press Enter to store its joint angles."""
    saved = {}
    if WAYPOINT_FILE.exists():
        saved = json.loads(WAYPOINT_FILE.read_text())
        print(f"Existing {WAYPOINT_FILE} found: poses you skip keep their old values.")
    print("\nTEACH MODE. Move the arm to each pose (free-drive or pendant jog); the live joint angles")
    print("below must change as it moves. Enter = record, s = skip, q = finish, d = topic report.\n")
    finished = False
    for name, hint in POSES:
        print(f"[{name}] {hint}")
        while True:
            key = _live_key(robot)
            if key == "d":
                robot.diagnose()
                continue
            if key in ("q", "s"):
                break
            if key != "":
                continue
            try:
                q_now = robot.fresh_q()         # a reading taken after the key press, not a cached one
            except RuntimeError as exc:
                print(f"    NOT saved: {exc}. Press 'd' for a report of what the Pi is sending.")
                continue
            saved[name] = {"q": q_now}
            print(f"    saved {name}: q (deg) = {np.round(np.degrees(q_now), 1).tolist()}")
            break
        if key == "q":
            finished = True
        if finished:
            break
    WAYPOINT_FILE.write_text(json.dumps(saved, indent=2))
    print(f"\nWrote {WAYPOINT_FILE}")


def load_waypoints():
    if not WAYPOINT_FILE.exists():
        sys.exit(f"No {WAYPOINT_FILE}: run  python ur3e_real.py teach  first.")
    wp = json.loads(WAYPOINT_FILE.read_text())
    missing = [n for n, _ in POSES if n not in wp]
    if missing:
        sys.exit(f"Missing taught poses: {missing}. Run teach again.")
    return wp


# =====================================================================
# Motion planning (joint space between taught waypoints) and the sequence
# =====================================================================
class Cell:
    def __init__(self, robot, wp, jspeed, step_mode):
        self.r, self.wp = robot, wp
        self.js = jspeed
        self.step_mode = step_mode

    # ---- checks ----
    def _ok(self):
        if self.r.stale():
            raise RuntimeError("Lost the joint_states feed -- stopping.")

    def _confirm(self, text):
        self._ok()
        print(f"  -> {text}")
        if self.step_mode and input("     Enter = go, q = quit > ").strip().lower() == "q":
            raise KeyboardInterrupt

    # ---- executing a timed trajectory ----
    def _run_traj(self, pts, times):
        start = np.array(self.r.q())
        self.r.send(pts, times)
        end = np.array(pts[-1])
        t_end = time.monotonic() + times[-1]
        while time.monotonic() < t_end:                     # sleeping, not a busy loop
            self._ok()
            time.sleep(0.05)
        # The scaled controller runs slower than planned when the pendant speed slider is below
        # 100%, so keep waiting while the arm is still closing in; give up once it stops moving.
        best, last_progress = np.inf, time.monotonic()
        while time.monotonic() - last_progress < STALL_TIME:
            self._ok()
            err = float(np.max(np.abs(np.array(self.r.q()) - end)))
            if err < ARRIVE_TOL:
                return
            if err < best - 0.005:
                best, last_progress = err, time.monotonic()
            time.sleep(0.05)
        now = np.array(self.r.q())
        moved = float(np.degrees(np.max(np.abs(now - start))))
        print(f"     start  (deg) {np.round(np.degrees(start), 1).tolist()}")
        print(f"     target (deg) {np.round(np.degrees(end), 1).tolist()}")
        print(f"     now    (deg) {np.round(np.degrees(now), 1).tolist()}")
        if moved < 1.0:
            raise RuntimeError("The arm never moved: the command isn't being executed. Check on the pendant: "
                               "robot in Remote Control, External Control program running, no protective "
                               "stop, speed slider above 0%.")
        raise RuntimeError(f"Arm moved {moved:.0f} deg then stopped short of the target "
                           f"(protective stop? obstruction? joint limit?) -- stopping.")

    # ---- joint move: quintic blend, slow ----
    def j(self, name, text=None):
        self._confirm(text or f"joint move to {name}")
        q0, q1 = np.array(self.r.q()), np.array(self.wp[name]["q"])
        big = float(np.max(np.abs(q1 - q0)))
        if big > BIG_MOVE:                                  # a joint swinging this far is unusual: make a human look
            print(f"     WARNING: a joint will turn {np.degrees(big):.0f} deg to reach '{name}'.")
            if input("     Type 'yes' to allow > ").strip().lower() != "yes":
                raise KeyboardInterrupt
        dur = max(MIN_MOVE_TIME, 1.875 * float(np.max(np.abs(q1 - q0))) / self.js)
        n = max(8, int(dur / SAMPLE_DT))
        traj = rtb.jtraj(q0, q1, n).q[1:]
        times = [dur * (i + 1) / n for i in range(n)]
        self._run_traj([list(q) for q in traj], times)

    # ---- the sequence (mirrors UR3eCell.pick_and_store) ----
    def pick_and_store(self):
        self.j("hover", "over the item at the belt end")
        self.j("grasp", "down onto the item")
        self._ok(); gripper_close(self.r.client)
        self.j("hover", "back up with the item")
        self.j("crate_hover", "swing to the crate")
        self.j("crate_drop", "lower into the crate")
        self._ok(); gripper_open(self.r.client)
        self.j("crate_hover", "back up")
        self.j("wait", "return to the wait pose")

    def run(self, cycles, auto_delay):
        self.r.start_program()                  # no need to press play on the pendant
        q_now = self.r.fresh_q()                # never plan from a cached or placeholder reading
        if max(abs(v) for v in q_now) < 1e-9 or self.r.stale():
            raise RuntimeError("Joint angles aren't coming from the arm -- refusing to move.")
        print(f"Arm is at q (deg) = {np.round(np.degrees(q_now), 1).tolist()}  <- check this matches the pendant")
        d = np.abs(np.array(self.wp["wait"]["q"]) - np.array(q_now))
        print(f"Moving to the wait pose first: largest joint change {np.degrees(d.max()):.0f} deg.")
        input("Workspace clear and hand on the e-stop? Enter to go > ")
        self.j("wait", "go to the wait pose")
        for n in range(1, cycles + 1):
            if auto_delay is None:
                input(f"\nCycle {n}/{cycles}: put the item at the belt end, then press Enter > ")
            else:
                print(f"\nCycle {n}/{cycles}: starting in {auto_delay:.0f} s (put the item at the belt end)")
                time.sleep(auto_delay)
            self.pick_and_store()
        print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description="Real UR3e pick-and-place for the Kerbside demo")
    ap.add_argument("mode", choices=["teach", "run", "check"])
    ap.add_argument("--host", default=ROS_HOST, help="rosbridge host (the Raspberry Pi)")
    ap.add_argument("--port", type=int, default=ROS_PORT)
    ap.add_argument("--joint-topic", default=JOINT_STATE_TOPIC, help="sensor_msgs/JointState topic to read")
    ap.add_argument("--mock", action="store_true", help="no robot: rehearse at your desk")
    ap.add_argument("--step", action="store_true", help="ask before every move (use for the first runs)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--auto-delay", type=float, default=None, help="seconds between cycles instead of waiting for Enter")
    ap.add_argument("--speed", type=float, default=JOINT_SPEED, help="joint speed rad/s")
    a = ap.parse_args()

    jspeed = min(a.speed, MAX_JOINT_SPEED)
    if jspeed != a.speed:
        print(f"Speed capped to {jspeed} rad/s.")

    robot = MockRobot() if a.mock else RosRobot(a.host, a.port, a.joint_topic)
    try:
        if a.mode == "teach":
            teach(robot)
        elif a.mode == "check":
            robot.check()
        else:
            Cell(robot, load_waypoints(), jspeed, a.step).run(a.cycles, a.auto_delay)
    except KeyboardInterrupt:
        print("\nStopped by user -- holding position.")
        robot.hold()
    except Exception as exc:
        print(f"\nABORTED: {exc}")
        robot.hold()
    finally:
        robot.close()


if __name__ == "__main__":
    main()