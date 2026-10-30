import argparse
import json
import sys
import time
from math import pi
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

ROS_HOST, ROS_PORT = "192.168.27.1", 9090              # Raspberry Pi running ROS (lab handout)
JOINT_STATE_TOPIC = "/ur/joint_states"
COMMAND_TOPIC = "/ur/scaled_pos_joint_traj_controller/command"
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

WAYPOINT_FILE = Path("ur3e_waypoints.json")

# ---- limits (edit only if the demonstrator agrees) -------------------------
JOINT_SPEED = 0.30        # rad/s  peak-ish speed of joint moves (slow on purpose)
LINE_SPEED = 0.05         # m/s    straight lifts / lowers
MAX_JOINT_SPEED = 0.8     # hard caps: --speed / --line-speed cannot exceed these
MAX_LINE_SPEED = 0.12
MIN_MOVE_TIME = 1.5       # s, no joint move is ever faster than this
SAMPLE_DT = 0.1           # s between points of a straight-line trajectory
MAX_STEP = 0.12           # rad: largest joint change between two samples of a line (catches IK flips)
STALE_AFTER = 1.0         # s without a joint_states message -> abort
BIG_MOVE = 2.6           # rad (~150 deg): bigger single-joint moves need a typed 'yes'
ARRIVE_TOL = 0.03         # rad: how close counts as "arrived"

# The five taught poses, in the order they are recorded.
POSES = [
    ("wait",        "safe waiting pose, clear of the belt and the crate"),
    ("hover",       "tool pointing DOWN, a few cm above the item at the belt end"),
    ("grasp",       "tool down with the gripper around the item (straight below 'hover')"),
    ("crate_hover", "above the crate, item clear of the rim"),
    ("crate_drop",  "low enough to let go inside the crate (straight below 'crate_hover')"),
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
# Kinematic model (only used to build straight lines; no base needed)
# =====================================================================
def make_model():
    try:
        from ir_support import UR3e                         # UTS package, if installed
        return UR3e()
    except Exception:
        return rtb.models.DH.UR3()                         # UR3 and UR3e share kinematics


# =====================================================================
# Robot back-ends
# =====================================================================
class RosRobot:
    """UR3e through rosbridge (roslibpy)."""

    def __init__(self, host, port):
        import roslibpy
        self._rl = roslibpy
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        if not self.client.is_connected:
            raise RuntimeError(f"Could not connect to rosbridge at {host}:{port} "
                               "(Pi booted? same network? firewall?)")
        self._q = None
        self._stamp = 0.0
        self._sub = roslibpy.Topic(self.client, JOINT_STATE_TOPIC, "sensor_msgs/JointState")
        self._sub.subscribe(self._on_state)
        self._pub = roslibpy.Topic(self.client, COMMAND_TOPIC, "trajectory_msgs/JointTrajectory")
        self._pub.advertise()
        t0 = time.monotonic()
        while self._q is None and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        if self._q is None:
            raise RuntimeError(f"No messages on {JOINT_STATE_TOPIC} after 5 s")

    def _on_state(self, msg):
        # the driver may list joints in a different order: reorder by name
        pos = dict(zip(msg["name"], msg["position"]))
        try:
            self._q = [float(pos[n]) for n in JOINT_NAMES]
            self._stamp = time.monotonic()
        except KeyError:
            pass

    def q(self):
        return list(self._q)

    def stale(self):
        return time.monotonic() - self._stamp > STALE_AFTER

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
def teach(robot):
    """Jog the arm to each pose (pendant), press Enter to store its joint angles."""
    saved = {}
    if WAYPOINT_FILE.exists():
        saved = json.loads(WAYPOINT_FILE.read_text())
        print(f"Existing {WAYPOINT_FILE} found: poses you skip keep their old values.")
    print("\nTEACH MODE. Move the arm to each pose (pendant jog / free-drive), then press Enter.")
    print("Type 's' to skip a pose, 'q' to finish.\n")
    for name, hint in POSES:
        ans = input(f"[{name}] {hint}\n    Enter = RECORD this pose > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        q_now = robot.q()
        if max(abs(v) for v in q_now) < 1e-9:
            print("    NOT saved: every joint reads exactly 0.0, so the robot is not sending real "
                  "data (press PLAY on the pendant; check ON/START). Try again.")
            continue
        saved[name] = {"q": q_now}
        print(f"    saved {name}: q = {np.round(saved[name]['q'], 3).tolist()}")
    WAYPOINT_FILE.write_text(json.dumps(saved, indent=2))
    print(f"\nWrote {WAYPOINT_FILE}")


def load_waypoints(model):
    if not WAYPOINT_FILE.exists():
        sys.exit(f"No {WAYPOINT_FILE}: run  python ur3e_real.py teach  first.")
    wp = json.loads(WAYPOINT_FILE.read_text())
    missing = [n for n, _ in POSES if n not in wp]
    if missing:
        sys.exit(f"Missing taught poses: {missing}. Run teach again.")
    # grasp must be straight below hover and drop straight below crate_hover
    for upper, lower in (("hover", "grasp"), ("crate_hover", "crate_drop")):
        a = model.fkine(wp[upper]["q"]).t
        b = model.fkine(wp[lower]["q"]).t
        off, drop = float(np.hypot(*(a[:2] - b[:2]))), float(a[2] - b[2])
        print(f"  check {upper} -> {lower}: sideways offset {off * 100:.1f} cm, "
              f"lower by {drop * 100:.1f} cm")
        if off > 0.03 or drop <= 0:
            sys.exit(f"'{lower}' should be straight below '{upper}' (sideways offset under 3 cm, "
                     f"and lower). Re-teach those two, or if the numbers above look nonsense "
                     f"(metres off), the joint order may be wrong.")
    return wp


# =====================================================================
# Motion planning (joint space + straight line) and the sequence
# =====================================================================
class Cell:
    def __init__(self, robot, wp, model, jspeed, lspeed, step_mode):
        self.r, self.wp, self.m = robot, wp, model
        self.js, self.ls = jspeed, lspeed
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
        self.r.send(pts, times)
        end = np.array(pts[-1])
        t_end = time.monotonic() + times[-1]
        while time.monotonic() < t_end:                     # sleeping, not a busy loop
            self._ok()
            time.sleep(0.05)
        for _ in range(60):                                 # up to 3 s extra to settle
            self._ok()
            if np.max(np.abs(np.array(self.r.q()) - end)) < ARRIVE_TOL:
                return
            time.sleep(0.05)
        raise RuntimeError("Arm did not reach the target (protective stop? obstruction?) -- stopping.")

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

    # ---- straight line between two taught poses (IK at every sample) ----
    def line(self, name, text=None):
        self._confirm(text or f"straight line to {name}")
        q_a = np.array(self.r.q())
        T_a, T_b = self.m.fkine(q_a), self.m.fkine(self.wp[name]["q"])
        dist = float(np.linalg.norm(T_b.t - T_a.t))
        dur = max(1.0, dist / self.ls)
        n = max(4, int(dur / SAMPLE_DT))
        pts, q = [], q_a.copy()
        for s in np.linspace(0, 1, n + 1)[1:]:
            sol = self.m.ikine_LM(T_a.interp(T_b, s), q0=q, ilimit=40, slimit=1)
            if not sol.success:
                raise RuntimeError(f"Straight line to {name}: IK failed {s:.0%} of the way (out of reach?)")
            qn = q + (sol.q - q + pi) % (2 * pi) - pi
            if np.max(np.abs(qn - q)) > MAX_STEP:
                raise RuntimeError(f"Straight line to {name}: arm would flip posture {s:.0%} of the way")
            q = qn
            pts.append(list(q))
        if np.max(np.abs(q - np.array(self.wp[name]["q"]))) < 0.1:
            pts[-1] = list(self.wp[name]["q"])              # land exactly on the taught pose
        else:
            raise RuntimeError(f"Straight line to {name} ends in a different arm posture than taught")
        self._run_traj(pts, [dur * (i + 1) / n for i in range(n)])

    # ---- the sequence (mirrors UR3eCell.pick_and_store) ----
    def pick_and_store(self):
        self.j("hover", "over the item at the belt end")
        self.line("grasp", "straight down onto the item")
        self._ok(); gripper_close(self.r.client)
        self.line("hover", "straight back up with the item")
        self.j("crate_hover", "swing to the crate")
        self.line("crate_drop", "lower into the crate")
        self._ok(); gripper_open(self.r.client)
        self.line("crate_hover", "straight back up")
        self.j("wait", "return to the wait pose")

    def run(self, cycles, auto_delay):
        d = np.abs(np.array(self.wp["wait"]["q"]) - np.array(self.r.q()))
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
    ap.add_argument("mode", choices=["teach", "run"])
    ap.add_argument("--host", default=ROS_HOST, help="rosbridge host (the Raspberry Pi)")
    ap.add_argument("--port", type=int, default=ROS_PORT)
    ap.add_argument("--mock", action="store_true", help="no robot: rehearse at your desk")
    ap.add_argument("--step", action="store_true", help="ask before every move (use for the first runs)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--auto-delay", type=float, default=None, help="seconds between cycles instead of waiting for Enter")
    ap.add_argument("--speed", type=float, default=JOINT_SPEED, help="joint speed rad/s")
    ap.add_argument("--line-speed", type=float, default=LINE_SPEED, help="straight-line speed m/s")
    a = ap.parse_args()

    jspeed, lspeed = min(a.speed, MAX_JOINT_SPEED), min(a.line_speed, MAX_LINE_SPEED)
    if (jspeed, lspeed) != (a.speed, a.line_speed):
        print(f"Speed capped to {jspeed} rad/s and {lspeed} m/s.")

    robot = MockRobot() if a.mock else RosRobot(a.host, a.port)
    try:
        if a.mode == "teach":
            teach(robot)
        else:
            model = make_model()
            Cell(robot, load_waypoints(model), model, jspeed, lspeed, a.step).run(a.cycles, a.auto_delay)
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