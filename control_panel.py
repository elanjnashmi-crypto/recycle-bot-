"""
Control panel (tkinter) for the recycle-bot workcell.

    E-STOP / Reset / Resume        (Esc also trips the e-stop)
    Conveyor belt speed
    Item feed: Auto (Kawasaki picks whatever is next in its tub) or
               Manual (you choose which item the Kawasaki picks from the tub)
    Collision test: drop an obstacle into the FANUC's path, or clear it
    FANUC teach / jog: Auto or Manual; in Manual, jog each joint, jog the TCP in
               x/y/z, type a joint state and Go (planned, collision-checked),
               gripper, wait pose; live state and fault messages

Runs on the MAIN thread (macOS only allows Tk there); the stations run in
their own threads. The panel never touches Swift itself: it only sets flags
and queues requests, which the World / station threads carry out. So it keeps
working during an e-stop, when the simulation clock is frozen.
"""
import os
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    from infeed_cell import RUBBISH_ITEMS
    ITEM_KINDS = list(RUBBISH_ITEMS)
except Exception:                   # infeed failed to import: manual feed has nothing to offer
    ITEM_KINDS = []

REFRESH_MS = 100
BELT_SPEED_RANGE = (0.05, 0.25)    # m/s; above ~0.25 the FANUC can't catch items in its reach
MODE_TEXT = {"auto": "AUTO", "manual": "MANUAL", "to_auto": "returning to auto..."}
STATE_TEXT = {
    "running": ("RUNNING", "#1b8a2e"),
    "estop": ("E-STOP", "#c62828"),
    "reset": ("RESET - press Resume", "#e08a00"),
}


class ControlPanel:
    def __init__(self, root, world, fanuc=None):
        self.root, self.world, self.fanuc = root, world, fanuc
        root.title("Recycle-bot control panel")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", lambda: os._exit(0))
        root.bind("<Escape>", lambda e: self._estop())
        self.safety = None              # set by run_panel; keys 1/2/3 move the test person
        for key, where in (("1", "outside"), ("2", "warning zone"), ("3", "inside cell")):
            root.bind(key, lambda e, w=where: self._person(w))

        self.msg = tk.StringVar(value="")
        self.left = ttk.Frame(root, padding=8)
        self.left.grid(row=0, column=0, sticky="n")
        self._build_estop(self.left)
        self._build_belts(self.left)
        self._build_feed(self.left)
        self._build_collision(self.left)
        self.right = ttk.Frame(root, padding=8)
        self.right.grid(row=0, column=1, sticky="n")
        self._build_jog(self.right)
        ttk.Label(root, textvariable=self.msg, padding=(8, 0, 8, 8), foreground="#555",
                  wraplength=700).grid(row=1, column=0, columnspan=2, sticky="w")
        self._refresh()

    # ---------- layout ----------
    def _build_estop(self, parent):
        box = ttk.LabelFrame(parent, text="Emergency stop", padding=8)
        box.pack(fill="x")
        # A Label, not a Button: macOS ignores Button background colours.
        btn = tk.Label(box, text="E-STOP", font=("Helvetica", 22, "bold"), fg="white", bg="#c62828",
                       width=12, height=2, relief="raised", bd=4, cursor="hand2")
        btn.pack(pady=(0, 6))
        btn.bind("<Button-1>", lambda e: self._estop())
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Button(row, text="Reset", command=self._reset).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Resume", command=self._resume).pack(side="left", expand=True, fill="x")
        self.state_lbl = tk.Label(box, text="", font=("Helvetica", 13, "bold"), fg="white", pady=4)
        self.state_lbl.pack(fill="x", pady=(6, 0))
        self.reason = tk.StringVar()
        ttk.Label(box, textvariable=self.reason, wraplength=260).pack(fill="x")
        ttk.Label(box, text="Esc = e-stop    test person: 1 out, 2 zone, 3 in", foreground="#777").pack(anchor="w")

    def _build_belts(self, parent):
        box = ttk.LabelFrame(parent, text="Conveyor belt speed (m/s)", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.belt_speed = tk.DoubleVar(value=self.world.belt_speed())
        tk.Scale(box, from_=BELT_SPEED_RANGE[0], to=BELT_SPEED_RANGE[1], resolution=0.01,
                 orient="horizontal", length=260, variable=self.belt_speed,
                 command=lambda v: self.world.set_belt_speed(float(v))).pack()

    def _build_feed(self, parent):
        box = ttk.LabelFrame(parent, text="Item feed", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.feed_var = tk.StringVar(value="auto" if self.world.auto_feed else "manual")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="Auto", value="auto", variable=self.feed_var,
                        command=self._feed).pack(side="left")
        ttk.Radiobutton(row, text="Manual", value="manual", variable=self.feed_var,
                        command=self._feed).pack(side="left", padx=8)
        self.pick_box = ttk.Frame(box)
        self.pick_box.pack(fill="x", pady=(6, 0))
        ttk.Label(self.pick_box, text="Item for the Kawasaki to pick from the tub:").pack(anchor="w")
        self.kind = tk.StringVar(value=ITEM_KINDS[0] if ITEM_KINDS else "")
        self.pick_widgets = [
            ttk.Combobox(self.pick_box, textvariable=self.kind, values=ITEM_KINDS,
                         state="readonly", width=30),
            ttk.Button(self.pick_box, text="Pick this item", command=self._pick),
        ]
        for w in self.pick_widgets:
            w.pack(fill="x", pady=2)
        self.feed_status = tk.StringVar()
        ttk.Label(self.pick_box, textvariable=self.feed_status, foreground="#555",
                  wraplength=260).pack(anchor="w")
        self._feed()
        self.msg.set("")

    def _build_collision(self, parent):
        box = ttk.LabelFrame(parent, text="Collision test (FANUC)", padding=8)
        box.pack(fill="x", pady=(8, 0))
        if self.fanuc is None:
            ttk.Label(box, text="FANUC cell not running").pack()
            self.collision_status = None
            return
        spots = list(self.fanuc.obstacle_spots)
        self.spot = tk.StringVar(value=spots[0])
        ttk.Combobox(box, textvariable=self.spot, values=spots, state="readonly", width=30).pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Drop obstacle", command=self._drop_obstacle).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Clear obstacles", command=self.fanuc.request_clear_obstacles).pack(
            side="left", expand=True, fill="x")
        self.collision_status = tk.StringVar()
        ttk.Label(box, textvariable=self.collision_status, foreground="#555", wraplength=260).pack(anchor="w")

    def _build_jog(self, parent):
        box = ttk.LabelFrame(parent, text="FANUC teach / jog", padding=8)
        box.pack(fill="x")
        self.jog_widgets = []
        if self.fanuc is None:
            ttk.Label(box, text="FANUC cell not running").pack()
            return
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(row, text="Auto", value="auto", variable=self.mode_var,
                        command=self._mode).pack(side="left")
        ttk.Radiobutton(row, text="Manual (teach)", value="manual", variable=self.mode_var,
                        command=self._mode).pack(side="left", padx=8)
        self.mode_lbl = ttk.Label(row, text="", font=("Helvetica", 11, "bold"))
        self.mode_lbl.pack(side="right")

        # live state / fault information
        self.state_info = tk.StringVar()
        ttk.Label(box, textvariable=self.state_info, font=("Courier", 10), justify="left").pack(anchor="w", pady=(4, 0))
        self.fault_info = tk.StringVar()
        ttk.Label(box, textvariable=self.fault_info, foreground="#b22", wraplength=380).pack(anchor="w")

        # joint jog sliders (target) + actual angle
        ttk.Label(box, text="Joint jog (deg)  slider = target, right = actual").pack(anchor="w", pady=(6, 0))
        qlim = np.degrees(np.array(self.fanuc.robot.qlim, dtype=float))
        self.joint_vars, self.joint_actual, self._last_set, self._drag = [], [], [None] * 6, set()
        for i in range(6):
            r = ttk.Frame(box)
            r.pack(fill="x")
            ttk.Label(r, text=f"J{i + 1}", width=3).pack(side="left")
            var = tk.DoubleVar(value=0.0)
            sc = tk.Scale(r, from_=qlim[0, i], to=qlim[1, i], resolution=0.5, orient="horizontal",
                          length=250, variable=var, command=lambda v, i=i: self._joint_slider(i, v))
            sc.pack(side="left")
            sc.bind("<ButtonPress-1>", lambda e, i=i: self._drag.add(i))
            sc.bind("<ButtonRelease-1>", lambda e, i=i: self._drag.discard(i))
            act = ttk.Label(r, text="", width=8, font=("Courier", 10))
            act.pack(side="left")
            self.joint_vars.append(var)
            self.joint_actual.append(act)
            self.jog_widgets.append(sc)

        # Cartesian jog
        cart = ttk.Frame(box)
        cart.pack(fill="x", pady=(6, 0))
        ttk.Label(cart, text="Cartesian jog, step (mm)").grid(row=0, column=0, columnspan=2, sticky="w")
        self.step_mm = tk.StringVar(value="20")
        cb = ttk.Combobox(cart, textvariable=self.step_mm, values=["5", "10", "20", "50"], width=5,
                          state="readonly")
        cb.grid(row=0, column=2, sticky="w")
        self.jog_widgets.append(cb)
        for c, axis in enumerate("XYZ"):
            for r_, sign in enumerate((+1, -1)):
                b = ttk.Button(cart, text=f"{'+' if sign > 0 else '-'}{axis}", width=5,
                               command=lambda a=c, s=sign: self._cart(a, s))
                b.grid(row=1 + r_, column=c, padx=2, pady=2)
                self.jog_widgets.append(b)

        # type a joint state and go there (planned, collision-checked)
        ttk.Label(box, text="Go to joint state (deg)").pack(anchor="w", pady=(6, 0))
        er = ttk.Frame(box)
        er.pack(fill="x")
        self.goto_vars = []
        for i in range(6):
            v = tk.StringVar(value="0")
            e = ttk.Entry(er, textvariable=v, width=6)
            e.pack(side="left", padx=1)
            self.goto_vars.append(v)
            self.jog_widgets.append(e)
        gr = ttk.Frame(box)
        gr.pack(fill="x", pady=(2, 0))
        for text, cmd in (("Go", self._goto), ("Use current", self._fill_current)):
            b = ttk.Button(gr, text=text, command=cmd)
            b.pack(side="left", expand=True, fill="x")
            self.jog_widgets.append(b)

        tools = ttk.Frame(box)
        tools.pack(fill="x", pady=(4, 0))
        for text, cmd in (("Open gripper", lambda: self.fanuc.jog_gripper(False)),
                          ("Close gripper", lambda: self.fanuc.jog_gripper(True)),
                          ("Wait pose", self.fanuc.jog_home)):
            b = ttk.Button(tools, text=text, command=cmd)
            b.pack(side="left", expand=True, fill="x")
            self.jog_widgets.append(b)

    def _mode(self):
        self.fanuc.request_mode(self.mode_var.get())
        if self.mode_var.get() == "manual" and self.fanuc.mode == "auto":
            self.msg.set("Manual requested: the FANUC finishes the item it is holding first.")

    def _joint_slider(self, i, value):
        value = float(value)
        if self.fanuc.mode != "manual":
            return
        if self._last_set[i] is not None and abs(value - self._last_set[i]) < 0.26:
            return                      # our own refresh moved the slider, not the user
        self._last_set[i] = value
        self.fanuc.jog_joint(i, np.radians(value))

    def _cart(self, axis, sign):
        d = [0.0, 0.0, 0.0]
        d[axis] = sign * float(self.step_mm.get()) / 1000
        self.fanuc.jog_cartesian(*d)

    def _goto(self):
        try:
            q = np.radians([float(v.get()) for v in self.goto_vars])
        except ValueError:
            self.msg.set("Joint state: type six numbers in degrees")
            return
        self.fanuc.goto_joints(q)
        self.msg.set("Go to joint state: " + ", ".join(v.get() for v in self.goto_vars) + " deg")

    def _fill_current(self):
        for v, a in zip(self.goto_vars, np.degrees(self.fanuc.robot.q)):
            v.set(f"{a:.1f}")

    def _drop_obstacle(self):
        self.fanuc.request_obstacle(self.spot.get())
        self.msg.set(f"Obstacle requested: {self.spot.get()}")

    # ---------- actions ----------
    def _estop(self):
        self.world.estop("E-stop button pressed (control panel)")
        self.msg.set("E-stop pressed. Clear the cause, then Reset, then Resume.")

    def _reset(self):
        self.msg.set(self.world.reset()[1])

    def _resume(self):
        self.msg.set(self.world.resume()[1])

    def _person(self, where):
        if self.safety is not None:
            self.safety.send_person(where)
            self.msg.set(f"Test person -> {where}")

    def _pick(self):
        if self.kind.get():
            self.world.request_pick(self.kind.get())
            self.msg.set(f"Queued: Kawasaki picks {self.kind.get()}")

    def _feed(self):
        manual = self.feed_var.get() == "manual"
        self.world.auto_feed = not manual
        if manual:
            self.world.infeed_status = ""
        else:
            self.world.pick_requests.clear()     # don't leave old choices queued for later
        for w in self.pick_widgets:
            w.state(["!disabled", "readonly"] if manual and isinstance(w, ttk.Combobox)
                    else ["!disabled"] if manual else ["disabled"])
        self.msg.set("Manual feed: the Kawasaki finishes its current item, then picks only what you choose."
                     if manual else "Auto feed: the Kawasaki picks items from the tub by itself.")

    # ---------- refresh ----------
    def _refresh(self):
        try:
            self._refresh_inner()
        finally:
            self.root.after(REFRESH_MS, self._refresh)

    def _refresh_inner(self):
        state = self.world.state()
        text, colour = STATE_TEXT[state]
        self.state_lbl.config(text=text, bg=colour)
        self.reason.set(self.world.stop_reason if state != "running" else "")
        if self.collision_status is not None:
            self.collision_status.set(self.fanuc.collision_msg)
        if self.fanuc is not None and self.jog_widgets:
            self._refresh_jog()
        if self.feed_var.get() == "manual":
            queued = len(self.world.pick_requests)
            status = self.world.infeed_status or "Kawasaki finishing its current item"
            self.feed_status.set(status + (f"   ({queued} queued)" if queued else ""))
        else:
            self.feed_status.set("")


    def _refresh_jog(self):
        st = self.fanuc.status()
        mode = st["mode"]
        label = "finishing item..." if mode == "auto" and st["requested"] == "manual" else MODE_TEXT.get(mode, mode)
        self.mode_lbl.config(text=label)
        x, y, z = st["tcp"]
        q = np.degrees(st["q"])
        self.state_info.set(f"TCP  x {x:+.3f}  y {y:+.3f}  z {z:+.3f} m\n"
                            f"q    " + " ".join(f"{a:+6.1f}" for a in q) + " deg\n"
                            f"cell {self.world.state().upper():8s} sorted: plastic {st['counts']['plastic']}"
                            f"  metal {st['counts']['metal']}")
        faults = [m for m in (st["jog_msg"], st["collision_msg"]) if m]
        if self.world.state() != "running":
            faults.insert(0, "E-STOP: " + (self.world.stop_reason or "reset - press Resume"))
        self.fault_info.set("\n".join(faults))
        manual = mode == "manual"
        for w in self.jog_widgets:
            if isinstance(w, ttk.Combobox):
                w.state(["!disabled", "readonly"] if manual else ["disabled"])
            elif isinstance(w, tk.Scale):
                w.config(state="normal" if manual else "disabled")
            else:
                w.state(["!disabled"] if manual else ["disabled"])
        q_show = st["target"] if manual and st["target"] is not None else st["q"]
        for i in range(6):
            deg = round(float(np.degrees(q_show[i])) * 2) / 2
            if abs(self.joint_vars[i].get() - deg) >= 0.26 and i not in self._drag:
                self._last_set[i] = deg
                self.joint_vars[i].set(deg)
            self.joint_actual[i].config(text=f"{q[i]:+6.1f}")


def run_panel(world, fanuc=None, safety=None):
    """Show the control panel on the main thread. Blocks until it is closed,
    which ends the program. Returns False if Tk is unavailable."""
    try:
        root = tk.Tk()
    except tk.TclError as err:
        print(f"[panel] no display for the control panel ({err}) - running without it")
        return False
    panel = ControlPanel(root, world, fanuc)
    panel.safety = safety
    root.mainloop()
    return True