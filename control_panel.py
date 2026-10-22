"""
Control panel (tkinter) for the recycle-bot workcell.

    E-STOP / Reset / Resume        (Esc also trips the e-stop)
    Conveyor belt speed
    Item feed: Auto (Kawasaki keeps feeding from the tub) or
               Manual (feed stops; you spawn chosen items yourself)

Runs on the MAIN thread (macOS only allows Tk there); the stations run in
their own threads. The panel never touches Swift itself: it only sets flags
and queues requests, which the World / station threads carry out. So it keeps
working during an e-stop, when the simulation clock is frozen.
"""
import os
import tkinter as tk
from tkinter import ttk

REFRESH_MS = 100
BELT_SPEED_RANGE = (0.05, 0.25)    # m/s; above ~0.25 the FANUC can't catch items in its reach
STATE_TEXT = {
    "running": ("RUNNING", "#1b8a2e"),
    "estop": ("E-STOP", "#c62828"),
    "reset": ("RESET - press Resume", "#e08a00"),
}


class ControlPanel:
    def __init__(self, root, world, fanuc=None, spawner=None):
        self.root, self.world, self.fanuc, self.spawner = root, world, fanuc, spawner
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
        ttk.Label(root, textvariable=self.msg, padding=(8, 0, 8, 8), foreground="#555",
                  wraplength=290).grid(row=1, column=0, sticky="w")
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
        self.spawn_box = ttk.Frame(box)
        self.spawn_box.pack(fill="x", pady=(6, 0))
        if self.spawner is None:
            ttk.Label(self.spawn_box, text="Spawner not loaded").pack()
            self.spawn_widgets = []
            return
        self.kind = tk.StringVar(value=self.spawner.kinds[0])
        self.place = tk.StringVar(value=self.spawner.places[-1])
        self.spawn_widgets = [
            ttk.Combobox(self.spawn_box, textvariable=self.kind, values=self.spawner.kinds,
                         state="readonly", width=30),
            ttk.Combobox(self.spawn_box, textvariable=self.place, values=self.spawner.places,
                         state="readonly", width=30),
            ttk.Button(self.spawn_box, text="Spawn", command=self._spawn),
        ]
        for w in self.spawn_widgets:
            w.pack(fill="x", pady=2)
        self._feed()
        self.msg.set("")

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

    def _spawn(self):
        self.spawner.request(self.kind.get(), self.place.get())
        self.msg.set(f"Queued {self.kind.get()} -> {self.place.get()}")

    def _feed(self):
        manual = self.feed_var.get() == "manual"
        self.world.auto_feed = not manual
        for w in self.spawn_widgets:
            w.state(["!disabled", "readonly"] if manual and isinstance(w, ttk.Combobox)
                    else ["!disabled"] if manual else ["disabled"])
        self.msg.set("Manual feed: the Kawasaki stops after its current item. Choose items to spawn."
                     if manual else "Auto feed: the Kawasaki feeds items from the tub.")

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


def run_panel(world, fanuc=None, safety=None, spawner=None):
    """Show the control panel on the main thread. Blocks until it is closed,
    which ends the program. Returns False if Tk is unavailable."""
    try:
        root = tk.Tk()
    except tk.TclError as err:
        print(f"[panel] no display for the control panel ({err}) - running without it")
        return False
    panel = ControlPanel(root, world, fanuc, spawner)
    panel.safety = safety
    root.mainloop()
    return True