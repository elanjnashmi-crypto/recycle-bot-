"""
Promo-video tour of the whole simulation.

Runs exactly what main_sim.py runs, then directs a scripted ~3 minute walkthrough: the
camera flies to each station, a lower-third caption explains what is happening, a dim
telemetry readout shows live cell data, and the control panel is driven on screen
(pointer glides to each control, buttons press, sliders move).

    python promo_tour.py --render [--out FILE.mp4] [--seconds N]

Best quality (Windows). Renders the video frame by frame instead of screen-recording it:
every sim step the cell pauses while each frame is drawn, grabbed and encoded, so the
result is perfectly even 1080p60 however slowly this machine runs (rendering takes several
times longer than the video). Leave the screen alone while it runs; it exits when done.
Default output: ~/Videos/recycle-bot-promo-<date>-<time>.mp4

    python promo_tour.py --auto

Live, hands-off: arranges the windows and starts the tour AUTO_START_DELAY s after READY,
so a screen recorder (OBS) can be started in between.

    python promo_tour.py

Live, manual: when the caption says READY, start recording, then click the caption.

Camera shots live in SHOTS (camera position, look-at point; world frame, metres).
Caption text lives in script(). Promo-only scenery fixes are in _promo_scenery().
"""
import math
import os
import sys
import time
import tkinter as tk
import traceback
import types
from tkinter import ttk

import numpy as np
import swift
from spatialgeometry import Cuboid

import control_panel
import main_sim
import safety
from factory import GLAZING, SHRINK_WRAP
from safety import PERSON_STOPS

# ---- modes ----
RENDER = "--render" in sys.argv
AUTO = "--auto" in sys.argv or RENDER
RENDER_FPS = 60
RENDER_SIZE = (1920, 1080)     # 16:9; the frame is cropped from the middle of the screen
RENDER_SETTLE = 0.045          # s for the browser to paint (and Tk to update) before each grab
RENDER_SECONDS = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else None

# ---- camera ----
SHOTS = {   # name: (camera position, look-at point)
    "wide":     ((7.6, -8.6, 5.2), (0.3, 0.6, 0.6)),       # same as factory.py's start view
    "kawasaki": ((1.6, 2.3, 2.0), (-0.4, 3.6, 0.8)),
    "fanuc":    ((1.5, -1.5, 2.1), (-0.2, 0.3, 0.8)),
    "organic":  ((-0.6, -3.4, 1.7), (-2.0, -2.8, 0.6)),
    "ur3e":     ((0.4, -3.4, 1.8), (-0.8, -2.4, 0.85)),
    "yaskawa":  ((2.9, 0.4, 2.3), (1.9, 1.6, 0.7)),         # inside the fence, above its top rail
    "gate":     ((2.4, -5.8, 2.0), (0.0, -3.9, 0.9)),
    "outro":    ((0.3, -7.5, 5.4), (0.3, 0.6, 0.5)),        # under the roof trusses
    # Panel-on-screen variants: look-at moved right so the subject sits left of the panel.
    "panel":    ((4.5, -5.5, 4.5), (1.8, 1.5, 0.5)),
    "kawasaki_l": ((1.6, 2.3, 2.0), (-0.08, 4.1, 0.8)),
    "teach_l":  ((1.1, -1.1, 1.6), (0.38, 0.53, 0.9)),
    "yaskawa_l": ((2.9, 0.4, 2.3), (2.25, 1.9, 0.7)),
    "gate_l":   ((2.4, -5.8, 2.0), (0.56, -3.2, 0.9)),
}
FLY_TIME = 2.5                 # s per camera move
ARC = 0.15                     # camera lifts by this fraction of the distance mid-flight (max 1.5 m)
DRIFT_SPEED = 0.05             # m/s slow orbit round the subject while a shot holds...
DRIFT_MAX = 0.5                # ...for at most this far, so it never wanders into walls or fences
CAMERA_HZ = 30                 # live modes only; --render sets the camera every frame

# ---- tour ----
WARMUP_SPEED = 5.0             # x real time before the tour starts
WARMUP_MAX_S = 150             # wall-clock cap on the warm-up
AUTO_START_DELAY = 12          # s between READY and the tour starting, in --auto
MOVE_POINTER = True            # live modes: glide the real mouse pointer to each control
GLIDE_S = 0.45
PRESS_S = 0.18

# ---- overlays (lower third + telemetry readout) ----
LOWER_THIRD_W = 900            # px (logical), max width
OVERLAY_ALPHA = 0.90
HUD_ALPHA = 0.78
BROWSER_CHROME = 88            # px (logical) of browser tabs + address bar above the 3D view (--auto)
FADE_OUT_S, FADE_IN_S = 0.18, 0.32
PANEL_FADE_S = 0.30
BG = "#0e1116"
ACCENT = "#4fa3c7"             # thin rule on the lower third's left edge
RULE = "#2b323c"
KICKER_FG, TITLE_FG, BODY_FG = "#8b95a3", "#f1f3f6", "#aab3bf"
SPEC_KEY_FG, SPEC_VAL_FG = "#5f6b7a", "#9aa4b1"
KICKER_FONT = ("Segoe UI", 10, "bold")
TITLE_FONT = ("Segoe UI Semibold", 23)
BODY_FONT = ("Segoe UI", 13)
MONO_FONT = ("Consolas", 10)
STATE_FG = {"running": "#5fb37a", "slow": "#d8a24a", "estop": "#d0605e", "reset": "#d8a24a"}

# Published manufacturer figures, shown as dim notes beside each station's caption.
SPECS = {
    "kawasaki": [("PAYLOAD", "7 kg"), ("REACH", "930 mm"), ("AXES", "6")],
    "fanuc": [("PAYLOAD", "7 kg"), ("REACH", "717 mm"), ("GUIDANCE", "Visual servo"), ("MOTION", "RMRC")],
    "ur3e": [("PAYLOAD", "3 kg"), ("REACH", "500 mm"), ("AXES", "6")],
    "yaskawa": [("PAYLOAD", "7 kg"), ("REACH", "927 mm"), ("GUIDANCE", "Visual servo"), ("SORT", "Load cell, 30 g")],
}

# ---- promo-only scenery fixes (applied as shapes are added; the sim's files are untouched) ----
FLOOR_DECAL_LIFT = 4.0         # floor markings sit 2-5 mm up: spread them out so they can't z-fight
OPAQUE_SCENERY = {             # see-through scenery that flickers when the renderer re-sorts it
    GLAZING: (0.66, 0.76, 0.84, 1),
    SHRINK_WRAP: (0.84, 0.86, 0.89, 1),
}


def _promo_scenery(shape):
    if not isinstance(shape, Cuboid):
        return
    if shape.scale[2] <= 0.008 and abs(shape.T[2, 3]) < 0.02:
        T = np.array(shape.T)
        T[2, 3] *= FLOOR_DECAL_LIFT
        shape.T = T
    colour = tuple(round(float(c), 3) for c in shape.color)
    for see_through, opaque in OPAQUE_SCENERY.items():
        if colour == tuple(round(c, 3) for c in see_through):
            shape.color = opaque


_swift_add = swift.Swift.add


def _swift_add_with_fixes(self, ob, *args, **kwargs):
    _promo_scenery(ob)
    return _swift_add(self, ob, *args, **kwargs)


def _all(w):
    yield w
    for c in w.winfo_children():
        yield from _all(c)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Clock:
    """Wall time, except while --render is writing frames: then video time (frames / fps),
    so every animation in the tour advances exactly one frame per frame."""

    def __init__(self):
        self.base, self.frames = None, 0

    def now(self):
        return time.time() if self.base is None else self.base + self.frames / RENDER_FPS

    def start_video(self):
        self.base, self.frames = time.time(), 0


CLOCK = Clock()


class Camera:
    """Flies between shots on a gentle arc, then drifts slowly round the subject."""

    def __init__(self, root, env, shot):
        self.root, self.env = root, env
        self.pos, self.look = (np.array(v, float) for v in SHOTS[shot])
        self.arrive_t = CLOCK.now()
        self._from = None                  # (pos, look, start time, duration) while flying
        if not RENDER:
            self._tick()

    def fly(self, shot, dur=FLY_TIME):
        p0, l0 = self._current()
        self._from = (p0, l0, CLOCK.now(), dur)
        self.pos, self.look = (np.array(v, float) for v in SHOTS[shot])
        self.arrive_t = CLOCK.now() + dur
        return dur

    def _current(self):
        now = CLOCK.now()
        d = self.pos - self.look
        r = max(float(np.hypot(d[0], d[1])), 0.5)
        a = min(DRIFT_SPEED * max(0.0, now - self.arrive_t), DRIFT_MAX) / r
        pos = self.look + np.array([d[0] * math.cos(a) - d[1] * math.sin(a),
                                    d[0] * math.sin(a) + d[1] * math.cos(a), d[2]])
        look = self.look
        if self._from is not None:
            p0, l0, t0, dur = self._from
            u = (now - t0) / dur
            if u >= 1:
                self._from = None
            else:
                k = _smooth(u)
                hop = min(ARC * np.linalg.norm(pos - p0), 1.5) * math.sin(math.pi * k)
                pos = p0 + k * (pos - p0) + np.array([0, 0, hop])
                look = l0 + k * (look - l0)
        return pos, look

    def apply(self):
        pos, look = self._current()
        self.env.set_camera_pose(pos.tolist(), look.tolist())     # only queues a message: thread-safe

    def _tick(self):
        self.apply()
        self.root.after(1000 // CAMERA_HZ, self._tick)


def _tracked(text):
    """Letter-spaced capitals for small labels (Tk has no letter-spacing option)."""
    return " ".join(text.upper()).replace("   ", " ")


def _overlay_window(root, alpha):
    w = tk.Toplevel(root)
    w.overrideredirect(True)
    w.attributes("-topmost", True)
    w.attributes("-alpha", alpha)
    w.configure(bg=BG)
    return w


def _animate(root, dur, step, then=None, alive=lambda: True):
    """Call step(u) with u going 0 -> 1 over `dur` seconds of CLOCK time, then then()."""
    start = CLOCK.now()

    def tick():
        if not alive():
            return
        u = 1.0 if dur <= 0 else min((CLOCK.now() - start) / dur, 1.0)
        step(u)
        if u < 1:
            root.after(10, tick)
        elif then:
            then()
    tick()


class LowerThird:
    """Broadcast-style caption: small kicker, title, one muted line of body text, and an
    optional dim column of notes on the right. Cross-fades on every change.
    Drag to move; a click (no drag) calls on_click."""

    def __init__(self, root):
        self.root, self.on_click = root, None
        w = self.win = _overlay_window(root, 0.01)
        tk.Frame(w, bg=ACCENT, width=3).pack(side="left", fill="y")
        main = tk.Frame(w, bg=BG, padx=20, pady=14)
        main.pack(side="left", fill="both", expand=True)
        self.kicker = tk.Label(main, font=KICKER_FONT, fg=KICKER_FG, bg=BG, anchor="w")
        self.kicker.pack(fill="x")
        self.title = tk.Label(main, font=TITLE_FONT, fg=TITLE_FG, bg=BG, anchor="w")
        self.title.pack(fill="x", pady=(1, 2))
        self.body = tk.Label(main, font=BODY_FONT, fg=BODY_FG, bg=BG, anchor="w", justify="left")
        self.body.pack(fill="x")
        self.rule = tk.Frame(w, bg=RULE, width=1)
        self.specs = tk.Frame(w, bg=BG, padx=20, pady=16)
        for widget in (w, main, self.kicker, self.title, self.body):
            widget.bind("<ButtonPress-1>", self._press)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<ButtonRelease-1>", self._release)
        self.width = LOWER_THIRD_W
        self.anchor = (40, w.winfo_screenheight() - 90)      # left edge, bottom edge
        self.alpha, self._token = 0.0, 0
        w.withdraw()

    def place(self, x, bottom, width):
        self.anchor, self.width = (x, bottom), min(width, LOWER_THIRD_W)
        if self.alpha > 0:
            self._layout()

    def show(self, kicker, title, body="", specs=None):
        """Fade out whatever is showing, swap the text, fade back in."""
        self._token += 1
        token = self._token

        def swap():
            self.kicker.config(text=_tracked(kicker))
            self.title.config(text=title)
            self.body.config(text=body)
            if body:
                self.body.pack(fill="x")
            else:
                self.body.pack_forget()
            self._set_specs(specs or [])
            self._layout()
            self.win.deiconify()
            self._fade(OVERLAY_ALPHA, FADE_IN_S, token)

        self._fade(0.0, FADE_OUT_S if self.alpha > 0 else 0, token, then=swap)

    def hide(self):
        self._token += 1
        self._fade(0.0, FADE_OUT_S, self._token, then=self.win.withdraw)

    def _set_specs(self, specs):
        for c in self.specs.winfo_children():
            c.destroy()
        if not specs:
            self.rule.pack_forget()
            self.specs.pack_forget()
            return
        for r, (key, val) in enumerate(specs):
            tk.Label(self.specs, text=key, font=MONO_FONT, fg=SPEC_KEY_FG, bg=BG,
                     anchor="w").grid(row=r, column=0, sticky="w", padx=(0, 14))
            tk.Label(self.specs, text=val, font=MONO_FONT, fg=SPEC_VAL_FG, bg=BG,
                     anchor="w").grid(row=r, column=1, sticky="w")
        self.rule.pack(side="left", fill="y", pady=16)
        self.specs.pack(side="left", fill="y")

    def _layout(self):
        self.win.update_idletasks()
        spec_w = self.specs.winfo_reqwidth() + 1 if self.specs.winfo_children() else 0
        self.body.config(wraplength=max(200, self.width - spec_w - 50))
        self.win.update_idletasks()
        h = self.win.winfo_reqheight()
        x, bottom = self.anchor
        self.win.geometry(f"{self.width}x{h}+{x}+{bottom - h}")

    def _fade(self, target, dur, token, then=None):
        a0 = self.alpha

        def step(u):
            self.alpha = a0 + _smooth(u) * (target - a0)
            self.win.attributes("-alpha", max(self.alpha, 0.01))
        _animate(self.root, dur, step, then, alive=lambda: token == self._token)

    def _press(self, e):
        self._grab = (e.x_root, e.y_root, self.win.winfo_x(), self.win.winfo_y())
        self._moved = False

    def _drag(self, e):
        x0, y0, wx, wy = self._grab
        if abs(e.x_root - x0) + abs(e.y_root - y0) > 4:
            self._moved = True
        x, y = wx + e.x_root - x0, wy + e.y_root - y0
        self.win.geometry(f"+{x}+{y}")
        self.anchor = (x, y + self.win.winfo_height())

    def _release(self, e):
        if not self._moved and self.on_click:
            self.on_click()


class Telemetry:
    """Small dim live readout of the cell: sim time, state, belt / sim speed, sort counts."""

    def __init__(self, root, world, fanuc, yaskawa, safety):
        self.root, self.world, self.fanuc, self.yaskawa, self.safety = root, world, fanuc, yaskawa, safety
        w = self.win = _overlay_window(root, HUD_ALPHA)
        box = tk.Frame(w, bg=BG, padx=14, pady=10)
        box.pack()
        tk.Label(box, text=_tracked("Cell telemetry"), font=KICKER_FONT, fg=KICKER_FG, bg=BG,
                 anchor="w").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.left = [tk.Label(box, font=MONO_FONT, fg=SPEC_VAL_FG, bg=BG, anchor="w", justify="left")
                     for _ in range(3)]
        for r, lbl in enumerate(self.left):
            lbl.grid(row=r + 1, column=0, sticky="w")
        self.state = tk.Label(box, font=MONO_FONT, bg=BG, anchor="e")
        self.state.grid(row=1, column=1, sticky="e", padx=(18, 0))
        self.anchor = (40, 40)
        self._on = False
        w.withdraw()

    def place(self, x, y):
        self.anchor = (x, y)
        self.win.geometry(f"+{x}+{y}")

    def show(self):
        self.win.geometry(f"+{self.anchor[0]}+{self.anchor[1]}")
        self.win.deiconify()
        if not self._on:
            self._on = True
            self._tick()

    def hide(self):
        self._on = False
        self.win.withdraw()

    def _tick(self):
        if not self._on:
            return
        world = self.world
        state = world.state()
        slow = state == "running" and world.slow_factor < 1
        word = {"running": "RUNNING", "estop": "E-STOP", "reset": "RESET"}[state]
        if slow:
            word = f"REDUCED {world.slow_factor:.0%}"
        self.state.config(text="● " + word, fg=STATE_FG["slow" if slow else state])
        self.left[0].config(text=f"T+ {world.t:7.1f} s")
        self.left[1].config(text=f"BELT {world.belt_speed():.2f} m/s   RATE {world.sim_speed:.1f}x")
        counts = []
        if self.fanuc:
            counts += [f"PL {self.fanuc.counts['plastic']}", f"ME {self.fanuc.counts['metal']}"]
        if self.yaskawa:
            counts += [f"AL {self.yaskawa.counts['Al']}", f"ST {self.yaskawa.counts['Non-Al']}"]
        self.left[2].config(text="SORTED " + "  ".join(counts) if counts else "")
        self.root.after(100 if RENDER else 250, self._tick)


class Renderer:
    """--render: frame-locked capture. The World calls _pace() after every sim step with its
    lock held, so the whole cell waits while each frame is drawn, grabbed and encoded."""

    def __init__(self, director, path):
        self.d, self.path = director, path
        self.world, self.env = director.world, director.world.env
        self.writer, self.due = None, 0.0
        self._pace0, self._pump0 = self.world._pace, self.world._pump_while_stopped
        self.world._pace = self._pace                    # instance attributes: world.py is untouched
        self.world._pump_while_stopped = self._pump

    def start(self):
        import cv2
        from PIL import ImageGrab
        self.cv2, self.grab = cv2, ImageGrab.grab
        phys_w, phys_h = self.grab().size
        self.scale = phys_w / self.d.root.winfo_screenwidth()         # Tk works in logical pixels
        crop_h = round(phys_w * RENDER_SIZE[1] / RENDER_SIZE[0])
        self.y0 = max(0, (phys_h - crop_h) // 2)
        self.crop_h = min(crop_h, phys_h)
        self.out_scale = RENDER_SIZE[0] / phys_w
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(self.path, cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"),
                                 RENDER_FPS, RENDER_SIZE)
        if not writer.isOpened():
            raise RuntimeError(f"can't open an H.264 writer for {self.path}")
        self.env.rate = 10_000                           # never let Swift skip a step's draw
        with self.world._cond:
            CLOCK.start_video()
            self.writer = writer
        print(f"[render] writing {self.path}")
        if RENDER_SECONDS:                               # test renders: stop early
            self._watch_length()

    def _watch_length(self):
        if CLOCK.frames >= RENDER_SECONDS * RENDER_FPS:
            self.d._finish()
        else:
            self.d.root.after(200, self._watch_length)

    def stop(self):
        with self.world._cond:                           # waits for a frame in progress
            writer, self.writer = self.writer, None
        if writer is not None:
            writer.release()
        print(f"[render] done: {self.path} ({CLOCK.frames} frames, {CLOCK.frames / RENDER_FPS:.1f} s)")

    def _pace(self):
        if self.writer is None:
            return self._pace0()
        w = self.world
        self.due += RENDER_FPS * w.dt / (w.sim_speed * max(w.slow_factor, 0.05))
        while self.due >= 1 and self.writer is not None:
            self.due -= 1
            self._frame()

    def _pump(self):
        """While e-stopped no steps happen, but the video (and the person walking) goes on."""
        if self.writer is None:
            return self._pump0()
        self.world._run_hooks()
        self.env.step(0)
        self._frame()

    def _frame(self):
        CLOCK.frames += 1
        self.d.cam.apply()
        self.env._send_socket("shape_poses", [], True)   # round trip: the browser has applied it all
        time.sleep(RENDER_SETTLE)                        # next browser paint; Tk updates the overlays
        img = np.asarray(self.grab())[self.y0:self.y0 + self.crop_h, :, ::-1]
        frame = self.cv2.resize(img, RENDER_SIZE, interpolation=self.cv2.INTER_AREA)
        if self.d.pointer_visible():
            self._draw_pointer(frame)
        self.writer.write(np.ascontiguousarray(frame))

    def _draw_pointer(self, frame):
        """Screen grabs don't include the mouse pointer, so draw the tour's virtual one."""
        cv2 = self.cv2
        x, y = self.d.pointer
        k = self.scale * self.out_scale
        px, py = x * self.scale * self.out_scale, (y * self.scale - self.y0) * self.out_scale
        arrow = np.array([(0, 0), (0, 17), (4.5, 13), (7.5, 19.5), (10, 18.5), (7, 12.2), (12.5, 12.2)])
        pts = np.round(np.array([px, py]) + arrow * k * 1.05).astype(np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, (20, 20, 20), max(1, round(k)), lineType=cv2.LINE_AA)


class Director:
    """Runs script() on the Tk thread. The script yields a number (wait that many seconds)
    or a callable (wait until it returns True). All timing runs on CLOCK."""

    def __init__(self, root, panel, world, fanuc, yaskawa, safety, out=None):
        self.root, self.panel, self.world = root, panel, world
        self.fanuc, self.yaskawa, self.safety = fanuc, yaskawa, safety
        self.cam = Camera(root, world.env, "wide")
        self.cap = LowerThird(root)
        self.cap.on_click = self._start_clicked
        self.hud = Telemetry(root, world, fanuc, yaskawa, safety)
        self.renderer = Renderer(self, out) if RENDER else None
        self.pointer = (root.winfo_screenwidth() * 0.8, root.winfo_screenheight() * 0.7)
        self.panel_alpha = 1.0
        self._started = False
        self._gen = self.script()
        root.after(500, self._advance)

    # ---------- runner ----------
    def after(self, secs, fn):
        """Run fn after `secs` seconds of CLOCK time."""
        end = CLOCK.now() + secs

        def check():
            if CLOCK.now() >= end:
                fn()
            else:
                self.root.after(10, check)
        check()

    def _advance(self):
        try:
            wait = next(self._gen)
        except StopIteration:
            self._finish()
            return
        except Exception:
            print("[tour] stopped by an error:")
            traceback.print_exc()
            self.cap.show("Tour", "Stopped by an error", "See the terminal.")
            if self.renderer:
                self.renderer.stop()
            return
        if callable(wait):
            self._poll(wait)
        else:
            self.after(wait, self._advance)

    def _poll(self, cond):
        if cond():
            self._advance()
        else:
            self.root.after(20, lambda: self._poll(cond))

    def _finish(self):
        if self.renderer:
            self.renderer.stop()
        print("[tour] finished")
        if self.renderer:
            os._exit(0)

    @staticmethod
    def until(cond, timeout):
        """Wait condition: cond() is true, or `timeout` seconds have passed."""
        end = CLOCK.now() + timeout
        return lambda: cond() or CLOCK.now() > end

    def _start_clicked(self):
        self._started = True

    def pointer_visible(self):
        return self.panel_alpha > 0.5

    def panel_in(self):
        """--auto / --render: fade the control panel in on the right (hidden for station shots)."""
        return self._panel_fade(1.0)

    def panel_out(self):
        return self._panel_fade(0.0)

    def _panel_fade(self, target):
        if not AUTO:
            return 0
        a0 = self.panel_alpha

        def step(u):
            self.panel_alpha = a0 + _smooth(u) * (target - a0)
            self.root.attributes("-alpha", max(self.panel_alpha, 0.01))
        _animate(self.root, PANEL_FADE_S, step)
        return PANEL_FADE_S + 0.1

    def say(self, kicker, title, body="", specs=None):
        self.cap.show(kicker, title, body, specs)
        print(f"[tour] {kicker} | {title}: {body}")

    # ---------- control-panel helpers ----------
    def find(self, box_title, text=None, cls=None, index=0):
        box = next((w for w in _all(self.root)
                    if isinstance(w, ttk.LabelFrame) and w.cget("text") == box_title), None)
        if box is None:
            raise LookupError(f"no '{box_title}' section on the control panel")
        hits = []
        for w in _all(box):
            if cls is not None and not isinstance(w, cls):
                continue
            if text is not None:
                try:
                    if w.cget("text") != text:
                        continue
                except tk.TclError:
                    continue
            hits.append(w)
        return hits[index]

    def _glide(self, widget, then):
        """Pointer to the middle of widget: the tour's virtual pointer in --render (drawn into
        the frames), the real one in the live modes."""
        root = self.root
        if not RENDER and not MOVE_POINTER:
            then()
            return
        x0, y0 = self.pointer if RENDER else root.winfo_pointerxy()
        x1 = widget.winfo_rootx() + widget.winfo_width() / 2
        y1 = widget.winfo_rooty() + widget.winfo_height() / 2

        def step(u):
            k = _smooth(u)
            self.pointer = (x0 + k * (x1 - x0), y0 + k * (y1 - y0))
            if not RENDER:
                root.event_generate("<Motion>", warp=True, x=int(self.pointer[0]) - root.winfo_rootx(),
                                    y=int(self.pointer[1]) - root.winfo_rooty())
        _animate(root, GLIDE_S, step, then)

    def click(self, widget, action=None):
        """Pointer to the widget, show it pressed, then run its action (invoke() by default).
        Returns how long that takes, for the script to yield."""
        label = isinstance(widget, tk.Label)

        def release():
            if label:
                widget.config(relief="raised")
            else:
                widget.state(["!pressed"])
            (action or widget.invoke)()

        def press():
            if label:
                widget.config(relief="sunken")
            else:
                widget.state(["pressed"])
            self.after(PRESS_S, release)

        self._glide(widget, press)
        return GLIDE_S + PRESS_S + 0.35

    def choose(self, combo, value):
        """Pointer to a combobox, then select `value` in it."""
        self._glide(combo, lambda: combo.set(value))
        return GLIDE_S + 0.4

    def slide(self, scale, to, dur, on_value=None):
        """Pointer to a slider, then move it smoothly to `to`, calling on_value as it goes."""
        v0 = float(scale.get())

        def step(u):
            scale.set(v0 + _smooth(u) * (to - v0))
            if on_value:
                on_value(float(scale.get()))

        self._glide(scale, lambda: _animate(self.root, dur, step))
        return GLIDE_S + dur + 0.3

    # ---------- cell state ----------
    def _sorted_total(self):
        n = 0
        if self.fanuc:
            n += self.fanuc.counts["plastic"] + self.fanuc.counts["metal"]
        if self.yaskawa:
            n += sum(self.yaskawa.counts.values())
        return n

    def _warm(self):
        fanuc_ok = self.fanuc is None or self.fanuc.counts["plastic"] + self.fanuc.counts["metal"] >= 3
        yask_ok = self.yaskawa is None or sum(self.yaskawa.counts.values()) >= 1
        return fanuc_ok and yask_ok

    def _person_at(self, where):
        return lambda: abs(self.safety.person.y - PERSON_STOPS[where]) < 0.02

    # ---------- the tour ----------
    def script(self):
        world, panel = self.world, self.panel
        sim_scale = self.find("Simulation speed (x real time)", cls=tk.Scale)
        belt_scale = self.find("Conveyor belt speed (m/s)", cls=tk.Scale)

        # -- warm-up (not recorded) --
        self.say("Standby", "Warming up", f"Running at {WARMUP_SPEED:g}x so every station has work. Not recording yet.")
        sim_scale.set(WARMUP_SPEED)
        world.set_sim_speed(WARMUP_SPEED)
        yield self.until(self._warm, WARMUP_MAX_S)
        sim_scale.set(1.0)
        world.set_sim_speed(1.0)
        self.cam.fly("wide", 1.0)
        print("[tour] READY")
        if AUTO:
            arrange_windows(self.root, self.cap, self.hud, self)
            self.cap.hide()                               # clean frames while the recorder starts
            if RENDER:
                yield 1.5                                 # let the windows settle
                self.renderer.start()
                yield 0.5
            else:
                yield AUTO_START_DELAY
            self._started = True
        else:
            self.say("Standby", "Ready", "Arrange the windows, start recording, then click here.")
        yield lambda: self._started and world.state() == "running"

        # -- intro --
        self.hud.show()
        self.say("Automated recycling cell", "Recycle-bot",
                 "Four robots and seven conveyors sorting mixed waste by material, under one controller.",
                 [("ROBOTS", "4"), ("CONVEYORS", "7"), ("STREAMS", "4")])
        yield 7

        # -- stations, following an item down the line --
        self.say("01  Infeed", "Kawasaki RS007L",
                 "Transfers mixed waste from the intake tub to the main conveyor.", SPECS["kawasaki"])
        yield self.cam.fly("kawasaki") + 10

        if self.fanuc:
            self.say("02  Classification", "FANUC LR Mate 200iD",
                     "An overhead camera closes the loop on each moving item.", SPECS["fanuc"])
            yield self.cam.fly("fanuc") + 7
            self.say("02  Classification", "FANUC LR Mate 200iD",
                     "Plastics and metals are diverted to dedicated lines. All other waste continues.",
                     SPECS["fanuc"])
            before = self._sorted_total()
            yield self.until(lambda: self._sorted_total() > before, 12)
            yield 3

        self.say("03  Organics", "Organic stream",
                 "Organic waste runs to the end of the general line and is collected in a crate.")
        yield self.cam.fly("organic") + 7

        self.say("04  Plastics", "Universal Robots UR3e",
                 "Bottles arriving on the plastic line are packed into a collection crate.", SPECS["ur3e"])
        yield self.cam.fly("ur3e") + 10

        if self.yaskawa:
            self.say("05  Metals", "Yaskawa Motoman GP7",
                     "Cans are tracked by a dedicated camera and picked from the return line.", SPECS["yaskawa"])
            yield self.cam.fly("yaskawa") + 6
            self.say("05  Metals", "Yaskawa Motoman GP7",
                     "A load cell weighs each can, separating aluminium from steel.", SPECS["yaskawa"])
            before = sum(self.yaskawa.counts.values())
            yield self.until(lambda: sum(self.yaskawa.counts.values()) > before, 18)
            yield 3

        # -- control panel (on screen from here until the safety walk-through) --
        self.say("Operator control", "Conveyor speed", "Set for the whole cell from a single control.",
                 [("RANGE", "0.05 - 0.25 m/s")])
        self.panel_in()
        yield self.cam.fly("panel")
        belt0 = float(belt_scale.get())
        yield self.slide(belt_scale, 0.22, 1.5, world.set_belt_speed) + 2.5
        yield self.slide(belt_scale, belt0, 1.2, world.set_belt_speed)

        self.say("Operator control", "Simulation rate", "Runs the entire cell faster than real time for testing.",
                 [("RANGE", "0.5 - 5x")])
        yield self.slide(sim_scale, 3.0, 1.5, world.set_sim_speed) + 4
        yield self.slide(sim_scale, 1.0, 1.0, world.set_sim_speed) + 0.5

        if control_panel.ITEM_KINDS:
            self.say("Operator control", "Manual feed", "The operator selects the next item for the infeed robot.")
            yield self.cam.fly("kawasaki_l", 2.0)
            yield self.click(self.find("Item feed", "Manual"))
            kind = "aluminium_can" if "aluminium_can" in control_panel.ITEM_KINDS else control_panel.ITEM_KINDS[0]
            yield self.choose(self.find("Item feed", cls=ttk.Combobox), kind)
            yield self.click(self.find("Item feed", "Pick this item")) + 8
            yield self.click(self.find("Item feed", "Auto")) + 1

        if self.yaskawa:
            self.say("Collision avoidance", "Path re-planning",
                     "An obstacle is placed in the GP7's path. The collision is predicted and the arm routes over it.",
                     [("MODEL", "Capsule links"), ("CHECK", "Every move")])
            yield self.cam.fly("yaskawa_l")
            spot = next((s for s in self.yaskawa.obstacle_spots if "swing" in s), next(iter(self.yaskawa.obstacle_spots)))
            yield self.choose(self.find("Collision test (Yaskawa)", cls=ttk.Combobox), spot)
            yield self.click(self.find("Collision test (Yaskawa)", "Drop obstacle"))
            # hold until the GP7 actually has to plan round it (the next can arriving), then watch it go over
            yield self.until(lambda: "Collision predicted" in self.yaskawa.collision_msg
                             or "Obstacle appeared" in self.yaskawa.collision_msg, 30)
            yield 6
            yield self.click(self.find("Collision test (Yaskawa)", "Clear obstacles")) + 1

        if self.fanuc:
            self.say("Teach mode", "Manual jog",
                     "The robot completes its current item before control is handed to the operator.")
            yield self.cam.fly("teach_l")
            yield self.click(self.find("FANUC teach / jog", "Manual (teach)"))
            yield self.until(lambda: self.fanuc.mode == "manual", 15)
            yield 0.4                                     # panel refresh enables the jog controls
            self.say("Teach mode", "Manual jog", "Joint and Cartesian jogs, each one collision-checked.",
                     [("MAX SPEED", "250 mm/s"), ("LIMITS", "Joint + zone")])
            j1 = self.find("FANUC teach / jog", cls=tk.Scale, index=0)
            lo, hi = float(j1.cget("from")), float(j1.cget("to"))
            target = min(max(float(j1.get()) + 35, lo), hi)
            panel._drag.add(0)                            # stop the panel's refresh fighting the slide
            yield self.slide(j1, target, 2.0, lambda v: panel._joint_slider(0, v)) + 1.5
            panel._drag.discard(0)
            yield self.choose(self.find("FANUC teach / jog", cls=ttk.Combobox), "50")
            for _ in range(2):
                yield self.click(self.find("FANUC teach / jog", "+Z")) + 0.6
            yield self.click(self.find("FANUC teach / jog", "Wait pose")) + 2.5
            yield self.click(self.find("FANUC teach / jog", "Auto"))
            yield self.until(lambda: self.fanuc.mode == "auto", 8)

        # -- safety: full-screen for the walk-in, panel back for the reset / resume --
        if self.safety:
            self.panel_out()
            self.say("Safeguarding", "Cell perimeter",
                     "Fencing, a light curtain at the gate, an area scanner, and a stack light and e-stop at each robot.")
            yield self.cam.fly("gate") + 4
            self.say("Safeguarding", "Area scanner", "Entering the zone reduces the speed of the entire cell.",
                     [("SPEED", "25 %")])
            panel._person("warning zone")
            yield self.until(lambda: self.safety.in_warning_zone, 4)
            yield 5
            self.say("Safeguarding", "Light curtain",
                     "Breaking the curtain triggers an emergency stop. All motion halts.",
                     [("RESPONSE", "Latched e-stop")])
            panel._person("inside cell")
            yield self.until(lambda: world.state() == "estop", 4)
            yield 5
            self.say("Safeguarding", "Interlock", "Reset is refused while anyone remains inside the fence.")
            self.panel_in()
            yield self.cam.fly("gate_l", 1.5)
            yield self.click(self.find("Emergency stop", "Reset")) + 3
            self.say("Safeguarding", "Restart", "With the cell clear, the operator resets, then resumes.")
            panel._person("outside")
            yield self._person_at("outside")
            yield 0.5
            yield self.click(self.find("Emergency stop", "Reset")) + 0.8
            yield self.click(self.find("Emergency stop", "Resume")) + 3.5

        self.say("Safeguarding", "Emergency stop",
                 "The panel e-stop halts the cell immediately. A physical e-stop button does the same.")
        self.panel_in()
        yield self.cam.fly("panel", 2.0)
        yield self.click(self.find("Emergency stop", "E-STOP", cls=tk.Label), panel._estop) + 3.5
        yield self.click(self.find("Emergency stop", "Reset")) + 0.8
        yield self.click(self.find("Emergency stop", "Resume")) + 2

        # -- outro --
        self.panel_out()
        self.cam.fly("outro", 3.5)
        counts = []
        if self.fanuc:
            counts += [("PLASTIC", str(self.fanuc.counts["plastic"])), ("METAL", str(self.fanuc.counts["metal"]))]
        if self.yaskawa:
            counts += [("ALUMINIUM", str(self.yaskawa.counts["Al"])), ("STEEL", str(self.yaskawa.counts["Non-Al"]))]
        self.say("Session summary", "Recycle-bot", "Sorted this session.", counts)
        yield 6
        self.say("Up next", "The physical cell")
        yield 4
        self.cap.hide()
        self.hud.hide()
        yield 1


def arrange_windows(root, cap, hud, director):
    """Windows only: Swift browser maximised; control panel parked on the right edge
    (always on top, faded out until a scene needs it); overlays over the 3D view. In
    --render everything is kept inside the 16:9 band that ends up in the video."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    work = wintypes.RECT()
    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0)          # SPI_GETWORKAREA

    found = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def check(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value.startswith("Swift"):
                found.append(hwnd)
        return True

    user32.EnumWindows(enum_proc(check), 0)
    if found:
        user32.ShowWindow(found[0], 3)                                       # SW_MAXIMIZE
        user32.SetForegroundWindow(found[0])
    else:
        print("[tour] couldn't find the Swift browser window to arrange")

    top, bottom = work.top + BROWSER_CHROME, work.bottom
    if RENDER:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        band = sw * RENDER_SIZE[1] / RENDER_SIZE[0]
        top, bottom = int((sh - band) / 2), int((sh + band) / 2)

    root.deiconify()                         # the process may have been started minimised
    root.state("normal")
    root.attributes("-alpha", 0.01)
    director.panel_alpha = 0.0
    root.attributes("-topmost", True)
    root.update_idletasks()
    pw, ph = root.winfo_width() + 16, root.winfo_height() + 39               # + border and title bar
    split = work.right - pw
    root.geometry(f"+{split}+{top + max(0, (bottom - top - ph) // 2)}")
    root.lift()
    cap.place(work.left + 28, bottom - 28, split - work.left - 56)
    hud.place(work.left + 28, top + 20)
    cap.win.lift()
    hud.win.lift()


def run_panel_with_tour(world, fanuc=None, safety=None, yaskawa=None):
    """control_panel.run_panel, plus the Director on the same Tk root."""
    try:
        root = tk.Tk()
    except tk.TclError as err:
        print(f"[tour] no display ({err}) - the tour needs the control panel")
        return False
    panel = control_panel.ControlPanel(root, world, fanuc, yaskawa)
    panel.safety = safety
    Director(root, panel, world, fanuc, yaskawa, safety, out=_render_path())
    root.mainloop()
    return True


def _render_path():
    if "--out" in sys.argv:
        return os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    stamp = time.strftime("%Y%m%d-%H%M")
    return os.path.join(os.path.expanduser("~"), "Videos", f"recycle-bot-promo-{stamp}.mp4")


if __name__ == "__main__":
    swift.Swift.add = _swift_add_with_fixes            # promo-only scenery fixes
    if RENDER:                                         # walking person + panel readouts on video time
        safety.time = types.SimpleNamespace(time=CLOCK.now)
        control_panel.time = types.SimpleNamespace(time=CLOCK.now)
    control_panel.run_panel = run_panel_with_tour      # main_sim imports run_panel when it needs it
    main_sim.main()
