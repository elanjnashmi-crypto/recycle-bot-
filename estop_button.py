"""
Physical e-stop button (Arduino on a USB serial port) -> the workcell.

The Arduino sketch prints "OK" or "ESTOP" about five times a second, and
immediately whenever the button changes. This module reads those lines on a
background thread and:

  * "ESTOP"                        -> calls on_estop("button")
  * no message for LINK_TIMEOUT s  -> calls on_estop("link lost")   (cable out / Arduino dead)
  * "OK" after an ESTOP            -> calls on_released() ONLY; it never resumes anything.
                                      Reset and resume stay deliberate, separate actions
                                      (world.reset() then world.resume()).

No busy loop: the thread blocks inside serial.readline() (0.2 s timeout) and, while the
Arduino is missing, sleeps a second between reconnect attempts.

Usage (in main_sim.py, once the World exists, before the control panel runs):
    from estop_button import EstopButton
    button = EstopButton(on_estop=lambda why: world.estop(f"Physical e-stop button ({why})"))
    button.start()
    world.add_interlock(button.interlock)     # Reset is refused while the button is held in

Test on its own (prints what the button does, no sim needed):
    python estop_button.py            # auto-detect the port
    python estop_button.py COM3       # or name it
Close the Arduino IDE's Serial Monitor first -- only one program can hold the port.
"""
import sys
import threading
import time

import serial
from serial.tools import list_ports

BAUD = 115200
LINK_TIMEOUT = 0.8          # s without any message from the Arduino -> treat as a stop
RECONNECT_PERIOD = 1.0      # s between attempts to find / open the port

# USB vendor ids of common Arduino boards and clones (official, CH340, FTDI, CP210x)
_ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4, 0x239A}


def find_arduino_port():
    """Best guess at the Arduino's serial port, or None."""
    for p in list_ports.comports():
        text = f"{p.description} {p.manufacturer}".lower()
        if p.vid in _ARDUINO_VIDS or "arduino" in text or "ch340" in text or "usb serial" in text:
            return p.device
    return None


class EstopButton:
    def __init__(self, on_estop, on_released=None, port=None, baud=BAUD,
                 link_timeout=LINK_TIMEOUT):
        self.on_estop = on_estop              # called with a reason string
        self.on_released = on_released        # called when the button is let go (optional)
        self.port = port
        self.baud = baud
        self.link_timeout = link_timeout
        self.pressed = False                  # latest button state reported by the Arduino
        self.connected = False
        self._link_lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="EstopButton", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def interlock(self):
        """For World.add_interlock: a reason string while Reset must be refused, else "".
        Pressed button -> refuse. Link lost after having been connected -> refuse.
        If the Arduino was never found the button simply isn't part of the system."""
        if self.pressed:
            return "the physical e-stop button is still pressed in"
        if self._link_lost:
            return "the physical e-stop button is not connected"
        return ""

    def stop(self):
        self._stop.set()

    # ---------- internals ----------
    def _trip(self, reason):
        try:
            self.on_estop(reason)
        except Exception as exc:              # a failing callback must not kill the reader
            print(f"[estop] on_estop callback failed: {exc!r}")

    def _open(self):
        port = self.port or find_arduino_port()
        if port is None:
            return None
        try:
            ser = serial.Serial(port, self.baud, timeout=0.2)
        except serial.SerialException as exc:
            print(f"[estop] cannot open {port}: {exc} (is the Arduino Serial Monitor still open?)")
            return None
        print(f"[estop] connected to {port}")
        return ser

    def _run(self):
        warned = False
        while not self._stop.is_set():
            ser = self._open()
            if ser is None:
                if not warned:
                    print("[estop] Arduino not found -- physical e-stop NOT active. Retrying...")
                    warned = True
                self._stop.wait(RECONNECT_PERIOD)
                continue
            warned = False
            self.connected = True
            self._link_lost = False
            last_seen = time.monotonic()
            try:
                while not self._stop.is_set():
                    line = ser.readline().decode("ascii", errors="ignore").strip()
                    now = time.monotonic()
                    if line in ("OK", "ESTOP"):
                        last_seen = now
                        if self._link_lost:
                            print("[estop] link to the button is back (still stopped until reset + resume)")
                            self._link_lost = False
                        if line == "ESTOP" and not self.pressed:
                            self.pressed = True
                            print("[estop] BUTTON PRESSED")
                            self._trip("button")
                        elif line == "OK" and self.pressed:
                            self.pressed = False
                            print("[estop] button released (reset + resume still required)")
                            if self.on_released:
                                self.on_released()
                    if now - last_seen > self.link_timeout and not self._link_lost:
                        self._link_lost = True
                        print("[estop] LINK LOST -- stopping")
                        self._trip("link lost")
            except (serial.SerialException, OSError) as exc:
                print(f"[estop] serial error: {exc}")
                if not self._link_lost:
                    self._link_lost = True
                    self._trip("link lost")
            finally:
                self.connected = False
                try:
                    ser.close()
                except Exception:
                    pass
            self._stop.wait(RECONNECT_PERIOD)


if __name__ == "__main__":
    chosen = sys.argv[1] if len(sys.argv) > 1 else None
    print("Listening for the e-stop button. Ctrl+C to quit.")
    EstopButton(on_estop=lambda why: print(f"  -> world.estop()  ({why})"),
                on_released=lambda: print("  -> button released"), port=chosen).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass