"""The user interface.

Design rules applied throughout, because the intended user may be elderly,
may have double vision, and may be doing this alone:

  * One instruction on screen at a time, in short sentences, no jargon.
  * Very large default text, adjustable with obvious A+ / A- buttons.
  * High contrast: near-black on near-white, or white on black.
  * Buttons are large, widely spaced, and labelled with verbs.
  * Nothing starts automatically. Every step waits for a press.
  * A visible countdown, so the person always knows how long is left.
  * "Stop" and "Do this step again" are always reachable.
  * Optional spoken instructions for anyone who cannot read the screen.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

import cv2
import numpy as np

from . import disclaimers, protocols, report, storage
from .landmarks import FaceTracker
from .metrics import measure_frame

try:
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False

try:
    import pyttsx3
    HAVE_TTS = True
except Exception:
    HAVE_TTS = False


BG = "#ffffff"
FG = "#101010"
ACCENT = "#1a4d8f"
DANGER = "#b32020"
CAUTION = "#8a6000"
MUTED = "#4a4a4a"


class Speaker:
    """Optional text to speech. Never blocks the interface."""

    def __init__(self) -> None:
        self.enabled = False
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = None
        if HAVE_TTS:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:  # pragma: no cover - audio hardware
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 145)
        except Exception:
            return
        while True:
            text = self._queue.get()
            if not self.enabled:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def say(self, text: str) -> None:
        if self.enabled and HAVE_TTS:
            self._queue.put(text)


class CameraWorker(threading.Thread):
    """Grabs frames, tracks the face, and measures - off the UI thread."""

    def __init__(self, camera_index: int = 0, model_path: str | None = None):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.model_path = model_path
        self.capture = None
        self.tracker = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_measure = None
        self.recording_key: str | None = None
        self.buffers: dict[str, list] = {}
        self.error: str | None = None
        self.started_ok = threading.Event()
        self._t0 = time.perf_counter()

    def run(self) -> None:
        try:
            self.tracker = FaceTracker(self.model_path)
        except Exception as exc:
            self.error = str(exc)
            self.started_ok.set()
            return

        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            self.error = (
                "Could not open the camera. Check that it is plugged in, that no "
                "other program is using it, and that this program has camera "
                "permission."
            )
            self.started_ok.set()
            return

        # Ask for a good resolution and the highest frame rate the camera allows.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        self.started_ok.set()

        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.02)
                continue
            t = time.perf_counter() - self._t0
            try:
                face = self.tracker.process(frame, t)
                measure = measure_frame(face)
            except Exception:
                continue
            with self._lock:
                self.latest_frame = frame
                self.latest_measure = measure
                if self.recording_key is not None:
                    self.buffers.setdefault(self.recording_key, []).append(measure)

        try:
            self.capture.release()
            self.tracker.close()
        except Exception:
            pass

    def snapshot(self):
        with self._lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return frame, self.latest_measure

    def start_recording(self, key: str) -> None:
        with self._lock:
            self.buffers[key] = []
            self.recording_key = key

    def stop_recording(self) -> None:
        with self._lock:
            self.recording_key = None

    def take_buffers(self) -> dict[str, list]:
        with self._lock:
            data = self.buffers
            self.buffers = {}
            return data

    def stop(self) -> None:
        self._stop.set()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{disclaimers.APP_NAME} {disclaimers.APP_VERSION}")
        self.configure(bg=BG)
        self.geometry("1180x820")
        self.minsize(900, 680)

        self.font_scale = 1.0
        self.speaker = Speaker()
        self.camera: CameraWorker | None = None
        self.session = {
            "app_version": disclaimers.APP_VERSION,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "person_label": "",
            "notes": "",
            "tests": {},
        }
        self.save_dir = storage.DEFAULT_DIR
        self._current = None
        self._after_ids: list[str] = []

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)
        self._build_footer()
        self.show_disclaimer()
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

    # -- shared widget helpers ----------------------------------------------

    def f(self, size: int, bold: bool = False) -> tuple:
        return ("Helvetica", int(size * self.font_scale), "bold" if bold else "normal")

    def clear(self) -> tk.Frame:
        self._cancel_timers()
        for child in self.container.winfo_children():
            child.destroy()
        frame = tk.Frame(self.container, bg=BG)
        frame.pack(fill="both", expand=True, padx=26, pady=18)
        self._current = frame
        return frame

    def big_button(self, parent, text, command, kind="primary", width=None):
        colours = {
            "primary": (ACCENT, "#ffffff"),
            "secondary": ("#e4e9f0", FG),
            "danger": (DANGER, "#ffffff"),
        }[kind]
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=self.f(19, bold=True),
            bg=colours[0],
            fg=colours[1],
            activebackground=colours[0],
            activeforeground=colours[1],
            relief="raised",
            bd=3,
            padx=22,
            pady=14,
            cursor="hand2",
            wraplength=420,
            justify="center",
        )
        if width:
            btn.configure(width=width)
        return btn

    def label(self, parent, text, size=18, bold=False, fg=FG, wrap=980, **kw):
        return tk.Label(
            parent, text=text, font=self.f(size, bold), bg=BG, fg=fg,
            wraplength=wrap, justify="left", **kw
        )

    def _build_footer(self) -> None:
        bar = tk.Frame(self, bg="#f0f0f0", height=46)
        bar.pack(side="bottom", fill="x")
        tk.Label(
            bar, text=disclaimers.PERSISTENT_FOOTER, bg="#f0f0f0", fg=DANGER,
            font=("Helvetica", 12, "bold"),
        ).pack(side="left", padx=14, pady=8)

        tk.Button(bar, text="A-", command=lambda: self.change_font(-0.1),
                  font=("Helvetica", 13, "bold"), width=3).pack(side="right", padx=4, pady=6)
        tk.Button(bar, text="A+", command=lambda: self.change_font(0.1),
                  font=("Helvetica", 13, "bold"), width=3).pack(side="right", padx=4, pady=6)

        self.tts_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bar, text="Read instructions aloud", variable=self.tts_var,
            command=self.toggle_tts, bg="#f0f0f0", font=("Helvetica", 12),
            state="normal" if HAVE_TTS else "disabled",
        ).pack(side="right", padx=12)

    def change_font(self, delta: float) -> None:
        self.font_scale = float(np.clip(self.font_scale + delta, 0.8, 2.0))
        if self._rebuild:
            self._rebuild()

    _rebuild = None

    def toggle_tts(self) -> None:
        self.speaker.enabled = bool(self.tts_var.get())

    @staticmethod
    def _canvas_size(canvas, fallback=(640, 480)) -> tuple[int, int]:
        """Usable canvas size.

        Tk reports a width of 1 until the widget has been laid out, which
        silently collapses any scaling done against it. Fall back to the
        requested size, then to a sane default.
        """
        w, h = int(canvas.winfo_width()), int(canvas.winfo_height())
        if w < 50 or h < 50:
            try:
                w, h = int(canvas.winfo_reqwidth()), int(canvas.winfo_reqheight())
            except Exception:
                w = h = 0
        if w < 50 or h < 50:
            w, h = fallback
        return w, h

    @staticmethod
    def _widget_alive(widget) -> bool:
        try:
            return widget is not None and bool(widget.winfo_exists())
        except Exception:
            return False

    def _after(self, ms: int, fn) -> None:
        self._after_ids.append(self.after(ms, fn))

    def _cancel_timers(self) -> None:
        for aid in self._after_ids:
            try:
                self.after_cancel(aid)
            except Exception:
                pass
        self._after_ids = []

    # -- screen 1: disclaimer -----------------------------------------------

    def show_disclaimer(self) -> None:
        self._rebuild = self.show_disclaimer
        frame = self.clear()
        self.label(frame, "Before you begin", size=30, bold=True, fg=DANGER).pack(anchor="w")

        box = tk.Frame(frame, bg="#fff6f6", highlightbackground=DANGER, highlightthickness=3)
        box.pack(fill="both", expand=True, pady=14)
        text = tk.Text(
            box, wrap="word", font=self.f(15), bg="#fff6f6", fg=FG,
            relief="flat", padx=18, pady=14, height=18,
        )
        text.insert("1.0", disclaimers.STARTUP_DISCLAIMER + "\n" + disclaimers.PRIVACY_NOTE)
        text.configure(state="disabled")
        scroll = tk.Scrollbar(box, command=text.yview, width=22)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        self.agree_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame,
            text="I have read this. I understand this is not a medical test and cannot diagnose anything.",
            variable=self.agree_var, bg=BG, font=self.f(16), wraplength=940,
            justify="left",
        ).pack(anchor="w", pady=10)

        row = tk.Frame(frame, bg=BG)
        row.pack(fill="x")
        self.big_button(row, "Continue", self._accept_disclaimer).pack(side="left")
        self.big_button(row, "Close the program", self.quit_app, kind="secondary").pack(
            side="left", padx=14
        )

    def _accept_disclaimer(self) -> None:
        if not self.agree_var.get():
            messagebox.showwarning(
                "Please confirm",
                "Please tick the box to confirm you have read the notice.",
            )
            return
        self.show_setup()

    # -- screen 2: setup and camera check -----------------------------------

    def show_setup(self) -> None:
        self._rebuild = self.show_setup
        frame = self.clear()
        self.label(frame, "Set up", size=30, bold=True).pack(anchor="w")
        self.label(
            frame,
            "Sit so your face fills the middle of the picture. Put a bright light "
            "in front of you, not behind you. Take off glasses if they reflect. "
            "Rest the camera on something solid so it does not wobble.",
            size=17, fg=MUTED,
        ).pack(anchor="w", pady=(6, 14))

        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True)

        self.preview = tk.Canvas(body, width=640, height=480, bg="#202020",
                                 highlightthickness=2, highlightbackground="#888")
        self.preview.pack(side="left")

        side = tk.Frame(body, bg=BG)
        side.pack(side="left", fill="both", expand=True, padx=20)

        self.status_label = self.label(side, "Starting the camera...", size=18, bold=True, wrap=380)
        self.status_label.pack(anchor="w", pady=(0, 10))
        self.detail_label = self.label(side, "", size=15, fg=MUTED, wrap=380)
        self.detail_label.pack(anchor="w")

        tk.Label(side, text="A name or label for these records:", bg=BG,
                 font=self.f(16)).pack(anchor="w", pady=(24, 4))
        self.name_entry = tk.Entry(side, font=self.f(18), width=22)
        self.name_entry.pack(anchor="w")
        self.name_entry.insert(0, self.session.get("person_label", ""))

        tk.Label(side, text="Where to save (folder):", bg=BG,
                 font=self.f(16)).pack(anchor="w", pady=(18, 4))
        self.dir_entry = tk.Entry(side, font=self.f(13), width=34)
        self.dir_entry.pack(anchor="w")
        self.dir_entry.insert(0, self.save_dir)

        btns = tk.Frame(side, bg=BG)
        btns.pack(anchor="w", pady=26)
        self.continue_btn = self.big_button(btns, "The picture looks good - continue",
                                            self._finish_setup)
        self.continue_btn.pack(anchor="w")
        self.big_button(btns, "Back", self.show_disclaimer, kind="secondary").pack(
            anchor="w", pady=10
        )

        if self.camera is None:
            self.camera = CameraWorker()
            self.camera.start()
        self._poll_setup_preview()

    def _poll_setup_preview(self) -> None:
        if not self._widget_alive(getattr(self, "preview", None)):
            return
        if self.camera and self.camera.error:
            self.status_label.configure(text="Camera problem", fg=DANGER)
            self.detail_label.configure(text=self.camera.error)
            self._after(500, self._poll_setup_preview)
            return

        frame, measure = self.camera.snapshot() if self.camera else (None, None)
        if frame is not None:
            self._draw_preview(self.preview, frame, measure)
            if measure is not None and measure.valid:
                q = measure.quality
                if q > 0.75:
                    self.status_label.configure(text="Face found. Picture is good.", fg="#2a6a3a")
                    self.detail_label.configure(text="")
                elif q > 0.45:
                    self.status_label.configure(text="Face found, but the picture could be better.",
                                                fg=CAUTION)
                    self.detail_label.configure(
                        text="Try moving closer, turning to face the camera squarely, "
                             "or adding more light in front of you."
                    )
                else:
                    self.status_label.configure(text="Picture quality is poor.", fg=DANGER)
                    self.detail_label.configure(
                        text="Move closer to the camera and face it directly. "
                             "Make sure light falls on your face, not behind you."
                    )
            else:
                self.status_label.configure(text="No face found yet.", fg=DANGER)
                self.detail_label.configure(
                    text="Sit in front of the camera so your whole face is visible."
                )
        self._after(60, self._poll_setup_preview)

    def _draw_preview(self, canvas: tk.Canvas, frame: np.ndarray, measure) -> None:
        if not HAVE_PIL:
            canvas.delete("all")
            canvas.create_text(
                320, 240, text="Install Pillow to see the picture\n(pip install pillow)",
                fill="white", font=self.f(15),
            )
            return
        w, h = self._canvas_size(canvas, (640, 480))
        # Mirror so it behaves like a bathroom mirror - far less confusing.
        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        scale = min(w / img.shape[1], h / img.shape[0])
        new_w = max(1, int(img.shape[1] * scale))
        new_h = max(1, int(img.shape[0] * scale))
        img = cv2.resize(img, (new_w, new_h))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.delete("preview")
        canvas.create_image(w // 2, h // 2, image=photo, tags="preview")
        canvas._photo = photo  # keep a reference alive
        colour = "#2ecc71" if (measure and measure.valid and measure.quality > 0.6) else "#e74c3c"
        canvas.create_rectangle(3, 3, w - 3, h - 3, outline=colour, width=4, tags="preview")

    def _finish_setup(self) -> None:
        if self.camera and self.camera.error:
            messagebox.showerror("Camera problem", self.camera.error)
            return
        self.session["person_label"] = self.name_entry.get().strip() or "session"
        self.save_dir = self.dir_entry.get().strip() or storage.DEFAULT_DIR
        self.show_menu()

    # -- screen 3: menu of tests --------------------------------------------

    def show_menu(self) -> None:
        self._rebuild = self.show_menu
        frame = self.clear()
        self.label(frame, "Choose what to record", size=30, bold=True).pack(anchor="w")
        self.label(
            frame,
            "Do them in any order. You can stop at any time. Each one takes one to "
            "three minutes. A green tick means you have already done it today.",
            size=16, fg=MUTED,
        ).pack(anchor="w", pady=(4, 12))

        canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview, width=22)
        inner = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for proto in protocols.PROTOCOLS:
            done = proto.key in self.session["tests"]
            row = tk.Frame(inner, bg="#f5f7fa" if not done else "#eef7ee",
                           highlightbackground="#ccd6e4", highlightthickness=1)
            row.pack(fill="x", pady=5, padx=2)

            left = tk.Frame(row, bg=row["bg"])
            left.pack(side="left", fill="x", expand=True, padx=14, pady=10)
            tk.Label(left, text=("\u2713 " if done else "") + proto.plain_title,
                     bg=row["bg"], fg=FG, font=self.f(18, bold=True),
                     wraplength=620, justify="left").pack(anchor="w")
            tk.Label(left, text=proto.why, bg=row["bg"], fg=MUTED, font=self.f(14),
                     wraplength=620, justify="left").pack(anchor="w", pady=(2, 0))
            if proto.needs_helper:
                tk.Label(left, text="Needs another person to help you",
                         bg=row["bg"], fg=CAUTION, font=self.f(13, bold=True)).pack(anchor="w")

            self.big_button(row, "Redo" if done else "Start",
                            lambda p=proto: self.start_test(p),
                            kind="secondary" if done else "primary").pack(side="right", padx=14)

        bottom = tk.Frame(frame, bg=BG)
        bottom.pack(side="bottom", fill="x", pady=14)
        self.big_button(bottom, "Finish and make a report",
                        self.finish_session).pack(side="left")
        self.big_button(bottom, "Back to setup", self.show_setup,
                        kind="secondary").pack(side="left", padx=12)

    # -- screen 4: running a test -------------------------------------------

    def start_test(self, proto: protocols.Protocol) -> None:
        self._rebuild = None
        self._cancel_timers()
        self.active_proto = proto
        self.phase_index = 0
        self.collected: dict[str, list] = {}
        self.repeat_index = 0
        self._build_test_screen()
        self._show_phase_intro()

    def _build_test_screen(self) -> None:
        frame = self.clear()
        proto = self.active_proto

        header = tk.Frame(frame, bg=BG)
        header.pack(fill="x")
        self.label(header, proto.plain_title, size=24, bold=True, wrap=900).pack(anchor="w")

        self.stage_canvas = tk.Canvas(frame, bg="#101010", highlightthickness=0, height=430)
        self.stage_canvas.pack(fill="both", expand=True, pady=10)

        self.instruction_label = self.label(frame, "", size=22, bold=True, wrap=1040)
        self.instruction_label.pack(anchor="w", pady=(6, 2))
        self.helper_label = self.label(frame, "", size=16, bold=True, fg=DANGER, wrap=1040)
        self.helper_label.pack(anchor="w")
        self.countdown_label = self.label(frame, "", size=34, bold=True, fg=ACCENT)
        self.countdown_label.pack(anchor="w", pady=4)
        self.track_label = self.label(frame, "", size=15, fg=MUTED)
        self.track_label.pack(anchor="w")

        controls = tk.Frame(frame, bg=BG)
        controls.pack(fill="x", pady=10)
        self.action_btn = self.big_button(controls, "Start this step", self._begin_phase)
        self.action_btn.pack(side="left")
        self.big_button(controls, "Do this step again", self._redo_phase,
                        kind="secondary").pack(side="left", padx=10)
        self.big_button(controls, "Stop and go back", self._abort_test,
                        kind="danger").pack(side="right")

        self._poll_stage()

    def _phase(self) -> protocols.Phase:
        return self.active_proto.phases[self.phase_index]

    def _show_phase_intro(self) -> None:
        phase = self._phase()
        proto = self.active_proto
        step = f"Step {self.phase_index + 1} of {len(proto.phases)}"
        if proto.repeats > 1:
            step += f"  (round {self.repeat_index + 1} of {proto.repeats})"
        self.instruction_label.configure(text=f"{step}:  {phase.instruction}")
        self.helper_label.configure(text=phase.helper_action)
        self.countdown_label.configure(text=f"{int(phase.duration_s)} seconds when you start")
        self.action_btn.configure(text="Start this step", state="normal")
        self.speaker.say(phase.instruction)
        self._recording = False

    def _begin_phase(self) -> None:
        phase = self._phase()
        self.action_btn.configure(state="disabled")
        self._recording = phase.record
        if phase.record and self.camera:
            # Repeat rounds append into the same phase buffer on purpose, so
            # the analyser sees all rounds together.
            self.camera.start_recording(phase.key)
        self._phase_end = time.perf_counter() + phase.duration_s
        self._tick_phase()

    def _tick_phase(self) -> None:
        remaining = self._phase_end - time.perf_counter()
        if remaining <= 0:
            self._end_phase()
            return
        seconds = int(np.ceil(remaining))
        self.countdown_label.configure(text=f"{seconds} seconds left")
        if not self._widget_alive(self.countdown_label):
            return
        self._after(100, self._tick_phase)

    def _end_phase(self) -> None:
        phase = self._phase()
        if self.camera:
            self.camera.stop_recording()
            for key, frames in self.camera.take_buffers().items():
                self.collected.setdefault(key, []).extend(frames)
        self.countdown_label.configure(text="Step finished")
        self.speaker.say("Step finished")

        self.phase_index += 1
        if self.phase_index >= len(self.active_proto.phases):
            self.repeat_index += 1
            if self.repeat_index < self.active_proto.repeats:
                self.phase_index = 0
                self._show_phase_intro()
                return
            self._finish_test()
            return
        self._show_phase_intro()

    def _redo_phase(self) -> None:
        self._cancel_timers()
        if self.camera:
            self.camera.stop_recording()
            self.camera.take_buffers()
        phase = self._phase()
        self.collected.pop(phase.key, None)
        self._show_phase_intro()

    def _abort_test(self) -> None:
        self._cancel_timers()
        if self.camera:
            self.camera.stop_recording()
            self.camera.take_buffers()
        self.show_menu()

    def _poll_stage(self) -> None:
        if not self._widget_alive(getattr(self, "stage_canvas", None)):
            return
        canvas = self.stage_canvas
        canvas.delete("all")
        w, h = self._canvas_size(canvas, (900, 430))

        phase = self._phase() if hasattr(self, "active_proto") else None
        target = phase.target if phase else None
        if phase and phase.key == "alternating" and getattr(self, "_recording", False):
            # Alternate the fixation target roughly once a second.
            target = (0.1, 0.5) if int(time.perf_counter()) % 2 == 0 else (0.9, 0.5)

        if target:
            cx, cy = target[0] * w, target[1] * h
            canvas.create_oval(cx - 34, cy - 34, cx + 34, cy + 34,
                               outline="#ffffff", width=3)
            canvas.create_oval(cx - 16, cy - 16, cx + 16, cy + 16,
                               fill="#ffd700", outline="")
            canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#b32020", outline="")
        elif phase and phase.key == "gentle_closure":
            canvas.create_text(w // 2, h // 2, text="Eyes gently closed",
                               fill="#dddddd", font=self.f(26, bold=True))

        frame, measure = self.camera.snapshot() if self.camera else (None, None)
        if frame is not None and HAVE_PIL:
            thumb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            th = 150
            scale = th / max(thumb.shape[0], 1)
            thumb = cv2.resize(thumb, (max(1, int(thumb.shape[1] * scale)), th))
            photo = ImageTk.PhotoImage(Image.fromarray(thumb))
            canvas.create_image(w - thumb.shape[1] // 2 - 12, th // 2 + 12, image=photo)
            canvas._thumb = photo

        if measure is not None:
            if measure.valid:
                ok = measure.quality > 0.5
                self.track_label.configure(
                    text=("Face is being tracked." if ok
                          else "Tracking is poor - move closer and face the camera."),
                    fg="#2a6a3a" if ok else DANGER,
                )
            else:
                self.track_label.configure(text="Face not visible.", fg=DANGER)
        self._after(70, self._poll_stage)

    def _finish_test(self) -> None:
        proto = self.active_proto
        try:
            results = proto.analyse(self.collected)
        except Exception as exc:
            results = {"description": f"Could not analyse this recording ({exc}).",
                       "quality": {"usable": False, "problems": ["analysis failed"]}}
        self.session["tests"][proto.key] = {
            "title": proto.title,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        self._show_test_result(proto, results)

    def _show_test_result(self, proto, results) -> None:
        self._rebuild = None
        frame = self.clear()
        self.label(frame, "What was recorded", size=28, bold=True).pack(anchor="w")
        self.label(frame, proto.plain_title, size=19, fg=MUTED).pack(anchor="w", pady=(2, 12))

        quality = results.get("quality") or {}
        if not quality.get("usable", True):
            box = tk.Frame(frame, bg="#fff8e6", highlightbackground=CAUTION, highlightthickness=2)
            box.pack(fill="x", pady=6)
            tk.Label(box, text="The recording was not good enough to rely on.",
                     bg="#fff8e6", fg=CAUTION, font=self.f(18, bold=True)).pack(anchor="w", padx=14, pady=(10, 2))
            for problem in quality.get("problems", []):
                tk.Label(box, text="- " + problem, bg="#fff8e6", fg=FG,
                         font=self.f(15)).pack(anchor="w", padx=26)
            tk.Label(box, text=disclaimers.LOW_CONFIDENCE_NOTE, bg="#fff8e6", fg=FG,
                     font=self.f(14), wraplength=900, justify="left").pack(anchor="w", padx=14, pady=(4, 12))

        desc = tk.Frame(frame, bg="#f5f7fa", highlightbackground="#ccd6e4", highlightthickness=1)
        desc.pack(fill="x", pady=10)
        tk.Label(desc, text=results.get("description", ""), bg="#f5f7fa", fg=FG,
                 font=self.f(18), wraplength=980, justify="left").pack(anchor="w", padx=16, pady=14)

        caveat = protocols.caveat_for(proto.key)
        if caveat:
            tk.Label(frame, text="Important limitation: " + caveat, bg=BG, fg=CAUTION,
                     font=self.f(15), wraplength=980, justify="left").pack(anchor="w", pady=6)

        tk.Label(
            frame,
            text=(
                "This describes what the camera measured. It does not say whether "
                "anything is wrong. Only a doctor can tell you that."
            ),
            bg=BG, fg=DANGER, font=self.f(16, bold=True), wraplength=980, justify="left",
        ).pack(anchor="w", pady=14)

        row = tk.Frame(frame, bg=BG)
        row.pack(anchor="w", pady=10)
        self.big_button(row, "Back to the list", self.show_menu).pack(side="left")
        self.big_button(row, "Record this one again",
                        lambda: self.start_test(proto), kind="secondary").pack(side="left", padx=12)

    # -- screen 5: finish ----------------------------------------------------

    def finish_session(self) -> None:
        if not self.session["tests"]:
            messagebox.showinfo("Nothing recorded yet",
                                "Record at least one item before making a report.")
            return
        self._rebuild = None
        frame = self.clear()
        self.label(frame, "Finish", size=30, bold=True).pack(anchor="w")
        self.label(frame,
                   "Anything you want the doctor to know? For example how you felt, "
                   "the time of day, or whether you had taken any medicine.",
                   size=16, fg=MUTED).pack(anchor="w", pady=(6, 6))
        notes = tk.Text(frame, height=6, font=self.f(15), wrap="word",
                        highlightbackground="#ccd6e4", highlightthickness=1)
        notes.pack(fill="x", pady=6)

        def do_save() -> None:
            self.session["notes"] = notes.get("1.0", "end").strip()
            try:
                path = storage.save_session(self.session, self.save_dir)
                history = storage.load_sessions(self.save_dir,
                                                self.session.get("person_label"))
                report_path = report.write_report(self.session, self.save_dir, history)
            except Exception as exc:
                messagebox.showerror("Could not save", str(exc))
                return
            self._show_saved(path, report_path)

        row = tk.Frame(frame, bg=BG)
        row.pack(anchor="w", pady=16)
        self.big_button(row, "Save and make the report", do_save).pack(side="left")
        self.big_button(row, "Back", self.show_menu, kind="secondary").pack(side="left", padx=12)

    def _show_saved(self, data_path: str, report_path: str) -> None:
        frame = self.clear()
        self.label(frame, "Saved", size=30, bold=True, fg="#2a6a3a").pack(anchor="w")
        self.label(frame, "Your report has been saved on this computer:",
                   size=18).pack(anchor="w", pady=(8, 4))
        self.label(frame, report_path, size=14, fg=ACCENT, wrap=980).pack(anchor="w")
        self.label(frame, "The measurements were saved as:", size=16,
                   fg=MUTED).pack(anchor="w", pady=(10, 2))
        self.label(frame, data_path, size=13, fg=MUTED, wrap=980).pack(anchor="w")

        self.label(
            frame,
            "Take the report to your doctor. Do not use it to make decisions on "
            "your own, and do not change any medicine because of it.",
            size=17, bold=True, fg=DANGER,
        ).pack(anchor="w", pady=18)

        row = tk.Frame(frame, bg=BG)
        row.pack(anchor="w")
        self.big_button(row, "Open the report",
                        lambda: webbrowser.open("file://" + report_path)).pack(side="left")
        self.big_button(row, "Record more", self.show_menu, kind="secondary").pack(
            side="left", padx=12)
        self.big_button(row, "Close the program", self.quit_app, kind="secondary").pack(
            side="left")

    # -- lifecycle -----------------------------------------------------------

    def quit_app(self) -> None:
        self._cancel_timers()
        if self.camera:
            self.camera.stop()
        try:
            self.destroy()
        except Exception:
            pass


def run() -> None:
    app = App()
    app.mainloop()
