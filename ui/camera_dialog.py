"""Small Tk camera preview used by the media chooser."""
import tkinter as tk

import cv2
from PIL import Image, ImageTk


def capture_image(parent, camera_index=0):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Unable to open the camera")

    window = tk.Toplevel(parent)
    window.title("Tarz camera")
    window.resizable(False, False)
    window.attributes("-topmost", True)
    preview = tk.Label(window)
    preview.pack(padx=10, pady=10)
    captured = {"data": None}
    closed = {"value": False}
    timer = {"id": None}

    def close():
        if closed["value"]:
            return
        closed["value"] = True
        if timer["id"] is not None:
            try:
                window.after_cancel(timer["id"])
            except tk.TclError:
                pass
        camera.release()
        window.destroy()

    def update_preview():
        if closed["value"]:
            return
        ok, frame = camera.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            preview.image = ImageTk.PhotoImage(image)
            preview.configure(image=preview.image)
        timer["id"] = window.after(30, update_preview)

    def capture():
        ok, frame = camera.read()
        if not ok:
            return
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            captured["data"] = encoded.tobytes()
        close()

    tk.Button(window, text="Capture", width=18, command=capture).pack(pady=(0, 8))
    tk.Button(window, text="Cancel", width=18, command=close).pack(pady=(0, 12))
    window.protocol("WM_DELETE_WINDOW", close)
    update_preview()
    parent.wait_window(window)
    return captured["data"]