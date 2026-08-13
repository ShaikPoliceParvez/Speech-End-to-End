"""Compact upload chooser shown before every voice or text turn."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from ui.camera_dialog import capture_image


MAX_UPLOAD_IMAGES = 3
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".pdf"}


class MediaChooser:
    def choose(self):
        root = tk.Tk()
        root.title("Tarz upload")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        selected = []

        def finish(paths=None):
            if paths:
                if isinstance(paths, (bytes, Path)):
                    selected.append(paths)
                else:
                    selected.extend(paths)
            root.destroy()

        def upload_files():
            paths = filedialog.askopenfilenames(
                parent=root,
                title="Choose up to 3 images or one document",
                filetypes=[
                    ("Images and PDF", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.pdf"),
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                    ("PDF documents", "*.pdf"),
                ],
            )
            if not paths:
                return

            paths = [Path(path) for path in paths]
            documents = [path for path in paths if path.suffix.lower() in DOCUMENT_EXTENSIONS]
            images = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
            if documents and (len(paths) != 1 or documents[0].stat().st_size > MAX_DOCUMENT_BYTES):
                messagebox.showerror("Upload limit", "Choose one document up to 20 MB.", parent=root)
                return
            if not documents and (not images or len(images) > MAX_UPLOAD_IMAGES):
                messagebox.showerror("Upload limit", "Choose up to 3 images at once.", parent=root)
                return
            finish(images or documents)

        def open_camera():
            try:
                image_bytes = capture_image(root)
                if image_bytes:
                    finish(image_bytes)
            except Exception as error:
                messagebox.showerror("Camera", str(error), parent=root)

        tk.Label(root, text="Upload media before your question").pack(padx=24, pady=(18, 10))
        tk.Button(root, text="Camera", width=18, command=open_camera).pack(pady=4)
        tk.Button(root, text="Upload", width=18, command=upload_files).pack(pady=4)
        tk.Button(root, text="Continue without media", width=18, command=finish).pack(pady=(4, 18))
        root.protocol("WM_DELETE_WINDOW", finish)
        root.mainloop()

        if not selected:
            return None
        print("\nUploaded media:")
        for index, path in enumerate(selected, start=1):
            if isinstance(path, bytes):
                print(f"  [OK] camera_capture_{index}.jpg (in memory)")
                continue
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {path.name} ({size_mb:.2f} MB)")
        print("Now ask your question about the uploaded media.")
        return selected