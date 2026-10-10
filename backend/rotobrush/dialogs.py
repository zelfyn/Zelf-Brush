"""Native file dialogs via Tk (bundled with Python on Windows). Runs in its own process."""
from __future__ import annotations

import json
import sys


def run(kind: str, name: str = "", ext: str = "") -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if kind == "video":
        p = filedialog.askopenfilename(title="Import video", filetypes=[("Video", "*.mp4 *.mkv *.mov"), ("All files", "*.*")])
    elif kind == "project":
        p = filedialog.askopenfilename(title="Open project", filetypes=[("RotoBrush project", "*.roto")])
    elif kind == "folder":
        p = filedialog.askdirectory(title="Choose folder")
    else:
        p = filedialog.asksaveasfilename(title="Export to", initialfile=name, defaultextension=ext)
    root.destroy()
    return p or ""


if __name__ == "__main__":
    print(json.dumps(run(*sys.argv[1:4])))
