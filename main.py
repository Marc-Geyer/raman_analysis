"""
Raman Data Viewer - Main Launcher
Opens a lightweight file-picker window. Each selected CSV is analysed in its
own independent process so windows never block each other.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import multiprocessing
import os
import sys

import theme as T


# ── helpers ──────────────────────────────────────────────────────────────────

def _launch_analysis(csv_path: str):
    """Target function that runs inside a fresh process."""
    from analysis_window import run_analysis_window
    run_analysis_window(csv_path)


def open_file(listbox: tk.Listbox, status_var: tk.StringVar):
    paths = filedialog.askopenfilenames(
        title="Select Raman CSV file(s)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    for path in paths:
        if path:
            listbox.insert(tk.END, path)
            _start_process(path, status_var)


def _start_process(path: str, status_var: tk.StringVar):
    p = multiprocessing.Process(target=_launch_analysis, args=(path,), daemon=False)
    p.start()
    name = os.path.basename(path)
    status_var.set(f"Opened: {name}  (PID {p.pid})")


# ── main window ───────────────────────────────────────────────────────────────

def build_main_window():
    root = tk.Tk()
    root.title("Raman Viewer – Launcher")
    # root.iconphoto(True, tk.PhotoImage(file="canvas.png"))
    root.geometry("600x400")
    root.minsize(480, 320)
    root.configure(bg=T.BG_APP)

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TFrame",
                    background=T.BG_APP)
    style.configure("TLabel",
                    background=T.BG_APP,
                    foreground=T.FG_LABEL,
                    font=(T.FONT_MONO, T.FONT_SIZE_BODY))
    style.configure("Title.TLabel",
                    background=T.BG_APP,
                    foreground=T.FG_TITLE,
                    font=(T.FONT_MONO, 18, "bold"))
    style.configure("Sub.TLabel",
                    background=T.BG_APP,
                    foreground=T.FG_SUBTLE,
                    font=(T.FONT_MONO, T.FONT_SIZE_SMALL))
    style.configure("Open.TButton",
                    font=(T.FONT_MONO, T.FONT_SIZE_BODY, "bold"),
                    foreground=T.FG_ON_ACCENT,
                    background=T.ACCENT,
                    borderwidth=0,
                    focusthickness=0,
                    padding=(16, 8))
    style.map("Open.TButton",
              background=[("active", T.ACCENT_HOVER),
                          ("pressed", T.ACCENT_PRESSED)])

    # ── header ──
    hdr = ttk.Frame(root, padding=(24, 20, 24, 8))
    hdr.pack(fill=tk.X)
    ttk.Label(hdr, text="⬡  RAMAN VIEWER",
              style="Title.TLabel").pack(anchor=tk.W)
    ttk.Label(hdr,
              text="Each CSV opens in its own independent analysis window",
              style="Sub.TLabel").pack(anchor=tk.W, pady=(2, 0))

    sep = ttk.Separator(root, orient=tk.HORIZONTAL)
    sep.pack(fill=tk.X, padx=24)

    # ── button area ──
    btn_frame = ttk.Frame(root, padding=(24, 16))
    btn_frame.pack(fill=tk.X)

    status_var = tk.StringVar(value="No file opened yet.")

    open_btn = ttk.Button(
        btn_frame, text="＋  Open CSV file(s)",
        style="Open.TButton",
        command=lambda: open_file(listbox, status_var),
    )
    open_btn.pack(side=tk.LEFT)

    ttk.Label(btn_frame,
              text="  or drag files into the list below",
              style="Sub.TLabel").pack(side=tk.LEFT, padx=8)

    # ── recent files list ──
    list_frame = ttk.Frame(root, padding=(24, 0, 24, 8))
    list_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(list_frame, text="Opened this session:",
              style="Sub.TLabel").pack(anchor=tk.W)

    lb_frame = tk.Frame(list_frame, bg=T.COLOR_SPINE, bd=1, relief=tk.SOLID)
    lb_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    scrollbar = tk.Scrollbar(
        lb_frame, orient=tk.VERTICAL,
        bg=T.BG_APP, troughcolor=T.BG_LISTBOX,
        activebackground=T.ACCENT,
    )
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(
        lb_frame, yscrollcommand=scrollbar.set,
        bg=T.BG_LISTBOX,
        fg=T.LB_FG,
        selectbackground=T.LB_SELECT_BG,
        selectforeground=T.LB_SELECT_FG,
        font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
        borderwidth=0, highlightthickness=0, activestyle="none",
    )
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    def on_double_click(event):
        sel = listbox.curselection()
        if sel:
            path = listbox.get(sel[0])
            if os.path.exists(path):
                _start_process(path, status_var)
            else:
                messagebox.showerror("File not found",
                                     f"Cannot find:\n{path}")

    listbox.bind("<Double-Button-1>", on_double_click)

    # ── status bar ──
    status_bar = tk.Label(
        root, textvariable=status_var,
        bg=T.BG_STATUS, fg=T.FG_STATUS,
        font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
        anchor=tk.W, padx=12, pady=4,
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    root.bind("<Control-o>", lambda e: open_file(listbox, status_var))
    root.bind("<Return>",    lambda e: open_file(listbox, status_var))

    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn")
    build_main_window()