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


# ── helpers ──────────────────────────────────────────────────────────────────

def _launch_analysis(csv_path: str):
    """Target function that runs inside a fresh process."""
    # Import here so the child process loads modules independently
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
    root.geometry("600x400")
    root.minsize(480, 320)
    root.configure(bg="#1a1a2e")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#1a1a2e")
    style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0",
                    font=("Courier New", 11))
    style.configure("Title.TLabel", background="#1a1a2e", foreground="#00d4ff",
                    font=("Courier New", 18, "bold"))
    style.configure("Sub.TLabel", background="#1a1a2e", foreground="#888",
                    font=("Courier New", 9))
    style.configure("Open.TButton", font=("Courier New", 11, "bold"),
                    foreground="#1a1a2e", background="#00d4ff",
                    borderwidth=0, focusthickness=0, padding=(16, 8))
    style.map("Open.TButton",
              background=[("active", "#00aacc"), ("pressed", "#007fa0")])
    style.configure("TListbox", background="#0d0d1a", foreground="#b0d4e0",
                    font=("Courier New", 9))

    # ── header ──
    hdr = ttk.Frame(root, padding=(24, 20, 24, 8))
    hdr.pack(fill=tk.X)
    ttk.Label(hdr, text="⬡  RAMAN VIEWER", style="Title.TLabel").pack(anchor=tk.W)
    ttk.Label(hdr, text="Each CSV opens in its own independent analysis window",
              style="Sub.TLabel").pack(anchor=tk.W, pady=(2, 0))

    ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=24)

    # ── drop-hint / button area ──
    btn_frame = ttk.Frame(root, padding=(24, 16))
    btn_frame.pack(fill=tk.X)

    status_var = tk.StringVar(value="No file opened yet.")

    open_btn = ttk.Button(
        btn_frame, text="＋  Open CSV file(s)",
        style="Open.TButton",
        command=lambda: open_file(listbox, status_var),
    )
    open_btn.pack(side=tk.LEFT)

    ttk.Label(btn_frame, text="  or drag files into the list below",
              style="Sub.TLabel").pack(side=tk.LEFT, padx=8)

    # ── recent files list ──
    list_frame = ttk.Frame(root, padding=(24, 0, 24, 8))
    list_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(list_frame, text="Opened this session:",
              style="Sub.TLabel").pack(anchor=tk.W)

    lb_frame = tk.Frame(list_frame, bg="#0d0d1a", bd=1, relief=tk.SOLID)
    lb_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    scrollbar = tk.Scrollbar(lb_frame, orient=tk.VERTICAL,
                              bg="#1a1a2e", troughcolor="#0d0d1a",
                              activebackground="#00d4ff")
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(
        lb_frame, yscrollcommand=scrollbar.set,
        bg="#0d0d1a", fg="#7ecfea", selectbackground="#00334d",
        selectforeground="#ffffff", font=("Courier New", 9),
        borderwidth=0, highlightthickness=0, activestyle="none",
    )
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    # Re-open on double-click
    def on_double_click(event):
        sel = listbox.curselection()
        if sel:
            path = listbox.get(sel[0])
            if os.path.exists(path):
                _start_process(path, status_var)
            else:
                messagebox.showerror("File not found", f"Cannot find:\n{path}")

    listbox.bind("<Double-Button-1>", on_double_click)

    # ── status bar ──
    status_bar = tk.Label(root, textvariable=status_var,
                          bg="#0d0d1a", fg="#555", font=("Courier New", 8),
                          anchor=tk.W, padx=12, pady=4)
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── keyboard shortcut ──
    root.bind("<Control-o>", lambda e: open_file(listbox, status_var))
    root.bind("<Return>",    lambda e: open_file(listbox, status_var))

    root.mainloop()


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    build_main_window()
