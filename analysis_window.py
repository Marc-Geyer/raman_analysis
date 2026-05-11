"""
Raman Data Viewer – Analysis Window
====================================
Displays a 2-D Raman dataset (intensity vs wavenumber × time) as an
interactive heatmap with draggable / arrow-key-tunable slice selectors.

CSV format expected
-------------------
  • Row 0 (header): first cell ignored, remaining cells = time values
  • Rows 1-N:  first cell = wavenumber,  remaining cells = intensity values

Layout
------
  ┌─────────────────────────────┬──────────────────────────┐
  │  Heatmap  (canvas)          │  Spectrum plot (right)   │
  │  – draggable vertical lines │  intensity vs wavenumber │
  │  – draggable horiz. lines   ├──────────────────────────┤
  │                             │  Time trace (right-bot)  │
  │                             │  intensity vs time       │
  └─────────────────────────────┴──────────────────────────┘
  Bottom toolbar: add / remove / recolor lines, colormap picker, crosshair info
"""

from __future__ import annotations
import os
import sys
import tkinter as tk
import traceback
from tkinter import ttk, messagebox, colorchooser
from typing import Optional
import csv

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ── color cycle for slices ────────────────────────────────────────────────────
SLICE_COLORS = [
    "#00d4ff", "#ff6b6b", "#69ff47", "#ffd700", "#bf5fff",
    "#ff9100", "#00e5ff", "#f06292", "#aeff00", "#ff4081",
]


def _next_color(used: list[str]) -> str:
    for c in SLICE_COLORS:
        if c not in used:
            return c
    return SLICE_COLORS[len(used) % len(SLICE_COLORS)]


# ── CSV loader ────────────────────────────────────────────────────────────────
def filter_positive_times(
    times: np.ndarray,
    intensities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes all columns where time < 0.
    Keeps times and intensities aligned.
    """

    mask = times >= 0

    filtered_times = times[mask]

    if intensities.ndim == 2:
        filtered_intensities = intensities[:, mask]
    else:
        filtered_intensities = intensities

    return filtered_times, filtered_intensities

def load_raman_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (wavenumbers [N], times [M], intensity [N×M]).
    Tries to be permissive about separators and decimal characters.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
    delimiter = "," if sample.count(",") >= sample.count(";") else ";"

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        rows = [r for r in reader if any(c.strip() for c in r)]

    def to_float(s):
        return float(s.strip().replace(",", "."))

    header = rows[0]
    times = np.array([to_float(c) for c in header[1:] if c.strip()])

    wavenumbers, intensities = [], []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        try:
            wn = to_float(row[0])
        except ValueError:
            continue
        vals = []
        for c in row[1: 1 + len(times)]:
            try:
                vals.append(to_float(c))
            except ValueError:
                vals.append(np.nan)
        if vals:
            wavenumbers.append(wn)
            intensities.append(vals)

    wavenumbers = np.array(wavenumbers, dtype=float)
    intensities = np.array(intensities, dtype=float)
    # Guarantee shape [N_wn, N_t]
    if intensities.ndim == 2 and intensities.shape[1] != len(times):
        intensities = intensities[:, : len(times)]

    times, intensities = filter_positive_times(
        times,
        intensities
    )

    return wavenumbers, times, intensities


# ── Slice descriptor ──────────────────────────────────────────────────────────

class Slice:
    """Represents one draggable selector line on the heatmap."""
    _id_counter = 0

    def __init__(self, kind: str, index: int, color: str, label: str = ""):
        Slice._id_counter += 1
        self.uid = Slice._id_counter
        self.kind = kind        # "vertical" | "horizontal"
        self.index = index      # index into wavenumbers or times array
        self.color = color
        self.label = label or f"{'W' if kind=='vertical' else 'T'}{self.uid}"
        self.line_obj: Optional[Line2D] = None   # matplotlib line on heatmap
        self.visible = True


# ── Analysis Window ───────────────────────────────────────────────────────────

class AnalysisWindow:
    CMAPS = ["inferno", "viridis", "plasma", "magma", "turbo",
             "hot", "gray", "RdYlBu_r", "seismic"]

    def __init__(self, root: tk.Tk, path: str):
        self.root = root
        self.path = path
        self.fname = os.path.basename(path)

        self.wavenumbers: np.ndarray = np.array([])
        self.times: np.ndarray = np.array([])
        self.intensity: np.ndarray = np.array([])

        self.slices: list[Slice] = []          # all selectors
        self._drag_slice: Optional[Slice] = None
        self._drag_start_xy: tuple = (0, 0)
        self._focused_slice: Optional[Slice] = None
        self._cmap = "inferno"
        self._heatmap_img = None

        self._build_ui()
        self._load_data()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root
        r.title(f"Raman Analysis – {self.fname}")
        r.geometry("1280x780")
        r.minsize(900, 600)
        r.configure(bg="#0f0f1a")

        # ── top toolbar ──
        toolbar = tk.Frame(r, bg="#16162a", pady=4)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(toolbar, text=f"  {self.fname}", bg="#16162a",
                 fg="#00d4ff", font=("Courier New", 10, "bold")).pack(side=tk.LEFT)

        # colormap picker
        tk.Label(toolbar, text="  cmap:", bg="#16162a",
                 fg="#888", font=("Courier New", 9)).pack(side=tk.LEFT)
        self._cmap_var = tk.StringVar(value=self._cmap)
        cmap_menu = ttk.Combobox(toolbar, textvariable=self._cmap_var,
                                 values=self.CMAPS, width=10, state="readonly")
        cmap_menu.pack(side=tk.LEFT, padx=(2, 12))
        cmap_menu.bind("<<ComboboxSelected>>", lambda e: self._refresh_heatmap())

        # add slice buttons
        tk.Button(toolbar, text="＋ Spectrum slice (V)",
                  bg="#003344", fg="#00d4ff", font=("Courier New", 9),
                  relief=tk.FLAT, padx=8,
                  command=self._add_vertical_slice).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="＋ Time trace (H)",
                  bg="#003328", fg="#69ff47", font=("Courier New", 9),
                  relief=tk.FLAT, padx=8,
                  command=self._add_horizontal_slice).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="✕ Remove selected",
                  bg="#330000", fg="#ff6b6b", font=("Courier New", 9),
                  relief=tk.FLAT, padx=8,
                  command=self._remove_focused_slice).pack(side=tk.LEFT, padx=2)

        # crosshair readout
        self._info_var = tk.StringVar(value="")
        tk.Label(toolbar, textvariable=self._info_var, bg="#16162a",
                 fg="#aaa", font=("Courier New", 9)).pack(side=tk.RIGHT, padx=12)

        # ── main pane: heatmap left, plots right ──
        paned = tk.PanedWindow(r, orient=tk.HORIZONTAL,
                               bg="#0f0f1a", sashwidth=6,
                               sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True)

        # left – heatmap figure
        left = tk.Frame(paned, bg="#0f0f1a")
        paned.add(left, minsize=400)

        self._fig_heat, self._ax_heat = plt.subplots(
            figsize=(6, 5), facecolor="#0f0f1a")
        self._ax_heat.set_facecolor("#0f0f1a")
        self._ax_heat.tick_params(colors="#aaa", labelsize=8)
        for sp in self._ax_heat.spines.values():
            sp.set_edgecolor("#333")
        self._ax_heat.set_xlabel("Time", color="#aaa", fontsize=9)
        self._ax_heat.set_ylabel("Wavenumber (cm⁻¹)", color="#aaa", fontsize=9)
        self._ax_heat.set_title("Raman Intensity Heatmap",
                                color="#ccc", fontsize=10)

        self._canvas_heat = FigureCanvasTkAgg(self._fig_heat, master=left)
        self._canvas_heat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # heatmap mouse events
        self._fig_heat.canvas.mpl_connect("button_press_event",
                                          self._on_heat_press)
        self._fig_heat.canvas.mpl_connect("motion_notify_event",
                                          self._on_heat_motion)
        self._fig_heat.canvas.mpl_connect("button_release_event",
                                          self._on_heat_release)
        self._fig_heat.canvas.mpl_connect("scroll_event",
                                          self._on_heat_scroll)

        # right – two stacked plot panels
        right = tk.Frame(paned, bg="#0f0f1a")
        paned.add(right, minsize=340)

        right_paned = tk.PanedWindow(right, orient=tk.VERTICAL,
                                     bg="#0f0f1a", sashwidth=5)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # spectrum panel (top-right)
        spec_frame = tk.Frame(right_paned, bg="#0f0f1a")
        right_paned.add(spec_frame, minsize=200)

        self._fig_spec, self._ax_spec = plt.subplots(
            figsize=(4, 3), facecolor="#0f0f1a")
        self._ax_spec.set_facecolor("#111120")
        self._ax_spec.tick_params(colors="#aaa", labelsize=8)
        for sp in self._ax_spec.spines.values():
            sp.set_edgecolor("#222")
        self._ax_spec.set_xlabel("Wavenumber (cm⁻¹)", color="#aaa", fontsize=8)
        self._ax_spec.set_ylabel("Intensity", color="#aaa", fontsize=8)
        self._ax_spec.set_title("Spectra (vertical slices)",
                                color="#ccc", fontsize=9)

        self._canvas_spec = FigureCanvasTkAgg(self._fig_spec, master=spec_frame)
        self._canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # time trace panel (bottom-right)
        time_frame = tk.Frame(right_paned, bg="#0f0f1a")
        right_paned.add(time_frame, minsize=200)

        self._fig_time, self._ax_time = plt.subplots(
            figsize=(4, 3), facecolor="#0f0f1a")
        self._ax_time.set_facecolor("#111120")
        self._ax_time.tick_params(colors="#aaa", labelsize=8)
        for sp in self._ax_time.spines.values():
            sp.set_edgecolor("#222")
        self._ax_time.set_xlabel("Time", color="#aaa", fontsize=8)
        self._ax_time.set_ylabel("Intensity", color="#aaa", fontsize=8)
        self._ax_time.set_title("Time traces (horizontal slices)",
                                color="#ccc", fontsize=9)

        self._canvas_time = FigureCanvasTkAgg(self._fig_time, master=time_frame)
        self._canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── slice list panel (bottom) ──
        slice_panel = tk.Frame(r, bg="#12121e", pady=4)
        slice_panel.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(slice_panel, text=" Active slices:",
                 bg="#12121e", fg="#555", font=("Courier New", 8)).pack(side=tk.LEFT)

        self._slice_list_frame = tk.Frame(slice_panel, bg="#12121e")
        self._slice_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # keyboard bindings
        r.bind("<Left>",  lambda e: self._nudge_focused(-1))
        r.bind("<Right>", lambda e: self._nudge_focused(+1))
        r.bind("<Up>",    lambda e: self._nudge_focused(+1))
        r.bind("<Down>",  lambda e: self._nudge_focused(-1))
        r.bind("<Delete>", lambda e: self._remove_focused_slice())
        r.bind("<Escape>", lambda e: self._set_focused(None))

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        try:
            wn, t, Z = load_raman_csv(self.path)
        except Exception as exc:
            messagebox.showerror("Load error",
                                 f"Could not read file:\n{self.path}\n\n{exc}",
                                 parent=self.root)
            self.root.destroy()
            traceback.print_exc()
            return

        self.wavenumbers = wn
        self.times = t
        self.intensity = Z

        self._refresh_heatmap()
        # Add one vertical and one horizontal slice by default
        self._add_vertical_slice()
        self._add_horizontal_slice()

    # ── heatmap ───────────────────────────────────────────────────────────────

    def _refresh_heatmap(self):
        if self.intensity.size == 0:
            return
        self._cmap = self._cmap_var.get()
        ax = self._ax_heat
        ax.cla()
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.set_xlabel("Time", color="#aaa", fontsize=9)
        ax.set_ylabel("Wavenumber (cm⁻¹)", color="#aaa", fontsize=9)
        ax.set_title("Raman Intensity Heatmap", color="#ccc", fontsize=10)

        # extent: [xmin, xmax, ymin, ymax]
        t0, t1 = self.times[0], self.times[-1]
        w0, w1 = self.wavenumbers[0], self.wavenumbers[-1]
        self._heatmap_img = ax.imshow(
            self.intensity,
            aspect="auto",
            origin="lower" if w1 > w0 else "upper",
            extent=[t0, t1, w0, w1],
            cmap=self._cmap,
            interpolation="nearest",
        )
        # Redraw existing slice lines
        for s in self.slices:
            s.line_obj = None
            self._draw_slice_line(s)

        self._canvas_heat.draw_idle()

    def _draw_slice_line(self, s: Slice):

        focused = s is self._focused_slice

        lw = 2.0 if focused else 1.2
        ls = "-" if focused else "--"

        if s.kind == "vertical":
            val = self.times[s.index]
            if s.line_obj is None:
                s.line_obj = self._ax_heat.axvline(
                    val,
                    color=s.color,
                    lw=lw,
                    ls=ls
                )
            else:
                s.line_obj.set_xdata([val, val])

        else:
            val = self.wavenumbers[s.index]

            if s.line_obj is None:
                s.line_obj = self._ax_heat.axhline(
                    val,
                    color=s.color,
                    lw=lw,
                    ls=ls
                )
            else:
                s.line_obj.set_ydata([val, val])

        s.line_obj.set_color(s.color)
        s.line_obj.set_linewidth(lw)
        s.line_obj.set_linestyle(ls)
        s.line_obj.set_visible(s.visible)

    # ── slice management ──────────────────────────────────────────────────────

    def _used_colors(self):
        return [s.color for s in self.slices]

    def _add_vertical_slice(self):
        idx = len(self.times) // 2 if self.times.size else 0
        color = _next_color(self._used_colors())
        s = Slice("vertical", idx, color)
        self.slices.append(s)
        self._draw_slice_line(s)
        self._set_focused(s)
        self._rebuild_slice_list()
        self._refresh_spectrum_plot()

    def _add_horizontal_slice(self):
        idx = len(self.wavenumbers) // 2 if self.wavenumbers.size else 0
        color = _next_color(self._used_colors())
        s = Slice("horizontal", idx, color)
        self.slices.append(s)
        self._draw_slice_line(s)
        self._set_focused(s)
        self._rebuild_slice_list()
        self._refresh_time_plot()

    def _remove_focused_slice(self):
        if self._focused_slice is None:
            return
        s = self._focused_slice
        if s.line_obj is not None:
            try:
                s.line_obj.remove()
            except Exception:
                pass
        self.slices.remove(s)
        self._set_focused(None)
        self._canvas_heat.draw_idle()
        self._rebuild_slice_list()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()

    def _set_focused(self, s: Optional[Slice]):
        prev = self._focused_slice
        self._focused_slice = s
        for sl in [prev, s]:
            if sl is not None:
                self._draw_slice_line(sl)
        self._rebuild_slice_list()

    def _nudge_focused(self, delta: int):
        s = self._focused_slice
        if s is None:
            return
        arr = self.times if s.kind == "vertical" else self.wavenumbers
        s.index = int(np.clip(s.index + delta, 0, len(arr) - 1))
        self._draw_slice_line(s)
        self._canvas_heat.draw_idle()
        if s.kind == "vertical":
            self._refresh_spectrum_plot()
        else:
            self._refresh_time_plot()
        self._rebuild_slice_list()

    # ── heatmap mouse interaction ─────────────────────────────────────────────

    def _nearest_slice(self, event) -> Optional[Slice]:
        """Return the slice line closest to the click, within a pixel threshold."""
        if event.xdata is None or event.ydata is None:
            return None
        ax = self._ax_heat
        best, best_dist = None, np.inf
        # convert data coords to display pixels for proper distance
        for s in self.slices:
            if s.kind == "vertical":
                data_val = self.times[s.index]
                # x-distance in data units, but compare in pixels
                disp = ax.transData.transform((data_val, event.ydata))
                click = ax.transData.transform((event.xdata, event.ydata))
                dist = abs(disp[0] - click[0])
            else:
                data_val = self.wavenumbers[s.index]
                disp = ax.transData.transform((event.xdata, data_val))
                click = ax.transData.transform((event.xdata, event.ydata))
                dist = abs(disp[1] - click[1])
            if dist < best_dist:
                best_dist = dist
                best = s
        return best if best_dist < 12 else None  # 12 px threshold

    def _on_heat_press(self, event):
        if event.inaxes != self._ax_heat or event.xdata is None:
            return
        s = self._nearest_slice(event)
        if s is not None:
            self._drag_slice = s
            self._set_focused(s)
        elif event.dblclick:
            # Double-click on empty area: no action (reserved for future use)
            pass

    def _on_heat_motion(self, event):
        if event.inaxes != self._ax_heat or event.xdata is None:
            return
        # update crosshair info
        ti = np.searchsorted(self.times, event.xdata)
        ti = int(np.clip(ti, 0, len(self.times) - 1))
        wi = np.searchsorted(self.wavenumbers, event.ydata)
        wi = int(np.clip(wi, 0, len(self.wavenumbers) - 1))
        val = self.intensity[wi, ti] if self.intensity.size else 0
        self._info_var.set(
            f"t={self.times[ti]:.3g}  wn={self.wavenumbers[wi]:.1f}  I={val:.4g}")

        if self._drag_slice is not None:
            s = self._drag_slice
            if s.kind == "vertical":
                idx = int(np.clip(
                    np.searchsorted(self.times, event.xdata),
                    0, len(self.times) - 1))
            else:
                idx = int(np.clip(
                    np.searchsorted(self.wavenumbers, event.ydata),
                    0, len(self.wavenumbers) - 1))
            if idx != s.index:
                s.index = idx
                self._draw_slice_line(s)
                self._canvas_heat.draw_idle()
                if s.kind == "vertical":
                    self._refresh_spectrum_plot()
                else:
                    self._refresh_time_plot()

    def _on_heat_release(self, event):
        self._drag_slice = None
        self._rebuild_slice_list()

    def _on_heat_scroll(self, event):
        """Scroll to nudge focused slice."""
        delta = 1 if event.button == "up" else -1
        self._nudge_focused(delta)

    # ── right-panel plots ─────────────────────────────────────────────────────

    def _refresh_spectrum_plot(self):
        ax = self._ax_spec
        ax.cla()
        ax.set_facecolor("#111120")
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222")
        ax.set_xlabel("Wavenumber (cm⁻¹)", color="#aaa", fontsize=8)
        ax.set_ylabel("Intensity", color="#aaa", fontsize=8)
        ax.set_title("Spectra (vertical slices)", color="#ccc", fontsize=9)

        vslices = [s for s in self.slices if s.kind == "vertical"]
        for s in vslices:
            if not s.visible or self.intensity.size == 0:
                continue
            idx = s.index
            spectrum = self.intensity[:, idx]
            ax.plot(self.wavenumbers, spectrum, color=s.color, lw=1.2,
                    label=f"{s.label}  t={self.times[idx]:.3g}")

        if vslices:
            ax.legend(fontsize=7, facecolor="#1a1a2e",
                      labelcolor="#ccc", edgecolor="#333")
        self._canvas_spec.draw_idle()

    def _refresh_time_plot(self):
        ax = self._ax_time
        ax.cla()
        ax.set_facecolor("#111120")
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#222")
        ax.set_xlabel("Time", color="#aaa", fontsize=8)
        ax.set_ylabel("Intensity", color="#aaa", fontsize=8)
        ax.set_title("Time traces (horizontal slices)", color="#ccc", fontsize=9)

        hslices = [s for s in self.slices if s.kind == "horizontal"]
        for s in hslices:
            if not s.visible or self.intensity.size == 0:
                continue
            idx = s.index
            trace = self.intensity[idx, :]
            ax.plot(self.times, trace, color=s.color, lw=1.2,
                    label=f"{s.label}  wn={self.wavenumbers[idx]:.1f}")

        if hslices:
            ax.legend(fontsize=7, facecolor="#1a1a2e",
                      labelcolor="#ccc", edgecolor="#333")
        self._canvas_time.draw_idle()

    # ── slice list bar ────────────────────────────────────────────────────────

    def _rebuild_slice_list(self):
        for w in self._slice_list_frame.winfo_children():
            w.destroy()

        for s in self.slices:
            frame = tk.Frame(self._slice_list_frame, bg="#12121e",
                             padx=3, pady=1)
            frame.pack(side=tk.LEFT)

            focused = s is self._focused_slice
            border_color = s.color if focused else "#333"
            inner = tk.Frame(frame, bg=border_color, padx=1, pady=1)
            inner.pack()

            row = tk.Frame(inner, bg="#1a1a2e")
            row.pack()

            # color swatch (clickable → open color picker)
            swatch = tk.Label(row, bg=s.color, width=2, cursor="hand2")
            swatch.pack(side=tk.LEFT)
            swatch.bind("<Button-1>", lambda e, sl=s: self._pick_color(sl))

            # label
            kind_icon = "│" if s.kind == "vertical" else "─"
            arr = self.times if s.kind == "vertical" else self.wavenumbers
            val = arr[s.index] if arr.size else 0
            unit = "t" if s.kind == "vertical" else "wn"
            lbl_text = f" {kind_icon}{s.label}={val:.3g} "
            lbl = tk.Label(row, text=lbl_text, bg="#1a1a2e",
                           fg=s.color, font=("Courier New", 8),
                           cursor="hand2")
            lbl.pack(side=tk.LEFT)
            lbl.bind("<Button-1>", lambda e, sl=s: self._set_focused(sl))

            # visibility toggle
            eye = "◉" if s.visible else "○"
            vis_btn = tk.Label(row, text=eye, bg="#1a1a2e",
                               fg="#555", font=("Courier New", 8),
                               cursor="hand2")
            vis_btn.pack(side=tk.LEFT)
            vis_btn.bind("<Button-1>", lambda e, sl=s: self._toggle_visible(sl))

    def _pick_color(self, s: Slice):
        color = colorchooser.askcolor(color=s.color,
                                      title=f"Color for {s.label}",
                                      parent=self.root)
        if color and color[1]:
            s.color = color[1]
            self._draw_slice_line(s)
            self._canvas_heat.draw_idle()
            self._refresh_spectrum_plot()
            self._refresh_time_plot()
            self._rebuild_slice_list()

    def _toggle_visible(self, s: Slice):
        s.visible = not s.visible
        self._draw_slice_line(s)
        self._canvas_heat.draw_idle()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()
        self._rebuild_slice_list()


# ── entry point called by the subprocess ─────────────────────────────────────

def run_analysis_window(csv_path: str):
    root = tk.Tk()
    app = AnalysisWindow(root, csv_path)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis_window.py <file.csv>")
        sys.exit(1)
    run_analysis_window(sys.argv[1])
