"""
Raman Data Viewer – Analysis Window
====================================
Displays a 2-D Raman dataset (intensity vs wavenumber × time) as an
interactive heatmap with draggable / arrow-key-tunable slice selectors
and rectangle region selectors that produce averaged spectra / time traces.

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
  │  – draggable rect regions   │  Time trace (right-bot)  │
  │                             │  intensity vs time       │
  └─────────────────────────────┴──────────────────────────┘
  Bottom toolbar: slices bar / regions bar, colormap picker, crosshair info

Region averaged plots
---------------------
  Vertical slices   → spectrum plot   (intensity vs wavenumber)
  Horizontal slices → time plot       (intensity vs time)
  Regions           → BOTH plots
      spectrum plot : mean intensity over the region's time range  vs wavenumber
      time plot     : mean intensity over the region's wn range    vs time
  Region lines are drawn with lw=2, ls="-." to distinguish from slice lines.
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import theme as T


def _next_color(used: list[str]) -> str:
    for c in T.SLICE_COLORS:
        if c not in used:
            return c
    return T.SLICE_COLORS[len(used) % len(T.SLICE_COLORS)]


# ── CSV loader ────────────────────────────────────────────────────────────────

def filter_positive_times(times, intensities):
    mask = times >= 0
    filtered_times = times[mask]
    filtered_intensities = intensities[:, mask] if intensities.ndim == 2 else intensities
    return filtered_times, filtered_intensities


def load_raman_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (wavenumbers [N], times [M], intensity [N×M])."""
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
    if intensities.ndim == 2 and intensities.shape[1] != len(times):
        intensities = intensities[:, : len(times)]

    times, intensities = filter_positive_times(times, intensities)
    return wavenumbers, times, intensities


# ── Slice descriptor ──────────────────────────────────────────────────────────

class Slice:
    """One draggable selector line on the heatmap."""
    _id_counter = 0

    def __init__(self, kind: str, index: int, color: str, label: str = ""):
        Slice._id_counter += 1
        self.uid = Slice._id_counter
        self.kind = kind        # "vertical" | "horizontal"
        self.index = index
        self.color = color
        self.label = label or f"{'W' if kind == 'vertical' else 'T'}{self.uid}"
        self.line_obj: Optional[Line2D] = None
        self.visible = True


# ── Region descriptor ─────────────────────────────────────────────────────────

class Region:
    """
    A rectangular selection on the heatmap defined by index ranges.

    ti0, ti1  – time-axis indices  (ti0 <= ti1)
    wi0, wi1  – wavenumber indices (wi0 <= wi1)

    Produces:
      averaged spectrum  : mean over times[ti0:ti1+1]  -> I vs wavenumber
      averaged time trace: mean over wavenumbers[wi0:wi1+1] -> I vs time
    """
    _id_counter = 0

    def __init__(self, ti0: int, ti1: int, wi0: int, wi1: int,
                 color: str, label: str = ""):
        Region._id_counter += 1
        self.uid = Region._id_counter
        self.ti0 = ti0
        self.ti1 = ti1
        self.wi0 = wi0
        self.wi1 = wi1
        self.color = color
        self.label = label or f"R{self.uid}"
        self.rect_obj: Optional[Rectangle] = None
        self.visible = True


# ── shared axes style helper ──────────────────────────────────────────────────

def _style_axes(ax, xlabel: str, ylabel: str, title: str,
                axes_facecolor: str = T.BG_AXES):
    ax.set_facecolor(axes_facecolor)
    ax.tick_params(colors=T.FG_TICK, labelsize=T.FONT_SIZE_SMALL)
    for sp in ax.spines.values():
        sp.set_edgecolor(T.COLOR_SPINE)
    ax.set_xlabel(xlabel, color=T.FG_AXIS_LABEL, fontsize=T.FONT_SIZE_SMALL)
    ax.set_ylabel(ylabel, color=T.FG_AXIS_LABEL, fontsize=T.FONT_SIZE_SMALL)
    ax.set_title(title, color=T.FG_PLOT_TITLE, fontsize=T.FONT_SIZE_BODY)
    ax.grid(True, color=T.COLOR_GRID, linewidth=0.5, linestyle="--", alpha=0.7)


# ── Analysis Window ───────────────────────────────────────────────────────────

class AnalysisWindow:

    # interaction modes
    MODE_SLICE  = "slice"   # default – drag existing slice lines
    MODE_REGION = "region"  # draw new rectangle with press+drag

    def __init__(self, root: tk.Tk, path: str):
        self.root = root
        self.path = path
        self.fname = os.path.basename(path)

        self.wavenumbers: np.ndarray = np.array([])
        self.times: np.ndarray = np.array([])
        self.intensity: np.ndarray = np.array([])

        # slices
        self.slices: list[Slice] = []
        self._drag_slice: Optional[Slice] = None
        self._focused_slice: Optional[Slice] = None
        self._slice_widgets: dict[int, dict] = {}

        # regions
        self.regions: list[Region] = []
        self._focused_region: Optional[Region] = None
        self._region_widgets: dict[int, dict] = {}

        # draw-region state
        self._mode = self.MODE_SLICE
        self._draw_start: Optional[tuple[float, float]] = None
        self._draw_rect_patch: Optional[Rectangle] = None

        self._cmap = T.DEFAULT_CMAP
        self._cbar = None
        self._heatmap_img = None

        self._build_ui()
        self._load_data()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root
        r.title(f"Raman Analysis – {self.fname}")
        r.geometry("1280x820")
        r.minsize(900, 640)
        r.configure(bg=T.BG_APP)

        # top toolbar
        toolbar = tk.Frame(r, bg=T.BG_TOOLBAR, pady=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(toolbar, text=f"  {self.fname}",
                 bg=T.BG_TOOLBAR, fg=T.FG_TITLE,
                 font=(T.FONT_MONO, T.FONT_SIZE_BODY, "bold")).pack(side=tk.LEFT)

        tk.Label(toolbar, text="  cmap:",
                 bg=T.BG_TOOLBAR, fg=T.FG_SUBTLE,
                 font=(T.FONT_MONO, T.FONT_SIZE_SMALL)).pack(side=tk.LEFT)

        self._cmap_var = tk.StringVar(value=self._cmap)
        cmap_menu = ttk.Combobox(toolbar, textvariable=self._cmap_var,
                                 values=T.AVAILABLE_CMAPS, width=10, state="readonly")
        cmap_menu.pack(side=tk.LEFT, padx=(2, 12))
        cmap_menu.bind("<<ComboboxSelected>>", lambda e: self._refresh_heatmap())

        btn_cfg = dict(font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
                       relief=tk.FLAT, padx=8, bd=0)

        tk.Button(toolbar, text="+ Spectrum slice (V)",
                  bg="#dbeafe", fg="#1e40af",
                  activebackground="#bfdbfe", activeforeground="#1e3a8a",
                  command=self._add_vertical_slice, **btn_cfg).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="+ Time trace (H)",
                  bg="#dcfce7", fg="#166534",
                  activebackground="#bbf7d0", activeforeground="#14532d",
                  command=self._add_horizontal_slice, **btn_cfg).pack(side=tk.LEFT, padx=2)

        tk.Label(toolbar, text="  |  ", bg=T.BG_TOOLBAR,
                 fg=T.COLOR_SPINE).pack(side=tk.LEFT)

        self._region_btn_text = tk.StringVar(value="[ ] Draw region")
        self._region_btn = tk.Button(
            toolbar, textvariable=self._region_btn_text,
            bg="#fef9c3", fg="#854d0e",
            activebackground="#fef08a", activeforeground="#713f12",
            command=self._toggle_region_mode, **btn_cfg)
        self._region_btn.pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="x Remove selected",
                  bg="#fee2e2", fg="#991b1b",
                  activebackground="#fecaca", activeforeground="#7f1d1d",
                  command=self._remove_focused, **btn_cfg).pack(side=tk.LEFT, padx=2)

        self._info_var = tk.StringVar(value="")
        tk.Label(toolbar, textvariable=self._info_var,
                 bg=T.BG_TOOLBAR, fg=T.FG_SUBTLE,
                 font=(T.FONT_MONO, T.FONT_SIZE_SMALL)).pack(side=tk.RIGHT, padx=12)

        # main paned: heatmap left / plots right
        paned = tk.PanedWindow(r, orient=tk.HORIZONTAL,
                               bg=T.COLOR_SASH, sashwidth=6, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(paned, bg=T.BG_APP)
        paned.add(left, minsize=400)

        self._fig_heat, self._ax_heat = plt.subplots(figsize=(6, 5),
                                                      facecolor=T.BG_FIGURE)
        _style_axes(self._ax_heat, "Time", "Wavenumber (cm-1)",
                    "Raman Intensity Heatmap")
        self._canvas_heat = FigureCanvasTkAgg(self._fig_heat, master=left)
        self._canvas_heat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._fig_heat.canvas.mpl_connect("button_press_event",   self._on_heat_press)
        self._fig_heat.canvas.mpl_connect("motion_notify_event",  self._on_heat_motion)
        self._fig_heat.canvas.mpl_connect("button_release_event", self._on_heat_release)
        self._fig_heat.canvas.mpl_connect("scroll_event",         self._on_heat_scroll)

        right = tk.Frame(paned, bg=T.BG_APP)
        paned.add(right, minsize=340)

        right_paned = tk.PanedWindow(right, orient=tk.VERTICAL,
                                     bg=T.COLOR_SASH, sashwidth=5)
        right_paned.pack(fill=tk.BOTH, expand=True)

        spec_frame = tk.Frame(right_paned, bg=T.BG_APP)
        right_paned.add(spec_frame, minsize=200)
        self._fig_spec, self._ax_spec = plt.subplots(figsize=(4, 3),
                                                      facecolor=T.BG_FIGURE)
        _style_axes(self._ax_spec, "Wavenumber (cm-1)", "Intensity",
                    "Spectra (V-slices / regions)")
        self._canvas_spec = FigureCanvasTkAgg(self._fig_spec, master=spec_frame)
        self._canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        time_frame = tk.Frame(right_paned, bg=T.BG_APP)
        right_paned.add(time_frame, minsize=200)
        self._fig_time, self._ax_time = plt.subplots(figsize=(4, 3),
                                                      facecolor=T.BG_FIGURE)
        _style_axes(self._ax_time, "Time", "Intensity",
                    "Time traces (H-slices / regions)")
        self._canvas_time = FigureCanvasTkAgg(self._fig_time, master=time_frame)
        self._canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._fig_spec.canvas.mpl_connect("motion_notify_event", self._on_spec_motion)
        self._fig_time.canvas.mpl_connect("motion_notify_event", self._on_time_motion)

        # bottom bar: slices row + regions row
        bottom = tk.Frame(r, bg=T.BG_SLICE_BAR)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        slice_row = tk.Frame(bottom, bg=T.BG_SLICE_BAR, pady=3)
        slice_row.pack(fill=tk.X)
        tk.Label(slice_row, text=" Slices: ",
                 bg=T.BG_SLICE_BAR, fg=T.FG_SUBTLE,
                 font=(T.FONT_MONO, T.FONT_SIZE_SMALL)).pack(side=tk.LEFT)
        self._slice_list_frame = tk.Frame(slice_row, bg=T.BG_SLICE_BAR)
        self._slice_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        _RRB = "#fefce8"
        region_row = tk.Frame(bottom, bg=_RRB, pady=3)
        region_row.pack(fill=tk.X)
        tk.Label(region_row, text=" Regions:",
                 bg=_RRB, fg=T.FG_SUBTLE,
                 font=(T.FONT_MONO, T.FONT_SIZE_SMALL)).pack(side=tk.LEFT)
        self._region_list_frame = tk.Frame(region_row, bg=_RRB)
        self._region_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._REGION_ROW_BG = _RRB

        r.bind("<Left>",   lambda e: self._nudge_focused(-1))
        r.bind("<Right>",  lambda e: self._nudge_focused(+1))
        r.bind("<Up>",     lambda e: self._nudge_focused(+1))
        r.bind("<Down>",   lambda e: self._nudge_focused(-1))
        r.bind("<Delete>", lambda e: self._remove_focused())
        r.bind("<Escape>", lambda e: self._escape_pressed())

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

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
        self._add_vertical_slice()
        self._add_horizontal_slice()

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmap rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_heatmap(self):
        if self.intensity.size == 0:
            return
        if self._cbar is not None:
            self._cbar.remove()
        self._cmap = self._cmap_var.get()
        ax = self._ax_heat
        fig = self._fig_heat
        ax.cla()
        _style_axes(ax, "Time (s)", "Wavenumber (cm-1)", "Raman Intensity Heatmap")
        t0, t1 = self.times[0], self.times[-1]
        w0, w1 = self.wavenumbers[0], self.wavenumbers[-1]
        self._heatmap_img = ax.imshow(
            self.intensity, aspect="auto",
            origin="lower" if w1 > w0 else "upper",
            extent=[t0, t1, w0, w1],
            cmap=self._cmap, interpolation="none",
        )

        self._cbar = fig.colorbar(self._heatmap_img, ax=ax, location="right")

        for s in self.slices:
            s.line_obj = None
            self._draw_slice_line(s)
        for rg in self.regions:
            rg.rect_obj = None
            self._draw_region_rect(rg)
        self._canvas_heat.draw_idle()

    def _draw_slice_line(self, s: Slice):
        focused = s is self._focused_slice
        lw = 2.2 if focused else 1.4
        ls = "-" if focused else "--"
        if s.kind == "vertical":
            val = self.times[s.index]
            if s.line_obj is None:
                s.line_obj = self._ax_heat.axvline(val, color=s.color, lw=lw, ls=ls)
            else:
                s.line_obj.set_xdata([val, val])
        else:
            val = self.wavenumbers[s.index]
            if s.line_obj is None:
                s.line_obj = self._ax_heat.axhline(val, color=s.color, lw=lw, ls=ls)
            else:
                s.line_obj.set_ydata([val, val])
        s.line_obj.set_color(s.color)
        s.line_obj.set_linewidth(lw)
        s.line_obj.set_linestyle(ls)
        s.line_obj.set_visible(s.visible)

    def _draw_region_rect(self, rg: Region):
        t_lo   = self.times[rg.ti0]
        t_hi   = self.times[rg.ti1]
        w_lo   = self.wavenumbers[min(rg.wi0, rg.wi1)]
        w_hi   = self.wavenumbers[max(rg.wi0, rg.wi1)]
        width  = t_hi - t_lo
        height = w_hi - w_lo
        focused = rg is self._focused_region
        lw = 2.2 if focused else 1.4
        alpha_fill = 0.30 if focused else 0.20
        if rg.rect_obj is None:
            rg.rect_obj = Rectangle(
                (t_lo, w_lo), width, height,
                linewidth=lw, edgecolor=rg.color,
                facecolor=rg.color, alpha=alpha_fill,
                linestyle="-", zorder=3,
            )
            self._ax_heat.add_patch(rg.rect_obj)
        else:
            rg.rect_obj.set_xy((t_lo, w_lo))
            rg.rect_obj.set_width(width)
            rg.rect_obj.set_height(height)
            rg.rect_obj.set_edgecolor(rg.color)
            rg.rect_obj.set_facecolor(rg.color)
            rg.rect_obj.set_linewidth(lw)
            rg.rect_obj.set_alpha(alpha_fill)
        rg.rect_obj.set_visible(rg.visible)

    # ─────────────────────────────────────────────────────────────────────────
    # Slice management
    # ─────────────────────────────────────────────────────────────────────────

    def _used_colors(self) -> list[str]:
        return [s.color for s in self.slices] + [rg.color for rg in self.regions]

    def _add_vertical_slice(self):
        idx = len(self.times) // 2 if self.times.size else 0
        s = Slice("vertical", idx, _next_color(self._used_colors()))
        self.slices.append(s)
        self._draw_slice_line(s)
        self._focused_slice = s
        self._focused_region = None
        self._rebuild_slice_list()
        self._refresh_spectrum_plot()

    def _add_horizontal_slice(self):
        idx = len(self.wavenumbers) // 2 if self.wavenumbers.size else 0
        s = Slice("horizontal", idx, _next_color(self._used_colors()))
        self.slices.append(s)
        self._draw_slice_line(s)
        self._focused_slice = s
        self._focused_region = None
        self._rebuild_slice_list()
        self._refresh_time_plot()

    def _set_focused_slice(self, s: Optional[Slice]):
        prev_sl = self._focused_slice
        prev_rg = self._focused_region
        self._focused_slice = s
        self._focused_region = None
        # deselect old region
        if prev_rg is not None:
            self._draw_region_rect(prev_rg)
            self._update_region_widgets(prev_rg)
        # update old and new slice border
        for sl in [prev_sl, s]:
            if sl is not None:
                self._draw_slice_line(sl)
                self._update_slice_widgets(sl)
        self._canvas_heat.draw_idle()

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
        self._update_slice_widgets(s)

    def _remove_focused(self):
        if self._focused_slice is not None:
            self._remove_slice(self._focused_slice)
        elif self._focused_region is not None:
            self._remove_region(self._focused_region)

    def _remove_slice(self, s: Slice):
        if s.line_obj is not None:
            try:
                s.line_obj.remove()
            except Exception:
                pass
        self.slices.remove(s)
        if self._focused_slice is s:
            self._focused_slice = None
        self._canvas_heat.draw_idle()
        self._rebuild_slice_list()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()

    def _escape_pressed(self):
        if self._mode == self.MODE_REGION and self._draw_start is not None:
            self._cancel_draw()
        else:
            if self._focused_slice is not None:
                self._set_focused_slice(None)
            elif self._focused_region is not None:
                self._set_focused_region(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Region management
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_region_mode(self):
        if self._mode == self.MODE_SLICE:
            self._mode = self.MODE_REGION
            self._region_btn.configure(bg="#fde047", fg="#713f12")
            self._region_btn_text.set("[ ] Drawing... (click+drag)")
            self._canvas_heat.get_tk_widget().configure(cursor="crosshair")
        else:
            self._cancel_draw()
            self._mode = self.MODE_SLICE
            self._region_btn.configure(bg="#fef9c3", fg="#854d0e")
            self._region_btn_text.set("[ ] Draw region")
            self._canvas_heat.get_tk_widget().configure(cursor="")

    def _cancel_draw(self):
        self._draw_start = None
        if self._draw_rect_patch is not None:
            try:
                self._draw_rect_patch.remove()
            except Exception:
                pass
            self._draw_rect_patch = None
        self._canvas_heat.draw_idle()

    def _commit_region(self, t0_d: float, t1_d: float,
                       w0_d: float, w1_d: float):
        ti0 = int(np.clip(np.searchsorted(self.times, min(t0_d, t1_d)),
                          0, len(self.times) - 1))
        ti1 = int(np.clip(np.searchsorted(self.times, max(t0_d, t1_d)),
                          0, len(self.times) - 1))
        wi0 = int(np.clip(np.searchsorted(self.wavenumbers, min(w0_d, w1_d)),
                          0, len(self.wavenumbers) - 1))
        wi1 = int(np.clip(np.searchsorted(self.wavenumbers, max(w0_d, w1_d)),
                          0, len(self.wavenumbers) - 1))
        if ti0 == ti1 or wi0 == wi1:
            return
        rg = Region(ti0, ti1, wi0, wi1, _next_color(self._used_colors()))
        self.regions.append(rg)
        self._draw_region_rect(rg)
        self._focused_region = rg
        self._focused_slice = None
        self._rebuild_region_list()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()
        self._canvas_heat.draw_idle()

    def _set_focused_region(self, rg: Optional[Region]):
        prev_rg = self._focused_region
        prev_sl = self._focused_slice
        self._focused_region = rg
        self._focused_slice = None
        if prev_sl is not None:
            self._draw_slice_line(prev_sl)
            self._update_slice_widgets(prev_sl)
        for r in [prev_rg, rg]:
            if r is not None:
                self._draw_region_rect(r)
                self._update_region_widgets(r)
        self._canvas_heat.draw_idle()

    def _remove_region(self, rg: Region):
        if rg.rect_obj is not None:
            try:
                rg.rect_obj.remove()
            except Exception:
                pass
        self.regions.remove(rg)
        if self._focused_region is rg:
            self._focused_region = None
        self._canvas_heat.draw_idle()
        self._rebuild_region_list()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmap mouse interaction
    # ─────────────────────────────────────────────────────────────────────────

    def _nearest_slice(self, event) -> Optional[Slice]:
        if event.xdata is None or event.ydata is None:
            return None
        ax = self._ax_heat
        best, best_dist = None, np.inf
        for s in self.slices:
            if s.kind == "vertical":
                dv = self.times[s.index]
                d  = ax.transData.transform((dv, event.ydata))
                c  = ax.transData.transform((event.xdata, event.ydata))
                dist = abs(d[0] - c[0])
            else:
                dv = self.wavenumbers[s.index]
                d  = ax.transData.transform((event.xdata, dv))
                c  = ax.transData.transform((event.xdata, event.ydata))
                dist = abs(d[1] - c[1])
            if dist < best_dist:
                best_dist = dist
                best = s
        return best if best_dist < 12 else None

    def _on_heat_press(self, event):
        if event.inaxes != self._ax_heat or event.xdata is None:
            return

        if self._mode == self.MODE_REGION:
            self._draw_start = (event.xdata, event.ydata)
            self._draw_rect_patch = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1.5, edgecolor="#854d0e",
                facecolor="#fde047", alpha=0.25,
                linestyle="--", zorder=4,
            )
            self._ax_heat.add_patch(self._draw_rect_patch)
            return

        # MODE_SLICE
        s = self._nearest_slice(event)
        if s is not None:
            self._drag_slice = s
            self._set_focused_slice(s)
            return

        # check region hit
        for rg in self.regions:
            t_lo = self.times[rg.ti0]
            t_hi = self.times[rg.ti1]
            w_lo = self.wavenumbers[min(rg.wi0, rg.wi1)]
            w_hi = self.wavenumbers[max(rg.wi0, rg.wi1)]
            if t_lo <= event.xdata <= t_hi and w_lo <= event.ydata <= w_hi:
                self._set_focused_region(rg)
                return

        # empty click – deselect all
        self._set_focused_slice(None)

    def _on_heat_motion(self, event):
        if event.inaxes != self._ax_heat or event.xdata is None:
            return

        ti = int(np.clip(np.searchsorted(self.times, event.xdata),
                         0, len(self.times) - 1))
        wi = int(np.clip(np.searchsorted(self.wavenumbers, event.ydata),
                         0, len(self.wavenumbers) - 1))
        val = self.intensity[wi, ti] if self.intensity.size else 0
        self._info_var.set(
            f"t={self.times[ti]:.3g}  wn={self.wavenumbers[wi]:.1f}  I={val:.4g}")

        # live rect preview
        if self._mode == self.MODE_REGION and self._draw_start is not None:
            x0, y0 = self._draw_start
            x1, y1 = event.xdata, event.ydata
            if self._draw_rect_patch is not None:
                self._draw_rect_patch.set_xy((min(x0, x1), min(y0, y1)))
                self._draw_rect_patch.set_width(abs(x1 - x0))
                self._draw_rect_patch.set_height(abs(y1 - y0))
                self._canvas_heat.draw_idle()
            return

        # slice drag
        if self._drag_slice is not None:
            s = self._drag_slice
            if s.kind == "vertical":
                idx = int(np.clip(np.searchsorted(self.times, event.xdata),
                                  0, len(self.times) - 1))
            else:
                idx = int(np.clip(np.searchsorted(self.wavenumbers, event.ydata),
                                  0, len(self.wavenumbers) - 1))
            if idx != s.index:
                s.index = idx
                self._draw_slice_line(s)
                self._canvas_heat.draw_idle()
                if s.kind == "vertical":
                    self._refresh_spectrum_plot()
                else:
                    self._refresh_time_plot()
                self._update_slice_widgets(s)

    def _on_heat_release(self, event):
        if self._mode == self.MODE_REGION and self._draw_start is not None:
            if event.xdata is not None and event.inaxes == self._ax_heat:
                x0, y0 = self._draw_start
                self._cancel_draw()
                self._commit_region(x0, event.xdata, y0, event.ydata)
            else:
                self._cancel_draw()
            # auto-return to slice mode
            self._mode = self.MODE_SLICE
            self._region_btn.configure(bg="#fef9c3", fg="#854d0e")
            self._region_btn_text.set("[ ] Draw region")
            self._canvas_heat.get_tk_widget().configure(cursor="")
            return
        self._drag_slice = None

    def _on_heat_scroll(self, event):
        delta = 1 if event.button == "up" else -1
        self._nudge_focused(delta)

    # ─────────────────────────────────────────────────────────────────────────
    # Crosshair info in side panels
    # ─────────────────────────────────────────────────────────────────────────

    def _on_spec_motion(self, event):
        if event.inaxes != self._ax_spec or event.xdata is None:
            return
        wi = int(np.clip(np.searchsorted(self.wavenumbers, event.xdata),
                         0, len(self.wavenumbers) - 1))
        if self.intensity.size:
            row = self.intensity[wi, :]
            self._info_var.set(
                f"wn={self.wavenumbers[wi]:.1f}  "
                f"I_min={np.nanmin(row):.4g}  I_max={np.nanmax(row):.4g}")
        else:
            self._info_var.set(f"wn={event.xdata:.1f}")

    def _on_time_motion(self, event):
        if event.inaxes != self._ax_time or event.xdata is None:
            return
        ti = int(np.clip(np.searchsorted(self.times, event.xdata),
                         0, len(self.times) - 1))
        if self.intensity.size:
            col = self.intensity[:, ti]
            self._info_var.set(
                f"t={self.times[ti]:.3g}  "
                f"I_min={np.nanmin(col):.4g}  I_max={np.nanmax(col):.4g}")
        else:
            self._info_var.set(f"t={event.xdata:.3g}")

    # ─────────────────────────────────────────────────────────────────────────
    # Right-panel plots  (slices + regions combined)
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_spectrum_plot(self):
        ax = self._ax_spec
        ax.cla()
        _style_axes(ax, "Wavenumber (cm-1)", "Intensity",
                    "Spectra (V-slices / regions)")
        has_legend = False

        for s in self.slices:
            if s.kind != "vertical" or not s.visible or self.intensity.size == 0:
                continue
            ax.plot(self.wavenumbers, self.intensity[:, s.index],
                    color=s.color, lw=1.4, ls="-",
                    label=f"{s.label}  t={self.times[s.index]:.3g}")
            has_legend = True

        for rg in self.regions:
            if not rg.visible or self.intensity.size == 0:
                continue
            mean_spec = np.nanmean(self.intensity[:, rg.ti0:rg.ti1 + 1], axis=1)
            t_lo = self.times[rg.ti0]
            t_hi = self.times[rg.ti1]
            ax.plot(self.wavenumbers, mean_spec,
                    color=rg.color, lw=2.0, ls="-.",
                    label=f"{rg.label} t:[{t_lo:.3g},{t_hi:.3g}]")
            has_legend = True

        if has_legend:
            ax.legend(fontsize=T.FONT_SIZE_SMALL, facecolor=T.BG_PANEL,
                      labelcolor=T.FG_LABEL, edgecolor=T.COLOR_SPINE)
        self._canvas_spec.draw_idle()

    def _refresh_time_plot(self):
        ax = self._ax_time
        ax.cla()
        _style_axes(ax, "Time (s)", "Intensity",
                    "Time traces (H-slices / regions)")
        has_legend = False

        for s in self.slices:
            if s.kind != "horizontal" or not s.visible or self.intensity.size == 0:
                continue
            ax.plot(self.times, self.intensity[s.index, :],
                    color=s.color, lw=1.4, ls="-",
                    label=f"{s.label}  wn={self.wavenumbers[s.index]:.1f}")
            has_legend = True

        for rg in self.regions:
            if not rg.visible or self.intensity.size == 0:
                continue
            mean_trace = np.nanmean(self.intensity[rg.wi0:rg.wi1 + 1, :], axis=0)
            w_lo = self.wavenumbers[rg.wi0]
            w_hi = self.wavenumbers[rg.wi1]
            ax.plot(self.times, mean_trace,
                    color=rg.color, lw=2.0, ls="-.",
                    label=f"{rg.label} wn:[{w_lo:.1f},{w_hi:.1f}]")
            has_legend = True

        if has_legend:
            ax.legend(fontsize=T.FONT_SIZE_SMALL, facecolor=T.BG_PANEL,
                      labelcolor=T.FG_LABEL, edgecolor=T.COLOR_SPINE)
        self._canvas_time.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Slice list bar  (stable widget cache)
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_slice_list(self):
        for w in self._slice_list_frame.winfo_children():
            w.destroy()
        self._slice_widgets.clear()
        for s in self.slices:
            self._slice_widgets[s.uid] = self._create_slice_widgets(s)

    def _create_slice_widgets(self, s: Slice) -> dict:
        focused = s is self._focused_slice
        border_color = s.color if focused else T.COLOR_SPINE
        frame = tk.Frame(self._slice_list_frame, bg=T.BG_SLICE_BAR, padx=3, pady=1)
        frame.pack(side=tk.LEFT)
        inner = tk.Frame(frame, bg=border_color, padx=1, pady=1)
        inner.pack()
        row = tk.Frame(inner, bg=T.BG_PANEL)
        row.pack()

        swatch = tk.Label(row, bg=s.color, width=2, cursor="hand2")
        swatch.pack(side=tk.LEFT)
        swatch.bind("<Button-1>", lambda e, sl=s: self._pick_slice_color(sl))

        arr = self.times if s.kind == "vertical" else self.wavenumbers
        val = arr[s.index] if arr.size else 0
        icon = "|" if s.kind == "vertical" else "-"
        lbl = tk.Label(row, text=f" {icon}{s.label}={val:.3g} ",
                       bg=T.BG_PANEL, fg=s.color,
                       font=(T.FONT_MONO, T.FONT_SIZE_SMALL), cursor="hand2")
        lbl.pack(side=tk.LEFT)
        lbl.bind("<Button-1>", lambda e, sl=s: self._set_focused_slice(sl))

        vis_btn = tk.Label(row, text="O" if s.visible else "o",
                           bg=T.BG_PANEL, fg=T.FG_SUBTLE,
                           font=(T.FONT_MONO, T.FONT_SIZE_SMALL), cursor="hand2")
        vis_btn.pack(side=tk.LEFT)
        vis_btn.bind("<Button-1>", lambda e, sl=s: self._toggle_slice_visible(sl))

        return dict(inner=inner, swatch=swatch, lbl=lbl, vis_btn=vis_btn)

    def _update_slice_widgets(self, s: Slice):
        widgets = self._slice_widgets.get(s.uid)
        if widgets is None:
            return
        focused = s is self._focused_slice
        widgets["inner"].configure(bg=s.color if focused else T.COLOR_SPINE)
        widgets["swatch"].configure(bg=s.color)
        arr = self.times if s.kind == "vertical" else self.wavenumbers
        val = arr[s.index] if arr.size else 0
        icon = "|" if s.kind == "vertical" else "-"
        widgets["lbl"].configure(text=f" {icon}{s.label}={val:.3g} ", fg=s.color)
        widgets["vis_btn"].configure(text="O" if s.visible else "o")

    def _pick_slice_color(self, s: Slice):
        color = colorchooser.askcolor(color=s.color, title=f"Color for {s.label}",
                                      parent=self.root)
        if color and color[1]:
            s.color = color[1]
            self._draw_slice_line(s)
            self._canvas_heat.draw_idle()
            self._refresh_spectrum_plot()
            self._refresh_time_plot()
            self._update_slice_widgets(s)

    def _toggle_slice_visible(self, s: Slice):
        s.visible = not s.visible
        self._draw_slice_line(s)
        self._canvas_heat.draw_idle()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()
        self._update_slice_widgets(s)

    # ─────────────────────────────────────────────────────────────────────────
    # Region list bar  (same stable-widget-cache pattern)
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_region_list(self):
        for w in self._region_list_frame.winfo_children():
            w.destroy()
        self._region_widgets.clear()
        for rg in self.regions:
            self._region_widgets[rg.uid] = self._create_region_widgets(rg)

    def _create_region_widgets(self, rg: Region) -> dict:
        focused = rg is self._focused_region
        border_color = rg.color if focused else T.COLOR_SPINE
        bg = self._REGION_ROW_BG

        frame = tk.Frame(self._region_list_frame, bg=bg, padx=3, pady=1)
        frame.pack(side=tk.LEFT)
        inner = tk.Frame(frame, bg=border_color, padx=1, pady=1)
        inner.pack()
        row = tk.Frame(inner, bg=T.BG_PANEL)
        row.pack()

        swatch = tk.Label(row, bg=rg.color, width=2, cursor="hand2")
        swatch.pack(side=tk.LEFT)
        swatch.bind("<Button-1>", lambda e, r=rg: self._pick_region_color(r))

        lbl = tk.Label(row, text=self._region_label_text(rg),
                       bg=T.BG_PANEL, fg=rg.color,
                       font=(T.FONT_MONO, T.FONT_SIZE_SMALL), cursor="hand2")
        lbl.pack(side=tk.LEFT)
        lbl.bind("<Button-1>", lambda e, r=rg: self._set_focused_region(r))

        vis_btn = tk.Label(row, text="O" if rg.visible else "o",
                           bg=T.BG_PANEL, fg=T.FG_SUBTLE,
                           font=(T.FONT_MONO, T.FONT_SIZE_SMALL), cursor="hand2")
        vis_btn.pack(side=tk.LEFT)
        vis_btn.bind("<Button-1>", lambda e, r=rg: self._toggle_region_visible(r))

        del_btn = tk.Label(row, text=" x",
                           bg=T.BG_PANEL, fg="#ef4444",
                           font=(T.FONT_MONO, T.FONT_SIZE_SMALL), cursor="hand2")
        del_btn.pack(side=tk.LEFT)
        del_btn.bind("<Button-1>", lambda e, r=rg: self._remove_region(r))

        return dict(inner=inner, swatch=swatch, lbl=lbl, vis_btn=vis_btn)

    def _region_label_text(self, rg: Region) -> str:
        if not self.times.size or not self.wavenumbers.size:
            return f" [{rg.label}] "
        t_lo = self.times[rg.ti0]
        t_hi = self.times[rg.ti1]
        w_lo = self.wavenumbers[min(rg.wi0, rg.wi1)]
        w_hi = self.wavenumbers[max(rg.wi0, rg.wi1)]
        return f" [{rg.label}] t:[{t_lo:.3g},{t_hi:.3g}] wn:[{w_lo:.0f},{w_hi:.0f}] "

    def _update_region_widgets(self, rg: Region):
        widgets = self._region_widgets.get(rg.uid)
        if widgets is None:
            return
        focused = rg is self._focused_region
        widgets["inner"].configure(bg=rg.color if focused else T.COLOR_SPINE)
        widgets["swatch"].configure(bg=rg.color)
        widgets["lbl"].configure(text=self._region_label_text(rg), fg=rg.color)
        widgets["vis_btn"].configure(text="O" if rg.visible else "o")

    def _pick_region_color(self, rg: Region):
        color = colorchooser.askcolor(color=rg.color, title=f"Color for {rg.label}",
                                      parent=self.root)
        if color and color[1]:
            rg.color = color[1]
            self._draw_region_rect(rg)
            self._canvas_heat.draw_idle()
            self._refresh_spectrum_plot()
            self._refresh_time_plot()
            self._update_region_widgets(rg)

    def _toggle_region_visible(self, rg: Region):
        rg.visible = not rg.visible
        self._draw_region_rect(rg)
        self._canvas_heat.draw_idle()
        self._refresh_spectrum_plot()
        self._refresh_time_plot()
        self._update_region_widgets(rg)


# ── entry point ───────────────────────────────────────────────────────────────

def run_analysis_window(csv_path: str):
    root = tk.Tk()
    app = AnalysisWindow(root, csv_path)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis_window.py <file.csv>")
        sys.exit(1)
    run_analysis_window(sys.argv[1])