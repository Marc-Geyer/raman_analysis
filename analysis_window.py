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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

import theme as T


def _next_color(used: list[str]) -> str:
    for c in T.SLICE_COLORS:
        if c not in used:
            return c
    return T.SLICE_COLORS[len(used) % len(T.SLICE_COLORS)]


# ── CSV loader ────────────────────────────────────────────────────────────────
def filter_positive_times(
    times: np.ndarray,
    intensities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = times >= 0
    filtered_times = times[mask]
    if intensities.ndim == 2:
        filtered_intensities = intensities[:, mask]
    else:
        filtered_intensities = intensities
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
    """Represents one draggable selector line on the heatmap."""
    _id_counter = 0

    def __init__(self, kind: str, index: int, color: str, label: str = ""):
        Slice._id_counter += 1
        self.uid = Slice._id_counter
        self.kind = kind        # "vertical" | "horizontal"
        self.index = index
        self.color = color
        self.label = label or f"{'W' if kind=='vertical' else 'T'}{self.uid}"
        self.line_obj: Optional[Line2D] = None
        self.visible = True


# ── helpers: apply consistent axes styling ────────────────────────────────────

def _style_axes(ax, xlabel: str, ylabel: str, title: str,
                axes_facecolor: str = T.BG_AXES):
    """Apply the shared theme style to any matplotlib Axes."""
    ax.set_facecolor(axes_facecolor)
    ax.tick_params(colors=T.FG_TICK, labelsize=T.FONT_SIZE_SMALL)
    for sp in ax.spines.values():
        sp.set_edgecolor(T.COLOR_SPINE)
    ax.set_xlabel(xlabel, color=T.FG_AXIS_LABEL,
                  fontsize=T.FONT_SIZE_SMALL)
    ax.set_ylabel(ylabel, color=T.FG_AXIS_LABEL,
                  fontsize=T.FONT_SIZE_SMALL)
    ax.set_title(title, color=T.FG_PLOT_TITLE,
                 fontsize=T.FONT_SIZE_BODY)
    ax.grid(True, color=T.COLOR_GRID, linewidth=0.5, linestyle="--", alpha=0.7)


# ── Analysis Window ───────────────────────────────────────────────────────────

class AnalysisWindow:

    def __init__(self, root: tk.Tk, path: str):
        self.root = root
        self.path = path
        self.fname = os.path.basename(path)

        self.wavenumbers: np.ndarray = np.array([])
        self.times: np.ndarray = np.array([])
        self.intensity: np.ndarray = np.array([])

        self.slices: list[Slice] = []
        self._drag_slice: Optional[Slice] = None
        self._drag_start_xy: tuple = (0, 0)
        self._focused_slice: Optional[Slice] = None
        self._cmap = T.DEFAULT_CMAP
        self._heatmap_img = None

        self._build_ui()
        self._load_data()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root
        r.title(f"Raman Analysis – {self.fname}")
        r.geometry("1280x780")
        r.minsize(900, 600)
        r.configure(bg=T.BG_APP)

        # ── top toolbar ──
        toolbar = tk.Frame(r, bg=T.BG_TOOLBAR, pady=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            toolbar, text=f"  {self.fname}",
            bg=T.BG_TOOLBAR, fg=T.FG_TITLE,
            font=(T.FONT_MONO, T.FONT_SIZE_BODY, "bold"),
        ).pack(side=tk.LEFT)

        # colormap picker
        tk.Label(
            toolbar, text="  cmap:",
            bg=T.BG_TOOLBAR, fg=T.FG_SUBTLE,
            font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
        ).pack(side=tk.LEFT)

        self._cmap_var = tk.StringVar(value=self._cmap)
        cmap_menu = ttk.Combobox(
            toolbar, textvariable=self._cmap_var,
            values=T.AVAILABLE_CMAPS, width=10, state="readonly",
        )
        cmap_menu.pack(side=tk.LEFT, padx=(2, 12))
        cmap_menu.bind("<<ComboboxSelected>>",
                       lambda e: self._refresh_heatmap())

        # add / remove slice buttons
        btn_cfg = dict(font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
                       relief=tk.FLAT, padx=8, bd=0)

        tk.Button(
            toolbar, text="＋ Spectrum slice (V)",
            bg="#dbeafe", fg="#1e40af",
            activebackground="#bfdbfe", activeforeground="#1e3a8a",
            command=self._add_vertical_slice, **btn_cfg,
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            toolbar, text="＋ Time trace (H)",
            bg="#dcfce7", fg="#166534",
            activebackground="#bbf7d0", activeforeground="#14532d",
            command=self._add_horizontal_slice, **btn_cfg,
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            toolbar, text="✕ Remove selected",
            bg="#fee2e2", fg="#991b1b",
            activebackground="#fecaca", activeforeground="#7f1d1d",
            command=self._remove_focused_slice, **btn_cfg,
        ).pack(side=tk.LEFT, padx=2)

        # crosshair readout
        self._info_var = tk.StringVar(value="")
        tk.Label(
            toolbar, textvariable=self._info_var,
            bg=T.BG_TOOLBAR, fg=T.FG_SUBTLE,
            font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
        ).pack(side=tk.RIGHT, padx=12)

        # ── main pane: heatmap left, plots right ──
        paned = tk.PanedWindow(
            r, orient=tk.HORIZONTAL,
            bg=T.COLOR_SASH, sashwidth=6, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True)

        # left – heatmap figure
        left = tk.Frame(paned, bg=T.BG_APP)
        paned.add(left, minsize=400)

        self._fig_heat, self._ax_heat = plt.subplots(
            figsize=(6, 5), facecolor=T.BG_FIGURE)
        _style_axes(self._ax_heat,
                    xlabel="Time",
                    ylabel="Wavenumber (cm⁻¹)",
                    title="Raman Intensity Heatmap")

        self._canvas_heat = FigureCanvasTkAgg(self._fig_heat, master=left)
        self._canvas_heat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._fig_heat.canvas.mpl_connect("button_press_event",
                                          self._on_heat_press)
        self._fig_heat.canvas.mpl_connect("motion_notify_event",
                                          self._on_heat_motion)
        self._fig_heat.canvas.mpl_connect("button_release_event",
                                          self._on_heat_release)
        self._fig_heat.canvas.mpl_connect("scroll_event",
                                          self._on_heat_scroll)

        # right – two stacked plot panels
        right = tk.Frame(paned, bg=T.BG_APP)
        paned.add(right, minsize=340)

        right_paned = tk.PanedWindow(
            right, orient=tk.VERTICAL,
            bg=T.COLOR_SASH, sashwidth=5,
        )
        right_paned.pack(fill=tk.BOTH, expand=True)

        # spectrum panel (top-right)
        spec_frame = tk.Frame(right_paned, bg=T.BG_APP)
        right_paned.add(spec_frame, minsize=200)

        self._fig_spec, self._ax_spec = plt.subplots(
            figsize=(4, 3), facecolor=T.BG_FIGURE)
        _style_axes(self._ax_spec,
                    xlabel="Wavenumber (cm⁻¹)",
                    ylabel="Intensity",
                    title="Spectra (vertical slices)")

        self._canvas_spec = FigureCanvasTkAgg(self._fig_spec, master=spec_frame)
        self._canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # time trace panel (bottom-right)
        time_frame = tk.Frame(right_paned, bg=T.BG_APP)
        right_paned.add(time_frame, minsize=200)

        self._fig_time, self._ax_time = plt.subplots(
            figsize=(4, 3), facecolor=T.BG_FIGURE)
        _style_axes(self._ax_time,
                    xlabel="Time",
                    ylabel="Intensity",
                    title="Time traces (horizontal slices)")

        self._canvas_time = FigureCanvasTkAgg(self._fig_time, master=time_frame)
        self._canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── slice list panel (bottom) ──
        slice_panel = tk.Frame(r, bg=T.BG_SLICE_BAR, pady=4)
        slice_panel.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            slice_panel, text=" Active slices:",
            bg=T.BG_SLICE_BAR, fg=T.FG_SUBTLE,
            font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
        ).pack(side=tk.LEFT)

        self._slice_list_frame = tk.Frame(slice_panel, bg=T.BG_SLICE_BAR)
        self._slice_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # keyboard bindings
        r.bind("<Left>",   lambda e: self._nudge_focused(-1))
        r.bind("<Right>",  lambda e: self._nudge_focused(+1))
        r.bind("<Up>",     lambda e: self._nudge_focused(+1))
        r.bind("<Down>",   lambda e: self._nudge_focused(-1))
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
        self._add_vertical_slice()
        self._add_horizontal_slice()

    # ── heatmap ───────────────────────────────────────────────────────────────

    def _refresh_heatmap(self):
        if self.intensity.size == 0:
            return
        self._cmap = self._cmap_var.get()
        ax = self._ax_heat
        ax.cla()
        _style_axes(ax,
                    xlabel="Time",
                    ylabel="Wavenumber (cm⁻¹)",
                    title="Raman Intensity Heatmap",
                    axes_facecolor=T.BG_AXES)

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

        for s in self.slices:
            s.line_obj = None
            self._draw_slice_line(s)

        self._canvas_heat.draw_idle()

    def _draw_slice_line(self, s: Slice):
        focused = s is self._focused_slice
        lw = 2.2 if focused else 1.4
        ls = "-" if focused else "--"

        if s.kind == "vertical":
            val = self.times[s.index]
            if s.line_obj is None:
                s.line_obj = self._ax_heat.axvline(val, color=s.color,
                                                   lw=lw, ls=ls)
            else:
                s.line_obj.set_xdata([val, val])
        else:
            val = self.wavenumbers[s.index]
            if s.line_obj is None:
                s.line_obj = self._ax_heat.axhline(val, color=s.color,
                                                   lw=lw, ls=ls)
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
        self._rebuild_slice_list()
        self._set_focused(s)
        self._draw_slice_line(s)
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
        if event.xdata is None or event.ydata is None:
            return None
        ax = self._ax_heat
        best, best_dist = None, np.inf
        for s in self.slices:
            if s.kind == "vertical":
                data_val = self.times[s.index]
                disp  = ax.transData.transform((data_val, event.ydata))
                click = ax.transData.transform((event.xdata, event.ydata))
                dist  = abs(disp[0] - click[0])
            else:
                data_val = self.wavenumbers[s.index]
                disp  = ax.transData.transform((event.xdata, data_val))
                click = ax.transData.transform((event.xdata, event.ydata))
                dist  = abs(disp[1] - click[1])
            if dist < best_dist:
                best_dist = dist
                best = s
        return best if best_dist < 12 else None

    def _on_heat_press(self, event):
        if event.inaxes != self._ax_heat or event.xdata is None:
            return
        s = self._nearest_slice(event)
        if s is not None:
            self._drag_slice = s
            self._set_focused(s)

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

    def _on_heat_release(self, event):
        self._drag_slice = None
        self._rebuild_slice_list()

    def _on_heat_scroll(self, event):
        delta = 1 if event.button == "up" else -1
        self._nudge_focused(delta)

    # ── right-panel plots ─────────────────────────────────────────────────────

    def _refresh_spectrum_plot(self):
        ax = self._ax_spec
        ax.cla()
        _style_axes(ax,
                    xlabel="Wavenumber (cm⁻¹)",
                    ylabel="Intensity",
                    title="Spectra (vertical slices)")

        vslices = [s for s in self.slices if s.kind == "vertical"]
        for s in vslices:
            if not s.visible or self.intensity.size == 0:
                continue
            spectrum = self.intensity[:, s.index]
            ax.plot(self.wavenumbers, spectrum, color=s.color, lw=1.4,
                    label=f"{s.label}  t={self.times[s.index]:.3g}")

        if vslices:
            ax.legend(fontsize=T.FONT_SIZE_SMALL,
                      facecolor=T.BG_PANEL,
                      labelcolor=T.FG_LABEL,
                      edgecolor=T.COLOR_SPINE)
        self._canvas_spec.draw_idle()

    def _refresh_time_plot(self):
        ax = self._ax_time
        ax.cla()
        _style_axes(ax,
                    xlabel="Time",
                    ylabel="Intensity",
                    title="Time traces (horizontal slices)")

        hslices = [s for s in self.slices if s.kind == "horizontal"]
        for s in hslices:
            if not s.visible or self.intensity.size == 0:
                continue
            trace = self.intensity[s.index, :]
            ax.plot(self.times, trace, color=s.color, lw=1.4,
                    label=f"{s.label}  wn={self.wavenumbers[s.index]:.1f}")

        if hslices:
            ax.legend(fontsize=T.FONT_SIZE_SMALL,
                      facecolor=T.BG_PANEL,
                      labelcolor=T.FG_LABEL,
                      edgecolor=T.COLOR_SPINE)
        self._canvas_time.draw_idle()

    # ── slice list bar ────────────────────────────────────────────────────────

    def _rebuild_slice_list(self):
        for w in self._slice_list_frame.winfo_children():
            w.destroy()

        for s in self.slices:
            frame = tk.Frame(self._slice_list_frame,
                             bg=T.BG_SLICE_BAR, padx=3, pady=1)
            frame.pack(side=tk.LEFT)

            focused = s is self._focused_slice
            border_color = s.color if focused else T.COLOR_SPINE
            inner = tk.Frame(frame, bg=border_color, padx=1, pady=1)
            inner.pack()

            row = tk.Frame(inner, bg=T.BG_PANEL)
            row.pack()

            # color swatch → open color picker
            swatch = tk.Label(row, bg=s.color, width=2, cursor="hand2")
            swatch.pack(side=tk.LEFT)
            swatch.bind("<Button-1>", lambda e, sl=s: self._pick_color(sl))

            # label with value
            kind_icon = "│" if s.kind == "vertical" else "─"
            arr = self.times if s.kind == "vertical" else self.wavenumbers
            val = arr[s.index] if arr.size else 0
            lbl_text = f" {kind_icon}{s.label}={val:.3g} "
            lbl = tk.Label(row, text=lbl_text,
                           bg=T.BG_PANEL, fg=s.color,
                           font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
                           cursor="hand2")
            lbl.pack(side=tk.LEFT)
            lbl.bind("<Button-1>", lambda e, sl=s: self._set_focused(sl))

            # visibility toggle
            eye = "◉" if s.visible else "○"
            vis_btn = tk.Label(row, text=eye,
                               bg=T.BG_PANEL, fg=T.FG_SUBTLE,
                               font=(T.FONT_MONO, T.FONT_SIZE_SMALL),
                               cursor="hand2")
            vis_btn.pack(side=tk.LEFT)
            vis_btn.bind("<Button-1>",
                         lambda e, sl=s: self._toggle_visible(sl))

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