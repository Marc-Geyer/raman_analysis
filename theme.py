"""
theme.py – Centralized visual constants for the Raman Viewer
============================================================
Edit values here to restyle the entire application at once.
"""

# ── Window / chrome ───────────────────────────────────────────────────────────
BG_APP          = "#f0f2f5"   # outer window background
BG_TOOLBAR      = "#e4e8ef"   # top toolbar strip
BG_PANEL        = "#ffffff"   # inner plot panel frames
BG_SLICE_BAR    = "#dde2ea"   # bottom slice-list strip
BG_LISTBOX      = "#f8f9fb"   # launcher file-list box

# ── Plot backgrounds ──────────────────────────────────────────────────────────
BG_FIGURE       = "#f7f8fa"   # matplotlib figure facecolor
BG_AXES         = "#ffffff"   # matplotlib axes facecolor

# ── Text ──────────────────────────────────────────────────────────────────────
FG_TITLE        = "#1a73e8"   # accent titles / labels (blue)
FG_LABEL        = "#333d4b"   # normal UI labels
FG_SUBTLE       = "#8a94a6"   # secondary / hint text
FG_AXIS_LABEL   = "#4a5568"   # matplotlib axis label color
FG_TICK         = "#6b7280"   # matplotlib tick label color
FG_PLOT_TITLE   = "#2d3748"   # matplotlib axes title color

# ── Plot lines ────────────────────────────────────────────────────────────────
LINE_WIDTH      = 1.0
LINE_STYLE      = "-"

# ── Borders / spines / grid ───────────────────────────────────────────────────
COLOR_SPINE     = "#cbd5e0"   # matplotlib spine edge color
COLOR_GRID      = "#e2e8f0"   # matplotlib grid line color (if enabled)

# ── Accent & interactive ──────────────────────────────────────────────────────
ACCENT          = "#1a73e8"   # primary accent (buttons, highlights)
ACCENT_HOVER    = "#1558b0"
ACCENT_PRESSED  = "#0d3d7a"
FG_ON_ACCENT    = "#ffffff"   # text drawn on top of ACCENT background

# ── Launcher listbox ──────────────────────────────────────────────────────────
LB_FG           = "#2c5282"   # listbox item foreground
LB_SELECT_BG    = "#bee3f8"   # listbox selected-row background
LB_SELECT_FG    = "#1a365d"   # listbox selected-row foreground

# ── Status bar ────────────────────────────────────────────────────────────────
BG_STATUS       = "#dde2ea"
FG_STATUS       = "#6b7280"

# ── Sash (pane dividers) ──────────────────────────────────────────────────────
COLOR_SASH      = "#c8d0dc"

# ── Slice colors (cycling palette for selectors and regions) ──────────────────────────────
SLICE_COLORS = [
    "#1a73e8",  # blue
    "#e53935",  # red
    "#2e7d32",  # green
    "#f57c00",  # orange
    "#6a1b9a",  # purple
    "#00838f",  # teal
    "#c62828",  # dark red
    "#558b2f",  # olive green
    "#1565c0",  # dark blue
    "#ad1457",  # pink
]

# ── Typography ────────────────────────────────────────────────────────────────
FONT_MONO       = "Courier New"
FONT_SIZE_TITLE = 11
FONT_SIZE_BODY  = 9
FONT_SIZE_SMALL = 8

# ── Default colormap ──────────────────────────────────────────────────────────
DEFAULT_CMAP    = "turbo"
AVAILABLE_CMAPS = ["viridis", "plasma", "inferno", "magma", "turbo",
                   "hot", "gray", "RdYlBu_r", "seismic", "cividis"]
