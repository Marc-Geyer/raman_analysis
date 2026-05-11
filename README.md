# Raman Viewer

Interactive Raman data viewer — built with Tkinter + Matplotlib.  
Each CSV file you open runs in its **own independent process** so windows never block each other.

## Requirements

```
pip install -r requirements.txt
```

## Usage

```bash
# 1. Start the launcher
python main.py

# 2. (optional) generate a test dataset
python generate_sample.py   # → sample_raman.csv
```

## CSV format

| cell [0,0]  | time₁  | time₂  | … |
|-------------|--------|--------|---|
| wavenumber₁ | I₁₁    | I₁₂    | … |
| wavenumber₂ | I₂₁    | I₂₂    | … |

Both comma (`,`) and semicolon (`;`) delimiters are accepted.

## Analysis window controls

### Heatmap
| Action | Effect |
|--------|--------|
| Click near a line | Select / focus that slice |
| Drag a line | Move slice |
| Scroll wheel | Nudge focused slice ±1 step |
| Arrow keys | Fine-tune focused slice ±1 step |
| `Delete` | Remove focused slice |
| `Esc` | Deselect |

### Toolbar
- **＋ Spectrum slice (V)** — add a vertical line → plots intensity vs wavenumber in top-right panel  
- **＋ Time trace (H)** — add a horizontal line → plots intensity vs time in bottom-right panel  
- **✕ Remove selected** — deletes the focused slice  
- **cmap** — change heatmap colormap  

### Slice bar (bottom strip)
- **Colored swatch** — click to open color picker  
- **Label** — click to focus that slice  
- **◉ / ○** — toggle slice visibility  

## File structure

```
raman_viewer/
├── main.py            # Launcher window (run this)
├── analysis_window.py # Analysis window (spawned per file)
├── generate_sample.py # Test CSV generator
└── README.md
```
