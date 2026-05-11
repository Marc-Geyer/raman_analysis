"""
generate_sample.py – creates a synthetic Raman CSV for testing.
Usage:  python generate_sample.py
Output: sample_raman.csv
"""
import numpy as np
import csv

N_WN = 200   # number of wavenumber points
N_T  = 80    # number of time points

wavenumbers = np.linspace(200, 3200, N_WN)
times       = np.linspace(0, 60, N_T)        # 0 – 60 s

# Simulate a slowly evolving Raman spectrum
np.random.seed(42)
peak_positions = [520, 1350, 2700]          # cm-1
peak_widths    = [15, 25, 30]
peak_base      = [1.0, 0.6, 0.4]

Z = np.zeros((N_WN, N_T))
for i, t in enumerate(times):
    # baseline drift
    bg = 0.05 * np.exp(-wavenumbers / 3000)
    spectrum = bg.copy()
    for pos, wid, amp in zip(peak_positions, peak_widths, peak_base):
        # amplitude changes with time
        a = amp * (1 + 0.5 * np.sin(2 * np.pi * t / 30))
        spectrum += a * np.exp(-((wavenumbers - pos) ** 2) / (2 * wid ** 2))
    spectrum += np.random.normal(0, 0.01, N_WN)
    Z[:, i] = spectrum

with open("sample_raman.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    header = ["wavenumber"] + [f"{t:.2f}" for t in times]
    writer.writerow(header)
    for wi, wn in enumerate(wavenumbers):
        row = [f"{wn:.2f}"] + [f"{Z[wi, ti]:.6f}" for ti in range(N_T)]
        writer.writerow(row)

print("Written: sample_raman.csv")
