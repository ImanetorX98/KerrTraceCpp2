#!/usr/bin/env python3
"""Plot measured boundary coordinates, without altering rendered frames.

Optional dependency: matplotlib. Input is a CSV emitted by metal_contour_test.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.matches.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("No contour matches to plot")
    points = {
        side: [(int(row["x"]), int(row["y"])) for row in rows if row["side"] == side]
        for side in ("reference", "candidate")
    }
    worst = max(rows, key=lambda row: int(row["distance_px"]))
    wx, wy = int(worst["x"]), int(worst["y"])
    nx, ny = int(worst["nearest_x"]), int(worst["nearest_y"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax in axes:
        for side, label, color, marker in (
            ("reference", "CPU, tolleranza 1e-10", "#137cbd", "s"),
            ("candidate", "Metal", "#e07018", "+"),
        ):
            xy = points[side]
            ax.scatter([p[0] for p in xy], [p[1] for p in xy],
                       s=13, color=color, marker=marker, label=label, alpha=.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x [pixel del frame]")
        ax.set_ylabel("y [pixel del frame]")
        ax.invert_yaxis()
        ax.grid(alpha=.15)
    axes[0].set_title("Contorni nella regione misurata")
    axes[0].legend(loc="lower center", fontsize=9)
    axes[0].scatter([wx], [wy], s=150, facecolors="none", edgecolors="#b22222")
    axes[1].set_xlim(min(wx, nx)-8, max(wx, nx)+8)
    axes[1].set_ylim(max(wy, ny)+8, min(wy, ny)-8)
    axes[1].plot([wx, nx], [wy, ny], color="#b22222", linewidth=2)
    axes[1].set_title(f"Scarto massimo: {worst['distance_px']} pixel (Chebyshev)")
    fig.suptitle("Confronto CPU / Metal — coordinate dei bordi nei frame originali", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
