#!/usr/bin/env python3
"""Training-curve figures from consolidated logs (built by build_curve_logs.py).
Main figure: pass@1, pass@16, dispersion (3x2). Appendix: pass@4, pass@8 (2x2).
Late-ROAM is a BRANCH off the GRPO curve at its warm-start step.

Usage:
  python curves_r1.py --data_dir results/curves
"""
import argparse, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

HOLD, OMNI = "MATH_L5_eval_n400", "OmniMATH_eval_n400"
XMAX = 7000
VAN_C, ROAM_C, LR5_C = "#2e7d32", "#c0392b", "#e8790c"
COLS = [(HOLD, "MATH-L5 test"), (OMNI, "Omni-MATH-L5/6")]
BRANCH_STEP = 5000


def ser(b, m):
    xs = sorted(int(k) for k in b if k.isdigit() and int(k) <= XMAX)
    return xs, [b[str(x)][m] for x in xs]


def pad_ylim(ax, frac):
    lo, hi = ax.get_ylim(); r = hi - lo
    ax.set_ylim(lo - frac * r, hi + frac * r)


def plot(data_dir, metrics, out, title, height):
    van = json.load(open(data_dir / "grpo.json"))["benchmarks"]
    roam = json.load(open(data_dir / "roam_mc8.json"))["benchmarks"]
    lr = json.load(open(data_dir / "delayed_roam.json"))["branches"]

    fig, axes = plt.subplots(len(metrics), len(COLS), figsize=(11.5, height),
                             sharex=True, squeeze=False)
    for ci, (bench, ctitle) in enumerate(COLS):
        for ri, m in enumerate(metrics):
            ax = axes[ri][ci]
            vx, vy = ser(van[bench], m); rx, ry = ser(roam[bench], m)
            ax.plot(vx, vy, "-o", color=VAN_C, ms=4, lw=1.8, label="GRPO")
            ax.plot(rx, ry, "-o", color=ROAM_C, ms=4, lw=1.8, label="ROAM-GRPO")
            b = lr["g5000"][bench]
            bx = sorted(int(k) for k in b if k.isdigit() and int(k) <= XMAX)
            by = [b[str(x)][m] for x in bx]
            v_at = van[bench].get(str(BRANCH_STEP), {}).get(m)
            X = ([BRANCH_STEP] + bx) if v_at is not None else bx
            Y = ([v_at] + by) if v_at is not None else by
            ax.plot(X, Y, ":o", color=LR5_C, ms=4, lw=1.8, alpha=.95,
                    label="Delayed ROAM-GRPO")
            ax.axvline(BRANCH_STEP, color="0.75", lw=1, ls=":", zorder=0)
            ax.grid(alpha=.22)
            pad_ylim(ax, 0.24 if m != "dispersion" else 0.10)
            if ci == 0:
                ax.set_ylabel(m, fontsize=14)
            if ri == 0:
                ax.set_title(ctitle, fontsize=15)
        axes[-1][ci].set_xlabel("global training step", fontsize=13)
        for ax in axes[:, ci]:
            ax.set_xlim(right=XMAX + 150)
    axes[0][0].legend(fontsize=11, loc="upper left", framealpha=.9)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.suptitle(title, fontsize=17, y=0.988)
    fig.savefig(out, dpi=150); print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="results/curves",
                    help="Directory with consolidated JSONs from build_curve_logs.py")
    args = ap.parse_args()
    d = pathlib.Path(args.data_dir)
    plot(d, ["pass@1", "pass@16", "dispersion"], d / "curves_r1.png",
         "pass@k Accuracy and Solution Dispersion over Training", 8.2)
    plot(d, ["pass@4", "pass@8"], d / "curves_r1_appendix.png",
         "pass@4 and pass@8 Over Training", 6.0)


if __name__ == "__main__":
    main()
