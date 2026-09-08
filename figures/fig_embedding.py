#!/usr/bin/env python3
"""Standalone solution-embedding diagram for a single problem: GRPO vs a ROAM.

Takes one self-contained completions file (see example_completions/*.json: the K completions
of both arms for one problem, each tagged with its arm + correctness + text), embeds every
completion (Qwen3-Embedding-0.6B), projects all points jointly to 2D with metric 
MDS on cosine distance, and draws:

  - one filled convex hull per arm (the region of solution-space that arm covers),
  - each completion as an 'x': solid = correct, faded = incorrect,
  - a legend with each arm's  <#correct>/<K>  count.

GRPO is always green; the second arm is orange if its name contains "delay", else red.
Output is a clean PNG with no title/text panels.

Example:
  python fig_embedding.py --completions ../example_completions/Omni_99.json --out omni_99.png
"""
import argparse, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "roam"))
from train import _strip_think                       # same think-stripping as the diversity metric

GRPO_C = "#2e7d32"                                   # GRPO green
RED, ORANGE = "#c0392b", "#e8790c"                   # ROAM red / Delayed ROAM orange

# Instruct prompt for embedding model
REASON = ("Instruct: Represent the core solution strategy and reasoning approach of this "
          "text (the sequence of mathematical steps and technique used), not its tone, "
          "reasoning narration style, wording, notation, verbosity, or formatting. Two "
          "texts should be close only if they use the same strategy.\nQuery:")


def embed(texts, device, cap):
    import torch
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device); enc.eval()
    with torch.no_grad():
        E = enc.encode([_strip_think(t)[:cap] for t in texts], convert_to_numpy=True,
                       normalize_embeddings=True, prompt=REASON,
                       show_progress_bar=False, batch_size=8).astype(np.float32)
    return E


def mds2d(E):
    from sklearn.manifold import MDS
    D = 1.0 - (E @ E.T)
    np.fill_diagonal(D, 0.0); D = np.clip(D, 0, None)
    return MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto",
               random_state=0, n_init=4, max_iter=500).fit_transform(D)


def pair_div(E):
    m = len(E)
    if m < 2:
        return None
    S = E @ E.T
    return 1.0 - float((S.sum() - np.trace(S)) / (m * (m - 1)))


def hull(ax, pts, color, alpha=0.18):
    if len(pts) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        v = pts[ConvexHull(pts).vertices]
        ax.fill(v[:, 0], v[:, 1], facecolor=color, alpha=alpha, edgecolor=color,
                linewidth=1.0, zorder=1)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", required=True,
                    help="single-problem completions json (see example_completions/*.json)")
    ap.add_argument("--out", default="", help="output png (default: <input stem>_embedding.png)")
    ap.add_argument("--grpo_arm", default="GRPO", help="name of the baseline arm (drawn green)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--emb_cap", type=int, default=4000, help="char cap per completion before embedding")
    ap.add_argument("--title", default="", help="optional title (default: none, clean figure)")
    ap.add_argument("--legend_loc", default="best")
    args = ap.parse_args()

    data = json.load(open(args.completions))
    comps = data["completions"]
    arms = list(dict.fromkeys(c["arm"] for c in comps))          # preserve first-seen order
    if args.grpo_arm not in arms:
        raise SystemExit(f"grpo_arm '{args.grpo_arm}' not among arms {arms}")
    other_arm = next(a for a in arms if a != args.grpo_arm)
    other_c = ORANGE if "delay" in other_arm.lower() else RED

    # embed ALL completions of BOTH arms together -> one joint 2D projection
    texts = [c["text"] or "" for c in comps]
    XY = mds2d(embed(texts, args.device, args.emb_cap))

    def arm_xy_correct(arm):
        idx = [i for i, c in enumerate(comps) if c["arm"] == arm]
        xy = XY[idx]
        corr = np.array([bool(comps[i]["is_correct"]) for i in idx])
        return xy, corr

    gxy, gc = arm_xy_correct(args.grpo_arm)
    oxy, oc = arm_xy_correct(other_arm)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"]})
    hull(ax, oxy, other_c); hull(ax, gxy, GRPO_C)               # second arm underneath
    for xy, corr, col in ((gxy, gc, GRPO_C), (oxy, oc, other_c)):
        for (x, y), ok in zip(xy, corr):
            if ok:
                ax.scatter(x, y, marker="x", s=95, c=col, linewidths=2.2, zorder=5)
            else:
                ax.scatter(x, y, marker="x", s=48, c=col, linewidths=1.1, alpha=.35, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    leg = [Line2D([0], [0], marker="x", color=GRPO_C, lw=0, markeredgewidth=2.2,
                  label=f"{args.grpo_arm} ({int(gc.sum())}/{len(gc)} correct)"),
           Line2D([0], [0], marker="x", color=other_c, lw=0, markeredgewidth=2.2,
                  label=f"{other_arm} ({int(oc.sum())}/{len(oc)} correct)"),
           Line2D([0], [0], marker="x", color="0.35", lw=0, markeredgewidth=2.2, label="solid = correct"),
           Line2D([0], [0], marker="x", color="0.6", lw=0, markeredgewidth=1.1, alpha=.5,
                  label="faded = incorrect")]
    ax.legend(handles=leg, fontsize=10, loc=args.legend_loc, frameon=False)
    if args.title:
        ax.set_title(args.title, fontsize=11)

    out = pathlib.Path(args.out) if args.out else \
        pathlib.Path(args.completions).with_name(pathlib.Path(args.completions).stem + "_embedding.png")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    dg = pair_div(embed([c["text"] or "" for c in comps
                         if c["arm"] == args.grpo_arm and c["is_correct"]], args.device, args.emb_cap)) \
        if gc.sum() >= 2 else None
    do = pair_div(embed([c["text"] or "" for c in comps
                         if c["arm"] == other_arm and c["is_correct"]], args.device, args.emb_cap)) \
        if oc.sum() >= 2 else None
    ds = lambda d: f"{d:.3f}" if d is not None else "n/a"
    print(f"[fig_embedding] {data.get('problem_id','?')}: "
          f"{args.grpo_arm} {int(gc.sum())}/{len(gc)} (diversity d={ds(dg)}), "
          f"{other_arm} {int(oc.sum())}/{len(oc)} (d={ds(do)}) -> {out}")


if __name__ == "__main__":
    main()
