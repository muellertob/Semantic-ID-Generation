import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # non-interactive backend mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

def compute_semantic_alignment(embeddings: torch.Tensor, semids: torch.Tensor,
                                layer_idx: int, n_between_samples: int = 10_000) -> dict:
    """
    Within- vs between-cluster cosine similarity for a single layer.

    Args:
        embeddings : (N, D) – L2-normalised item embeddings (original input space).
        semids     : (N, L) – raw semids, NO collision token.
        layer_idx  : which layer to evaluate.
        n_between_samples: number of cross-cluster pairs to sample for between estimate.

    Returns dict with keys: within_mean, between_mean, separation_score.
    """
    emb = embeddings.float()
    codes = semids[:, layer_idx]
    unique_codes = torch.unique(codes)

    # within-cluster: mean cosine sim of each item to its cluster centroid
    within_sims = []
    for code in unique_codes:
        mask = codes == code
        if mask.sum() < 2:
            continue
        cluster = emb[mask]                                               # (k, D)
        centroid = F.normalize(cluster.mean(0, keepdim=True), p=2, dim=1) # (1, D)
        sims = (cluster * centroid).sum(dim=1)                            # (k,)
        within_sims.append(sims.mean().item())

    within_mean = float(np.mean(within_sims)) if within_sims else 0.0

    # between-cluster: sample random pairs from different clusters
    n = len(emb)
    # draw an even number of indices so both halves have equal length
    sample_size = min(n_between_samples * 2, n)
    sample_size = sample_size - (sample_size % 2)   # force even
    idx = torch.randperm(n)[:sample_size]
    half = sample_size // 2
    a_idx, b_idx = idx[:half], idx[half:half * 2]
    diff_mask = codes[a_idx] != codes[b_idx]
    if diff_mask.sum() > 0:
        a = emb[a_idx[diff_mask]]
        b = emb[b_idx[diff_mask]]
        between_mean = float((a * b).sum(dim=1).mean().item())
    else:
        between_mean = 0.0

    separation = within_mean / (abs(between_mean) + 1e-8)

    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "separation_score": separation,
    }


def compute_hierarchy_coherence(semids: torch.Tensor) -> dict:
    """
    Conditional entropy H(L_{k+1} | L_k) for each consecutive layer pair.

    Low value → knowing the parent code is informative about the child code
    (good hierarchical structure).

    Returns dict: {'H(L2|L1)': float, 'H(L3|L2)': float, ...}
    """
    n_layers = semids.shape[1]
    n = len(semids)
    results = {}
    for k in range(n_layers - 1):
        parents = semids[:, k]
        children = semids[:, k + 1]
        unique_parents = torch.unique(parents)
        weighted_H = 0.0
        for p in unique_parents:
            mask = parents == p
            n_p = int(mask.sum())
            child_subset = children[mask]
            _, child_counts = torch.unique(child_subset, return_counts=True)
            probs = child_counts.float() / child_counts.sum()
            H = float(-(probs * probs.log()).sum().item())
            weighted_H += (n_p / n) * H
        results[f"H(L{k+2}|L{k+1})"] = weighted_H
    return results


def compute_utilisation(semids: torch.Tensor, codebook_size: int) -> dict:
    """
    Per-layer codebook utilisation: coverage, Shannon entropy, Gini coefficient.

    Returns dict with key 'per_layer': list of dicts, one per layer.
    """
    n_layers = semids.shape[1]
    per_layer = []
    for i in range(n_layers):
        codes = semids[:, i].numpy()
        counts = np.bincount(codes, minlength=codebook_size).astype(float)
        used = int((counts > 0).sum())
        coverage = used / codebook_size

        # Shannon entropy (nats, then convert to ratio of max)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        H = float(-np.sum(probs_nz * np.log(probs_nz)))
        H_max = float(np.log(codebook_size))
        entropy_ratio = H / H_max if H_max > 0 else 0.0

        # Gini coefficient (standard formula on sorted counts)
        sc = np.sort(counts)
        n = len(sc)
        gini = float((2 * np.sum((np.arange(1, n + 1) * sc)) /
                      (n * sc.sum()) - (n + 1) / n) if sc.sum() > 0 else 0.0)

        per_layer.append({
            "layer": i,
            "codes_used": used,
            "coverage": coverage,
            "shannon_entropy": H,
            "entropy_ratio": entropy_ratio,
            "gini": gini,
        })
    return {"per_layer": per_layer}


def compute_collision_stats(semids: torch.Tensor) -> dict:
    """
    Collision statistics from raw semids (before adding the collision token).

    Returns dict with collision_rate, n_collisions, max/mean/median/p90/p99 depth,
    and depth_distribution (np.ndarray, one entry per item).
    """
    n = len(semids)
    _, inverse = torch.unique(semids, sorted=True, return_inverse=True, dim=0)
    perm = torch.argsort(inverse)
    inv_sorted = inverse[perm]
    _, group_counts = torch.unique_consecutive(inv_sorted, return_counts=True)

    # assign sequential depths within each group: first occurrence = 0, next = 1, etc.
    group_starts = torch.cat([torch.zeros(1, dtype=torch.long), group_counts.cumsum(0)[:-1]])
    expanded_starts = group_starts.repeat_interleave(group_counts)
    sorted_depths = torch.arange(n, dtype=torch.long) - expanded_starts

    # restore original item order
    depth_per_item = torch.zeros(n, dtype=torch.long)
    depth_per_item[perm] = sorted_depths

    d = depth_per_item.numpy()
    colliding = d > 0
    n_collisions = int(colliding.sum())

    stats: dict = {
        "collision_rate": float(colliding.mean()),
        "n_collisions": n_collisions,
        "max_depth": int(d.max()),
        "depth_distribution": d,
    }
    if n_collisions > 0:
        d_pos = d[colliding]
        stats.update({
            "mean_depth": float(d_pos.mean()),
            "median_depth": float(np.median(d_pos)),
            "p90_depth": float(np.percentile(d_pos, 90)),
            "p99_depth": float(np.percentile(d_pos, 99)),
        })
    else:
        stats.update({"mean_depth": 0.0, "median_depth": 0.0,
                      "p90_depth": 0.0, "p99_depth": 0.0})
    return stats


def _plot_semantic_alignment(ax, alignment_per_layer: list) -> None:
    """Grouped bar chart: within vs between cosine sim per layer, annotated with sep score."""
    n_layers = len(alignment_per_layer)
    x = np.arange(n_layers)
    w = 0.30

    within = [s["within_mean"] for s in alignment_per_layer]
    between = [s["between_mean"] for s in alignment_per_layer]
    sep = [s["separation_score"] for s in alignment_per_layer]

    bars_w = ax.bar(x - w / 2, within, w, label="Within-cluster", color="#1a9e8f", zorder=3)
    bars_b = ax.bar(x + w / 2, between, w, label="Between-cluster", color="#d95f02", zorder=3)

    # Annotate separation score above the within bar
    y_top = max(max(within), max(between))
    for xi, s_val in zip(x, sep):
        ax.text(xi, y_top * 1.02, f"sep={s_val:.2f}",
                ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}" for i in range(n_layers)])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Semantic Alignment per Layer")
    ax.legend(fontsize="small")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_ylim(min(0, min(between) * 1.3), y_top * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_utilisation(ax, util_per_layer: list) -> None:
    """
    Three-line plot: coverage, entropy ratio, and (1 - Gini) per layer.
    We use 1-Gini so all three metrics go up = better.
    """
    n_layers = len(util_per_layer)
    x = np.arange(1, n_layers + 1)

    coverage = [u["coverage"] for u in util_per_layer]
    entropy_r = [u["entropy_ratio"] for u in util_per_layer]
    uniformity = [1 - u["gini"] for u in util_per_layer]   # higher = more uniform

    ax.plot(x, coverage, "o-", color="#1a9e8f", label="Coverage %", linewidth=2)
    ax.plot(x, entropy_r, "s-", color="#5b2d8e", label="Entropy ratio", linewidth=2)
    ax.plot(x, uniformity, "^-", color="#d95f02", label="Uniformity (1−Gini)", linewidth=2)

    for xi, cov, ent, uni in zip(x, coverage, entropy_r, uniformity):
        ax.annotate(f"{cov:.0%}", (xi, cov), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7, color="#1a9e8f")
        ax.annotate(f"{ent:.0%}", (xi, ent), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#5b2d8e")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}" for i in range(n_layers)])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Ratio (0–1, higher = better)")
    ax.set_title("Codebook Utilisation per Layer")
    ax.legend(fontsize="small")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_collision_hist(ax, collision_stats: dict) -> None:
    """Histogram of collision depths (items with depth > 0 only)."""
    d = collision_stats["depth_distribution"]
    d_pos = d[d > 0]

    if len(d_pos) == 0:
        ax.text(0.5, 0.5, "No collisions!", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title("Collision Depth Distribution")
        return

    max_d = int(d_pos.max())
    bins = min(max_d + 1, 50)
    ax.hist(d_pos, bins=bins, color="#c0392b", edgecolor="white", linewidth=0.4, alpha=0.85)

    # Vertical lines for percentiles
    for pct, val, ls in [(50, collision_stats["median_depth"], "--"),
                          (90, collision_stats["p90_depth"], ":"),
                          (99, collision_stats["p99_depth"], "-.")]:
        ax.axvline(val, color="#222", linestyle=ls, linewidth=1.2,
                   label=f"p{pct}={val:.0f}")

    rate = collision_stats["collision_rate"]
    n_col = collision_stats["n_collisions"]
    ax.set_xlabel("Collision depth (0 = unique)")
    ax.set_ylabel("Number of items")
    ax.set_title(f"Collision Depth Distribution  [{rate:.1%} collision rate, n={n_col}]")
    ax.legend(fontsize="small")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_hierarchy_coherence(ax, coherence: dict) -> None:
    """Bar chart of conditional entropy H(L_{k+1} | L_k)."""
    labels = list(coherence.keys())
    values = [coherence[k] for k in labels]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color="#5b2d8e", alpha=0.85, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Conditional entropy (nats)")
    ax.set_title("Hierarchy Coherence  [↓ better]")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_evaluation_plots(all_stats: dict, save_dir: str) -> None:
    """
    Produce a single 2×2 figure with all four evaluation plots and save it.

    Args:
        all_stats: dict returned by evaluate_semids()
        save_dir : directory to save the PNG into (created if needed)
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "sid_eval_report.png")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax_align = fig.add_subplot(gs[0, 0])
    ax_util  = fig.add_subplot(gs[0, 1])
    ax_hist  = fig.add_subplot(gs[1, 0])
    ax_hier  = fig.add_subplot(gs[1, 1])

    _plot_semantic_alignment(ax_align, all_stats["alignment"])
    _plot_utilisation(ax_util, all_stats["utilisation"]["per_layer"])
    _plot_collision_hist(ax_hist, all_stats["collision"])
    _plot_hierarchy_coherence(ax_hier, all_stats["coherence"])

    fig.suptitle("Semantic ID Evaluation Report", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    logger.info(f"Evaluation plot saved to: {out_path}")


def print_evaluation_report(all_stats: dict, config) -> None:
    """Print a structured text report to stdout."""
    n_layers = len(all_stats["utilisation"]["per_layer"])
    cb_size = getattr(getattr(config, "model", None), "codebook_clusters", "?")

    lines = [
        "",
        "=" * 60,
        " SEMANTIC ID EVALUATION REPORT",
        "=" * 60,
        f"  Codebook: {n_layers} layers × {cb_size} codes",
        "",
        "── Semantic Alignment (cosine sim, original embedding space) ──",
    ]
    for i, s in enumerate(all_stats["alignment"]):
        lines.append(
            f"  L{i+1}: within={s['within_mean']:.4f}  "
            f"between={s['between_mean']:.4f}  "
            f"separation={s['separation_score']:.3f}"
        )

    lines += ["", "── Hierarchy Coherence (H lower = better structure) ──"]
    for key, val in all_stats["coherence"].items():
        lines.append(f"  {key}: {val:.4f} nats")

    lines += ["", "── Codebook Utilisation ──"]
    for u in all_stats["utilisation"]["per_layer"]:
        lines.append(
            f"  L{u['layer']+1}: {u['codes_used']}/{cb_size} codes used "
            f"({u['coverage']:.1%})  "
            f"entropy={u['entropy_ratio']:.1%} of max  "
            f"Gini={u['gini']:.3f}"
        )

    c = all_stats["collision"]
    lines += [
        "",
        "── Collision Stats ──",
        f"  Collision rate : {c['collision_rate']:.1%}  ({c['n_collisions']} items)",
        f"  Max depth      : {c['max_depth']}",
        f"  Mean / Median  : {c['mean_depth']:.2f} / {c['median_depth']:.1f}",
        f"  p90 / p99      : {c['p90_depth']:.1f} / {c['p99_depth']:.1f}",
        "=" * 60,
        "",
    ]
    print("\n".join(lines))


def evaluate_semids(embeddings: torch.Tensor, raw_semids: torch.Tensor,
                    config, plot_dir: str | None = None) -> dict:
    """
    Run all evaluation metrics and (optionally) save plots.

    Args:
        embeddings  : (N, D) L2-normalised item embeddings from the data loader.
        raw_semids  : (N, L) tensor of semantic IDs BEFORE collision resolution.
        config      : OmegaConf config object.
        plot_dir    : directory to save plots; None → no plots saved.

    Returns:
        dict with keys: alignment, coherence, utilisation, collision.
    """
    cb_size = getattr(getattr(config, "model", None), "codebook_clusters", 128)
    n_layers = raw_semids.shape[1]

    logger.info("Running semantic ID evaluation…")

    alignment = [
        compute_semantic_alignment(embeddings, raw_semids, i)
        for i in range(n_layers)
    ]
    coherence = compute_hierarchy_coherence(raw_semids)
    utilisation = compute_utilisation(raw_semids, cb_size)
    collision = compute_collision_stats(raw_semids)

    all_stats = {
        "alignment": alignment,
        "coherence": coherence,
        "utilisation": utilisation,
        "collision": collision,
    }

    print_evaluation_report(all_stats, config)

    if plot_dir is not None:
        save_evaluation_plots(all_stats, plot_dir)

    return all_stats
