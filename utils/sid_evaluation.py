import logging
import numpy as np
import torch

def compute_utilisation(semids: torch.Tensor, codebook_size: int) -> dict:
    """
    Per-layer codebook utilisation: coverage and perplexity.

    Perplexity = exp(-∑_k p_k log p_k), range [1, codebook_size].
    Inspired by the codebook utilisation diagnostic in:
        Zheng & Vedaldi, "Online Clustered Codebook" (CVQ-VAE), ICCV 2023, Sec. 4.1.
    The per-layer breakdown and perplexity_ratio are adaptations for RQ-VAE.

    Returns dict with key 'per_layer': list of dicts, one per layer.
    """
    n_layers = semids.shape[1]
    per_layer = []
    for i in range(n_layers):
        codes = semids[:, i].numpy()
        counts = np.bincount(codes, minlength=codebook_size).astype(float)
        used = int((counts > 0).sum())
        coverage = used / codebook_size

        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        H = float(-np.sum(probs_nz * np.log(probs_nz)))
        perplexity = float(np.exp(H))
        perplexity_ratio = perplexity / codebook_size

        per_layer.append({
            "layer": i,
            "codes_used": used,
            "coverage": coverage,
            "perplexity": perplexity,
            "max_perplexity": codebook_size,
            "perplexity_ratio": perplexity_ratio,
        })
    return {"per_layer": per_layer}


def compute_collision_stats(semids: torch.Tensor) -> dict:
    """
    Collision statistics for raw semids (before adding the collision token).

    The depth distribution and percentile breakdown are adaptations not
    present in the original paper.

    Returns dict with collision_rate, n_collisions, max/mean/median/p90/p99
    depth, and depth_distribution (np.ndarray, one entry per item).
    """
    n = len(semids)
    _, inverse = torch.unique(semids, sorted=True, return_inverse=True, dim=0)
    perm = torch.argsort(inverse)
    inv_sorted = inverse[perm]
    _, group_counts = torch.unique_consecutive(inv_sorted, return_counts=True)

    group_starts = torch.cat([torch.zeros(1, dtype=torch.long), group_counts.cumsum(0)[:-1]])
    expanded_starts = group_starts.repeat_interleave(group_counts)
    sorted_depths = torch.arange(n, dtype=torch.long) - expanded_starts

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


def _utilisation_table(util_per_layer: list, cb_size: int) -> list:
    """Render codebook utilisation as a dynamically aligned table."""
    headers = ["Layer", "Codes Used", "Coverage", "Perplexity", "Perp. Ratio"]

    cb_digits = len(str(cb_size))
    rows = []
    for u in util_per_layer:
        rows.append([
            f"L{u['layer'] + 1}",
            f"{str(u['codes_used']).rjust(cb_digits)} / {cb_size}",
            f"{u['coverage']:.1%}",
            f"{u['perplexity']:.1f}",
            f"{u['perplexity_ratio']:.1%}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    indent = "   "
    header_line = " │ ".join(h.center(w) for h, w in zip(headers, col_widths))
    separator   = "─┼─".join("─" * w for w in col_widths)

    align = [str.ljust, str.rjust, str.rjust, str.rjust, str.rjust]
    table = [f"{indent}{header_line}", f"{indent}{separator}"]
    for row in rows:
        table.append(indent + " │ ".join(fn(cell, w) for fn, cell, w in zip(align, row, col_widths)))
    return table


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
        "── Codebook Utilisation ──────────────────────────────────",
        "   Perplexity = exp(-∑ p_k log p_k),  range: [1, K]",
        "",
        *_utilisation_table(all_stats["utilisation"]["per_layer"], cb_size),
    ]

    c = all_stats["collision"]
    lines += [
        "",
        "── Collision Stats ───────────────────────────────────────",
        f"  Collision rate : {c['collision_rate']:.1%}  ({c['n_collisions']} items)",
        f"  Max depth      : {c['max_depth']}",
        f"  Mean / Median  : {c['mean_depth']:.2f} / {c['median_depth']:.1f}",
        f"  p90 / p99      : {c['p90_depth']:.1f} / {c['p99_depth']:.1f}",
        "=" * 60,
        "",
    ]
    print("\n".join(lines))


def evaluate_semids(raw_semids: torch.Tensor, config) -> dict:
    """
    Run all SID evaluation metrics and print a text report.

    Metrics:
      - Codebook utilisation + perplexity per level
      - Collision statistics

    Args:
        raw_semids  : (N, L) semantic IDs BEFORE collision resolution.
        config      : OmegaConf config object.

    Returns:
        dict with keys: utilisation, collision.
    """
    cb_size = getattr(getattr(config, "model", None), "codebook_clusters", 128)

    all_stats = {
        "utilisation": compute_utilisation(raw_semids, cb_size),
        "collision":   compute_collision_stats(raw_semids),
    }

    print_evaluation_report(all_stats, config)

    return all_stats
