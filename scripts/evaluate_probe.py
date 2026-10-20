#!/usr/bin/env python3
"""Evaluate a train_probe.py checkpoint exactly as alphagenome-pytorch evaluates its own.

JAX port of alphagenome-pytorch ``scripts/evaluate_checkpoint.py --metrics`` (commit
c2c32ee) for one 1-bp track head, so the two frameworks' results compare key by key:

* the run's ``best`` checkpoint (lowest validation loss) on every window of ``--test-bed``,
  centered to the training window size (``--max-regions`` takes the same seeded random
  subset as the PyTorch script);
* test loss: the segmented multinomial + count loss on full windows in scaled space
  (``NUM_SEGMENTS`` = 8, positional weight 5, count weight 1), averaged over batches;
* metrics: predictions in BigWig units, the central ``--score-window-bp`` cropped,
  sum-pooled to each ``--metric-bin-sizes`` resolution, then the PyTorch metric
  functions (ported verbatim below);
* outputs: ``summary.json``, ``summary.txt`` and the same ``metrics/`` plots.

Apache-2.0: the metric, plotting and summary code is from alphagenome-pytorch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NUM_SEGMENTS = 8  # alphagenome_pytorch.extensions.finetuning.training.NUM_SEGMENTS
POSITIONAL_WEIGHT = 5.0  # evaluate_split default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="train_probe.py --checkpoint-dir (with best/ and run_config.json)")
    p.add_argument("--base-checkpoint", type=Path, required=True, help="Pretrained JAX AlphaGenome checkpoint")
    p.add_argument("--genome", type=Path, required=True, help="Reference genome FASTA")
    p.add_argument("--bigwig", type=Path, required=True, help="BigWig signal file (the trained track)")
    p.add_argument("--test-bed", type=Path, required=True, help="Test split BED file")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--score-window-bp", type=int, default=None,
                   help="Centered scoring width in bp, applied after full-context inference")
    p.add_argument("--metric-bin-sizes", default=None,
                   help="Comma-separated scoring resolutions sum-pooled from the 1 bp output (e.g. 1,32)")
    p.add_argument("--metric-source-resolution", type=int, default=1)
    p.add_argument("--max-regions", type=int, default=0, help="Max regions for metrics (0: evaluate all)")
    return p.parse_args()


# =============================================================================
# Ported verbatim from alphagenome-pytorch scripts/evaluate_checkpoint.py (c2c32ee)
# =============================================================================


def plot_scatter(preds, targets, out_path: Path, title_suffix: str = "") -> None:
    """Scatter plot of predicted vs observed (subsampled)."""
    p = preds.flatten()
    t = targets.flatten()
    n = min(len(p), 100_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(p), n, replace=False)
    p, t = p[idx], t[idx]
    r = stats.pearsonr(p, t)[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.05, s=1, color="steelblue", rasterized=True)
    lim = max(t.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed signal")
    ax.set_ylabel("Predicted signal")
    ax.set_title(f"Pred vs Obs {title_suffix} (r={r:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_counts(preds, targets, out_path: Path, title_suffix: str = "") -> None:
    """Scatter plot of total counts per region."""
    pred_sums = preds.sum(axis=1).flatten()
    target_sums = targets.sum(axis=1).flatten()
    r = stats.pearsonr(pred_sums, target_sums)[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target_sums, pred_sums, alpha=0.15, s=5, color="steelblue", rasterized=True)
    lim = max(target_sums.max(), pred_sums.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed total count")
    ax.set_ylabel("Predicted total count")
    ax.set_title(f"Count correlation {title_suffix} (r={r:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_histogram(ft_values, out_path: Path, native_values=None,
                               xlabel: str = "Pearson r (per region)",
                               title: str = "Per-region correlation distribution",
                               ft_label: str = "Finetuned", native_label: str = "Native") -> None:
    """Histogram of per-region values, optionally overlaid with native."""
    ft_values = np.asarray(ft_values)
    ft_values = ft_values[np.isfinite(ft_values)]
    if native_values is not None:
        native_values = np.asarray(native_values)
        native_values = native_values[np.isfinite(native_values)]
    if ft_values.size == 0:
        log.warning("Skipping %s because no finite values are available", out_path)
        return
    if native_values is not None and native_values.size == 0:
        native_values = None

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(ft_values.min(), native_values.min() if native_values is not None else ft_values.min()),
        max(ft_values.max(), native_values.max() if native_values is not None else ft_values.max()),
        51,
    )
    ax.hist(ft_values, bins=bins, alpha=0.6, color="steelblue",
            edgecolor="white", label=f"{ft_label} (med={np.median(ft_values):.3f})")
    if native_values is not None:
        ax.hist(native_values, bins=bins, alpha=0.5, color="forestgreen",
                edgecolor="white", label=f"{native_label} (med={np.median(native_values):.3f})")
    ax.axvline(np.median(ft_values), color="steelblue", linestyle="--", linewidth=1.2)
    if native_values is not None:
        ax.axvline(np.median(native_values), color="forestgreen", linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclasses.dataclass
class _PearsonRState:
    xy_sum: np.ndarray
    x_sum: np.ndarray
    xx_sum: np.ndarray
    y_sum: np.ndarray
    yy_sum: np.ndarray
    count: np.ndarray

    def __add__(self, other: "_PearsonRState") -> "_PearsonRState":
        return _PearsonRState(*(
            getattr(self, field) + getattr(other, field)
            for field in self.__dataclass_fields__
        ))


def _pearsonr_initialize() -> _PearsonRState:
    return _PearsonRState(
        xy_sum=np.zeros((), dtype=np.float32),
        x_sum=np.zeros((), dtype=np.float32),
        xx_sum=np.zeros((), dtype=np.float32),
        y_sum=np.zeros((), dtype=np.float32),
        yy_sum=np.zeros((), dtype=np.float32),
        count=np.zeros((), dtype=np.float32),
    )


def _pearsonr_update(x, y, axis=None, mask=None) -> _PearsonRState:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    where = True if mask is None else mask
    return _PearsonRState(
        xy_sum=np.sum(x * y, axis=axis, where=where, dtype=np.float32),
        x_sum=np.sum(x, axis=axis, where=where, dtype=np.float32),
        xx_sum=np.sum(np.square(x), axis=axis, where=where, dtype=np.float32),
        y_sum=np.sum(y, axis=axis, where=where, dtype=np.float32),
        yy_sum=np.sum(np.square(y), axis=axis, where=where, dtype=np.float32),
        count=np.sum(np.ones_like(x), axis=axis, where=where, dtype=np.float32),
    )


def _pearsonr_result(state: _PearsonRState) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = state.x_sum / state.count
        y_mean = state.y_sum / state.count
        covariance = state.xy_sum - state.count * x_mean * y_mean
        x_var = state.xx_sum - state.count * x_mean * x_mean
        y_var = state.yy_sum - state.count * y_mean * y_mean
        variance = x_var**0.5 * y_var**0.5
        eps = np.finfo(variance.dtype).eps
        return covariance / (variance + eps)


def center_crop_profiles(preds, targets, score_window_bp, source_resolution):
    """Center-crop aligned arrays after full-context model inference."""
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} != {targets.shape}")
    if preds.ndim != 3:
        raise ValueError(f"Expected (regions, positions, tracks), got {preds.shape}")
    if source_resolution <= 0:
        raise ValueError("source_resolution must be positive")
    if score_window_bp is None:
        return preds, targets
    if score_window_bp <= 0 or score_window_bp % source_resolution:
        raise ValueError("score_window_bp must be positive and divisible by source_resolution")
    width = score_window_bp // source_resolution
    available = preds.shape[1]
    if width > available:
        raise ValueError(
            f"Requested {score_window_bp:,} bp scoring window, but predictions "
            f"cover only {available * source_resolution:,} bp."
        )
    left = (available - width) // 2
    right = left + width
    return preds[:, left:right, :], targets[:, left:right, :]


def pool_profiles(values, factor: int, reduction: str = "sum"):
    """Aggregate adjacent positions with a sum or mean reduction."""
    if factor <= 0:
        raise ValueError("Pooling factor must be positive")
    if factor == 1:
        return values
    if values.shape[1] % factor:
        raise ValueError(f"Profile length {values.shape[1]} is not divisible by pooling factor {factor}")
    n_regions, n_positions, n_tracks = values.shape
    grouped = values.reshape(n_regions, n_positions // factor, factor, n_tracks)
    if reduction == "sum":
        return grouped.sum(axis=2)
    if reduction == "mean":
        return grouped.mean(axis=2)
    raise ValueError(f"Unknown pooling reduction: {reduction}")


def build_metric_views(preds, targets, *, source_resolution, bin_sizes, score_window_bp, reduction="sum"):
    """Crop once, then construct aligned metric-resolution views."""
    cropped_preds, cropped_targets = center_crop_profiles(preds, targets, score_window_bp, source_resolution)
    pred_views, target_views = {}, {}
    for bin_size in bin_sizes:
        if bin_size < source_resolution or bin_size % source_resolution:
            raise ValueError(f"Metric bin size {bin_size} must be a multiple of source resolution {source_resolution}")
        factor = bin_size // source_resolution
        pred_views[bin_size] = pool_profiles(cropped_preds, factor, reduction)
        target_views[bin_size] = pool_profiles(cropped_targets, factor, reduction)
    return pred_views, target_views


def _pearson_or_nan(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def compute_comparison_metrics(preds, targets, *, profile_transform: str = "none") -> dict:
    """Compute per-region profiles, regional counts, and profile JSD metrics."""
    if preds.shape != targets.shape or preds.ndim != 3:
        raise ValueError("preds and targets must have matching (regions, positions, tracks) shapes")
    if np.any(preds < 0) or np.any(targets < 0):
        raise ValueError("Jensen-Shannon metrics require non-negative profiles")

    if profile_transform == "none":
        profile_preds, profile_targets = preds, targets
    elif profile_transform == "log1p":
        profile_preds, profile_targets = np.log1p(preds), np.log1p(targets)
    else:
        raise ValueError(f"Unknown profile transform: {profile_transform}")

    profile_rs = np.asarray([
        _pearson_or_nan(profile_preds[i], profile_targets[i])
        for i in range(preds.shape[0])
    ])
    pred_counts = preds.sum(axis=1).reshape(-1)
    target_counts = targets.sum(axis=1).reshape(-1)
    count_raw = _pearson_or_nan(pred_counts, target_counts)
    count_log1p = _pearson_or_nan(np.log1p(pred_counts), np.log1p(target_counts))

    eps = 1e-8
    p = targets / (targets.sum(axis=1, keepdims=True) + eps)
    q = preds / (preds.sum(axis=1, keepdims=True) + eps)
    m = 0.5 * (p + q)
    js_by_track = 0.5 * (
        np.sum(p * np.log((p + eps) / (m + eps)), axis=1)
        + np.sum(q * np.log((q + eps) / (m + eps)), axis=1)
    )
    js_divergence = np.mean(js_by_track, axis=1)
    js_distance = np.sqrt(np.maximum(js_divergence, 0.0))

    pearson_state = _pearsonr_initialize()
    pearson_state += _pearsonr_update(profile_targets, profile_preds, axis=(-2, -3))
    track_pearson = _pearsonr_result(pearson_state)
    counted_tracks = pearson_state.count > 0

    flat_preds = profile_preds.reshape(-1)
    flat_targets = profile_targets.reshape(-1)
    if flat_preds.size > 2_000_000:
        idx = np.random.default_rng(42).choice(flat_preds.size, 2_000_000, replace=False)
        flat_preds, flat_targets = flat_preds[idx], flat_targets[idx]

    return {
        "profile_transform": profile_transform,
        "profile_pearson_r_all": profile_rs,
        "profile_pearson_r_mean": float(np.nanmean(profile_rs)),
        "profile_pearson_r_median": float(np.nanmedian(profile_rs)),
        "profile_pearson_r_n_valid": int(np.isfinite(profile_rs).sum()),
        "track_pearson_r_accumulated_all": track_pearson,
        "track_pearson_r_accumulated_mean": float(np.mean(track_pearson, where=counted_tracks)),
        "track_pearson_r_accumulated_n_valid": int(counted_tracks.sum()),
        "count_pearson_r": count_raw,
        "count_pearson_r_raw": count_raw,
        "count_pearson_r_log1p": count_log1p,
        "jsd_all": js_divergence,
        "jsd_mean": float(np.nanmean(js_divergence)),
        "jsd_median": float(np.nanmedian(js_divergence)),
        "js_divergence_all": js_divergence,
        "js_divergence_mean": float(np.nanmean(js_divergence)),
        "js_distance_all": js_distance,
        "js_distance_mean": float(np.nanmean(js_distance)),
        "js_distance_median": float(np.nanmedian(js_distance)),
        "mse": float(np.mean(np.square(preds - targets))),
        "spearman_global": float(stats.spearmanr(flat_preds, flat_targets)[0]),
        "n_regions": int(preds.shape[0]),
        "n_positions_per_region": int(preds.shape[1]),
        "n_tracks": int(preds.shape[2]),
    }


def format_summary_table(ft_metrics: dict, resolutions) -> str:
    """Format a human-readable summary table (finetuned column only)."""
    lines = []
    for res in resolutions:
        ft = ft_metrics.get(res)
        if ft is None:
            continue
        lines.append(f"\n--- {res}bp resolution ---")
        header = f"{'Metric':<28}{'Finetuned':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        rows = [
            ("Profile r (mean)", "profile_pearson_r_mean"),
            ("Profile r (median)", "profile_pearson_r_median"),
            ("Track r (accumulated)", "track_pearson_r_accumulated_mean"),
            ("Count r (raw)", "count_pearson_r_raw"),
            ("Count r (log1p; paper)", "count_pearson_r_log1p"),
            ("JS divergence (mean)", "js_divergence_mean"),
            ("JS distance (mean; paper)", "js_distance_mean"),
            ("MSE", "mse"),
            ("Spearman (global)", "spearman_global"),
        ]
        for label, key in rows:
            lines.append(f"{label:<28}{ft.get(key, float('nan')):>12.4f}")
        lines.append(f"{'N regions':<28}{ft['n_regions']:>12d}")
    return "\n".join(lines)


def save_summary_json(ft_metrics, checkpoint_meta, loss, out_path: Path, evaluation_info) -> None:
    """Save machine-readable JSON summary."""
    def _clean(m: dict) -> dict:
        return {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}

    data: dict = {"checkpoint": checkpoint_meta}
    if evaluation_info:
        data["evaluation"] = evaluation_info
    if loss is not None:
        data["loss"] = loss
    if ft_metrics:
        data["finetuned"] = {str(res): _clean(m) for res, m in ft_metrics.items()}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# JAX model loading and inference (the only framework-specific part)
# =============================================================================


def load_test_windows(test_bed: Path, genome: Path, window_size: int):
    """Center each interval on ``window_size`` and skip out-of-bounds windows (train_probe rule)."""
    import pandas as pd

    sizes = dict(line.split()[:2] for line in open(f"{genome}.fai"))
    frame = pd.read_csv(test_bed, sep=r"\s+", comment="#", header=None,
                        usecols=(0, 1, 2), names=("chromosome", "start", "end"))
    frame["start"] = (frame["start"] + frame["end"]) // 2 - window_size // 2
    frame["end"] = frame["start"] + window_size
    size = frame["chromosome"].map(lambda c: int(sizes[c]) if c in sizes else -1)
    return frame[(size > 0) & (frame["start"] >= 0) & (frame["end"] <= size)].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    checkpoint = args.run_dir / "best"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    log.info("Output: %s", out_dir)

    bin_sizes = (tuple(int(b) for b in args.metric_bin_sizes.split(","))
                 if args.metric_bin_sizes else None)
    source = args.metric_source_resolution
    if source != 1:
        raise ValueError("train_probe heads predict at 1 bp; --metric-source-resolution must be 1")
    metric_resolutions = bin_sizes or (1,)
    window_size = int(run_config["window_size"])

    import jax
    import jax.numpy as jnp
    import pandas as pd
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model
    from alphagenome_ft import load_checkpoint
    from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
    from alphagenome_ft.finetune.data import BigWigDataModule, load_intervals_from_dataframe
    from alphagenome_ft.finetune.evaluation import make_track_loss
    from alphagenome_ft.finetune.train import register_predefined_heads

    head = run_config["head_name"]
    output_type = run_config["output_type"]
    labels = [t["label"] for t in run_config["targets"]]
    track_means = run_config["track_nonzero_means"]
    if len(labels) != 1:
        raise ValueError("Expected a single target track")
    apply_squashing = output_type == "rna_seq"
    # Same head registration as train_probe.py, with the run's saved track scaling.
    specs = prepare_head_specs(load_targets_config({"heads": [{
        "id": head, "source": "predefined", "kind": output_type, "resolutions": [1],
        "apply_squashing": apply_squashing,
        "targets": [{"label": labels[0], "path": str(args.bigwig), "nonzero_mean": float(track_means[0])}],
    }]}), organism="HOMO_SAPIENS")
    register_predefined_heads(specs)
    settings = {
        ag_dna_model.Organism.HOMO_SAPIENS: research_dna_model.OrganismSettings(fasta_path=str(args.genome)),
        ag_dna_model.Organism.MUS_MUSCULUS: research_dna_model.OrganismSettings(),
    }
    model = load_checkpoint(checkpoint, base_model_version=run_config["model_version"],
                            base_checkpoint_path=args.base_checkpoint, organism_settings=settings)
    best = json.loads((checkpoint / "best_metric.json").read_text())
    modality = output_type.lower()
    ckpt_meta = {"modality": [modality], "track_names": {modality: labels},
                 "epoch": best["epoch"], "val_loss": best["value"]}
    log.info("Modality: %s, Tracks: %s, Resolutions: (1,)", modality, labels)
    log.info("Checkpoint epoch: %s, val_loss: %s", best["epoch"], best["value"])

    log.info("=" * 60)
    log.info("Computing test set metrics")
    log.info("=" * 60)
    frame = load_test_windows(args.test_bed, args.genome, window_size)
    if args.max_regions and len(frame) > args.max_regions:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(frame), args.max_regions, replace=False).tolist()
        frame = frame.iloc[indices].reset_index(drop=True)
    frame["split"] = "test"
    # The data module requires a train split; a one-row placeholder is never iterated.
    intervals = load_intervals_from_dataframe(
        pd.concat([frame.iloc[:1].assign(split="train"), frame], ignore_index=True), window_size=None)
    dm = BigWigDataModule(intervals={"train": intervals["test"][:1], "test": intervals["test"]},
                          fasta_path=args.genome, head_specs=specs, batch_size=1, shuffle=False)
    windows = dm._intervals["test"]

    loss_fn = make_track_loss(track_means, apply_squashing=apply_squashing, num_segments=NUM_SEGMENTS,
                              positional_weight=POSITIONAL_WEIGHT, count_weight=1.0)
    organism = ag_dna_model.Organism.HOMO_SAPIENS
    org_index = research_dna_model.convert_to_organism_index(organism)
    strand = model._metadata[organism].strand_reindexing

    @jax.jit
    def predict(params, state, seqs, org, neg, targets):
        preds = model._predict(params, state, seqs, org, negative_strand_mask=neg,
                               strand_reindexing=strand, requested_outputs=frozenset())[head]
        loss = loss_fn(preds, {"targets": targets, "organism_index": org})["loss"]
        return preds["predictions_1bp"], loss  # unscaled predictions, BigWig units

    # Keep only the scored center of each window: the same positions center_crop_profiles
    # selects from the full arrays (it is then a no-op), at a fraction of the host memory.
    n = len(windows)
    kept = args.score_window_bp or window_size
    if kept > window_size or (window_size - kept) % 2:
        raise ValueError("--score-window-bp must not exceed the window and must center evenly")
    left = (window_size - kept) // 2
    preds = np.zeros((n, kept, 1), dtype=np.float32)
    targets = np.zeros((n, kept, 1), dtype=np.float32)
    total_loss = 0.0
    with model._device_context:
        for i, batch in enumerate(dm.iter_batches("test")):
            target = np.asarray(batch[f"targets_{head}"], dtype=np.float32)
            pred, loss = predict(model._params, model._state, jnp.asarray(batch["sequences"]),
                                 jnp.full((len(batch["sequences"]),), org_index, dtype=jnp.int32),
                                 jnp.asarray(batch["negative_strand_mask"]), jnp.asarray(target))
            preds[i] = np.asarray(jax.device_get(pred), dtype=np.float32)[0, left:left + kept]
            targets[i] = target[0, left:left + kept]
            total_loss += float(loss)
            if (i + 1) % 500 == 0 or i + 1 == n:
                log.info("Evaluating (finetuned): %d/%d", i + 1, n)
    loss = total_loss / max(n, 1)
    log.info("Loss: %.4f", loss)

    ft_metric_preds, metric_targets = build_metric_views(
        preds, targets, source_resolution=source, bin_sizes=metric_resolutions,
        score_window_bp=args.score_window_bp, reduction="sum",
    )
    del preds, targets
    ft_metrics_by_res = {}
    for res in metric_resolutions:
        p, t = ft_metric_preds[res], metric_targets[res]
        log.info("Resolution %dbp: preds %s, targets %s", res, p.shape, t.shape)
        ft_m = compute_comparison_metrics(p, t, profile_transform="none")
        ft_metrics_by_res[res] = ft_m
        log.info("  Profile r (mean):  %.4f", ft_m["profile_pearson_r_mean"])
        log.info("  Track r (accum.):  %.4f", ft_m["track_pearson_r_accumulated_mean"])
        log.info("  Count r (raw):     %.4f", ft_m["count_pearson_r_raw"])
        log.info("  Count r (log1p):   %.4f", ft_m["count_pearson_r_log1p"])
        log.info("  JS divergence:     %.4f", ft_m["js_divergence_mean"])
        log.info("  JS distance:       %.4f", ft_m["js_distance_mean"])
        plot_scatter(p, t, metrics_dir / f"scatter_test_{res}bp.png", title_suffix=f"(test, {res}bp)")
        plot_scatter_counts(p, t, metrics_dir / f"scatter_counts_test_{res}bp.png",
                            title_suffix=f"(test, {res}bp)")
        plot_correlation_histogram(ft_m["profile_pearson_r_all"],
                                   metrics_dir / f"correlation_hist_test_{res}bp.png",
                                   xlabel="Pearson r (per region)",
                                   title=f"Profile correlation distribution ({res}bp)")
        plot_correlation_histogram(ft_m["jsd_all"], metrics_dir / f"jsd_hist_test_{res}bp.png",
                                   xlabel="JSD (per region)", title=f"JSD distribution ({res}bp)")

    summary_text = format_summary_table(ft_metrics_by_res, metric_resolutions)
    if summary_text:
        print(summary_text)
        with open(out_dir / "summary.txt", "w") as f:
            f.write(summary_text)
    save_summary_json(
        ft_metrics_by_res, ckpt_meta, loss, out_dir / "summary.json",
        evaluation_info={
            "test_bed": str(args.test_bed),
            "sequence_length_bp": window_size,
            "score_window_bp": args.score_window_bp or window_size,
            "metric_source_resolution_bp": source,
            "metric_bin_sizes_bp": list(metric_resolutions),
            "pooling": "sum" if bin_sizes else "native_model_output",
            "profile_transform": "none",
            "fold": "fold_1 expected; determined by --test-bed",
        },
    )
    log.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
