"""One-bp track losses and evaluation matching the PyTorch fine-tuning CLI."""

from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
from alphagenome.models import dna_model
from alphagenome_research.model import dna_model as research_dna_model
from alphagenome_research.model import heads, losses

from .data import prepare_batch


def make_track_loss(track_means, *, apply_squashing=False, num_segments=8,
                    positional_weight=5.0, count_weight=1.0):
    """Scale raw targets and split the actual sequence into equal segments.

    Unlike the pretrained head's fixed 131072-bp segment size, the PyTorch
    fine-tuning CLI uses sequence_length / num_segments at every context size.
    This factory is for the driver's single-organism, 1-bp track heads.
    """
    means = np.asarray(track_means, dtype=np.float32)
    if means.ndim != 1 or not np.all(np.isfinite(means) & (means > 0)):
        raise ValueError("track_means must be a positive finite vector")
    if num_segments < 1:
        raise ValueError("num_segments must be positive")

    def loss_fn(predictions, batch):
        pred = predictions["scaled_predictions_1bp"]
        targets = batch["targets"]
        if pred.shape[-1] != len(means):
            raise ValueError("Prediction track count does not match track_means")
        if pred.shape[-2] % num_segments:
            raise ValueError("Sequence length must be divisible by num_segments")
        scaled = heads.targets_scaling(
            targets, track_means=jnp.asarray(means)[None, :], resolution=1,
            apply_squashing=apply_squashing,
        )
        result = losses.multinomial_loss(
            y_true=scaled, y_pred=pred,
            mask=jnp.ones((pred.shape[0], 1, pred.shape[-1]), dtype=bool),
            multinomial_resolution=pred.shape[-2] // num_segments,
            positional_weight=positional_weight,
        )
        return {**result, "loss": count_weight * result["loss_total"]
                + positional_weight * result["loss_positional"]}

    return loss_fn


def pearson_r(pred, true, axis):
    """Match torch.metrics.pearson_r, including zero for constant profiles."""
    pred, true = jnp.asarray(pred, jnp.float32), jnp.asarray(true, jnp.float32)
    pred = pred - pred.mean(axis=axis, keepdims=True)
    true = true - true.mean(axis=axis, keepdims=True)
    numerator = (pred * true).sum(axis=axis)
    denominator = jnp.sqrt((pred * pred).sum(axis=axis)) * jnp.sqrt((true * true).sum(axis=axis))
    return numerator / (denominator + 1e-8)


def summarize_batches(batches):
    """Reduce losses by batch, profiles by region/track, counts over all regions.

    Keep only per-region correlations and counts on the host, never full genomic
    predictions. A partial final batch has equal weight in loss, as in PyTorch.
    """
    batches = list(batches)
    if not batches:
        raise ValueError("Cannot evaluate an empty split")
    result = {}
    for name in batches[0]:
        rows = [batch[name] for batch in batches]
        metrics = {key: float(np.mean([row[key] for row in rows]))
                   for key in ("loss", "loss_total", "loss_positional")}
        profiles = np.concatenate([row["profile"].reshape(-1, row["profile"].shape[-1]) for row in rows])
        pred = np.concatenate([row["pred_counts"].reshape(-1, row["pred_counts"].shape[-1]) for row in rows])
        true = np.concatenate([row["true_counts"].reshape(-1, row["true_counts"].shape[-1]) for row in rows])
        count_r = np.asarray(pearson_r(pred, true, axis=0)) if len(pred) > 1 else np.full(pred.shape[-1], np.nan)
        metrics.update(
            profile_pearson_r_mean=float(profiles.mean()),
            profile_pearson_r_std=float(profiles.std(ddof=1)) if profiles.size > 1 else float("nan"),
            count_pearson_r=float(count_r.mean()),
            profile_pearson_r_per_track=profiles.mean(axis=0).tolist(),
            count_pearson_r_per_track=count_r.tolist(),
            num_examples=len(pred),
        )
        result[name] = metrics
    return result


def create_track_evaluator(model, loss_fns, *, organism="HOMO_SAPIENS", num_devices=1):
    """Build once and reuse the compiled evaluator during training and testing."""
    # Local import avoids a cycle with train's optional evaluation integration.
    from .train import _replicate_tree, _shard_batch

    devices = jax.local_devices()[:num_devices]
    if len(devices) != num_devices:
        raise ValueError("Not enough local devices")
    organism_enum = getattr(dna_model.Organism, organism)
    organism_index = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = _replicate_tree(model._metadata[organism_enum].strand_reindexing, devices)
    names = list(loss_fns)

    @functools.partial(jax.pmap, axis_name="data")
    def step(params, state, batch):
        predictions = model._predict(
            params, state, batch["sequences"], batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=batch["strand_reindexing"], requested_outputs=frozenset(),
        )
        result = {}
        for name in names:
            target = batch[f"targets_{name}"]
            loss = loss_fns[name](predictions[name], {"targets": target, "organism_index": batch["organism_index"]})
            pred = predictions[name]["predictions_1bp"]
            result[name] = {
                **{key: jax.lax.pmean(loss[key], "data") for key in ("loss", "loss_total", "loss_positional")},
                "profile": pearson_r(pred, target, axis=1),
                "pred_counts": pred.sum(axis=1), "true_counts": target.sum(axis=1),
            }
        return result

    def evaluate(data_module, split, *, params=None, state=None):
        # Training supplies replicated trees. Standalone evaluation uses model weights.
        # The upstream device context owns a non-reentrant lock. The trainer
        # already holds it when supplying replicated parameters.
        context = model._device_context if params is None else contextlib.nullcontext()
        with context:
            if params is None:
                params = _replicate_tree(model._params, devices)
                state = _replicate_tree(model._state, devices)

            def batches():
                for batch_np in data_module.iter_batches(split):
                    batch = _shard_batch(prepare_batch(batch_np, organism_index, names), num_devices)
                    batch["strand_reindexing"] = strand_reindexing
                    yield jax.device_get(step(params, state, batch))

            return summarize_batches(batches())

    return evaluate
