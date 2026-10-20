#!/usr/bin/env python3
"""Train a C8-compatible probing head with a frozen backbone and no adapters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


WINDOW_SIZE_1MB = 1_048_576


def parse_target(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("targets must use LABEL=/path/to/track.bw")
    label, path_value = value.split("=", 1)
    label = label.strip()
    if not label or not path_value.strip():
        raise argparse.ArgumentTypeError("target label and path must both be non-empty")
    return label, Path(path_value).expanduser().resolve()


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--head-name", required=True)
    parser.add_argument("--output-type", choices=("atac", "rna_seq", "procap"), required=True)
    parser.add_argument("--target", action="append", type=parse_target, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--fold", default="1", choices=("0", "1", "2", "3"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE_1MB)
    parser.add_argument("--model-version", default="fold_1")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Saved best/last directory; restores weights with a fresh optimizer")
    parser.add_argument("--completed-epochs", type=int, default=0, help="Completed epochs before this continuation; --epochs remains the original total")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=optional_positive_int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--head-init-scheme", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--expected-total-steps", type=optional_positive_int)
    parser.add_argument("--expected-trainable-params", type=optional_positive_int)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--track-means-samples", type=optional_positive_int, default=1000
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=optional_positive_int)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--limit-train", type=optional_positive_int)
    parser.add_argument("--limit-valid", type=optional_positive_int)
    parser.add_argument("--limit-test", type=optional_positive_int)
    parser.add_argument("--preflight", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if args.resume_from is not None:
        errors.append("Probing launcher requires a fresh run; optimizer resume is not implemented")
    if args.window_size not in (32768, 131072, 262144, WINDOW_SIZE_1MB):
        errors.append("--window-size must be 32768, 131072, 262144, or 1048576")
    for name in ("batch_size", "epochs", "lora_rank", "num_devices"):
        if getattr(args, name) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmup_steps", "early_stopping_patience", "early_stopping_min_epochs"):
        if getattr(args, name) < 0:
            errors.append(f"--{name.replace('_', '-')} cannot be negative")
    if args.num_devices > 0 and args.batch_size % args.num_devices:
        errors.append("--batch-size must be divisible by --num-devices")
    if not 0 <= args.completed_epochs < args.epochs:
        errors.append("--completed-epochs must be nonnegative and below --epochs")
    if args.completed_epochs and args.resume_from is None:
        errors.append("--completed-epochs requires --resume-from")
    if args.resume_from is not None:
        try:
            resume_config(args)
        except (ValueError, OSError, KeyError) as error:
            errors.append(f"Invalid resume configuration: {error}")

    from alphagenome_ft.finetune.data import normalize_model_fold

    try:
        selected_fold = normalize_model_fold(args.fold)
        if normalize_model_fold(args.model_version) != selected_fold:
            errors.append("--model-version must match --fold to preserve held-out partitions")
        if args.split_dir is not None and args.split_dir.name.lower().startswith("fold_"):
            if normalize_model_fold(args.split_dir.name) != selected_fold:
                errors.append("--split-dir fold must match --fold and --model-version")
    except ValueError as error:
        errors.append(str(error))

    required_files = [args.fasta, Path(f"{args.fasta}.fai")]
    required_files.extend(path for _label, path in args.target)
    if args.split_dir is not None:
        required_files.extend(args.split_dir / f"{split}.bed" for split in ("train", "valid", "test"))
    if args.base_checkpoint is not None:
        required_files.append(args.base_checkpoint)
    for path in required_files:
        if not path.exists():
            errors.append(f"required input does not exist: {path}")

    labels = [label for label, _path in args.target]
    if len(labels) != len(set(labels)):
        errors.append("target labels must be unique")
    return errors


def serializable_config(args: argparse.Namespace) -> dict:
    return {
        "run_name": args.run_name,
        "adapter_scope": None, "mode": "linear_probe",
        "head_name": args.head_name,
        "output_type": args.output_type,
        "targets": [{"label": label, "path": str(path)} for label, path in args.target],
        "fasta": str(args.fasta.resolve()),
        "split_dir": str(args.split_dir.resolve()) if args.split_dir else None,
        "fold": args.fold,
        "window_size": args.window_size,
        "model_version": args.model_version,
        "base_checkpoint": str(args.base_checkpoint.resolve()) if args.base_checkpoint else None,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "completed_epochs": args.completed_epochs,
        "optimizer_state_restored": False,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_schedule": args.lr_schedule,
        "max_grad_norm": args.max_grad_norm,
        "expected_total_steps": args.expected_total_steps,
        "expected_trainable_params": args.expected_trainable_params,
        "initialization_seed": 42,
        "head_init_scheme": args.head_init_scheme,
        "shuffle_seed": args.seed,
        "lora_rank": 0,
        "lora_alpha": None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_targets": [],
        "track_means_samples": args.track_means_samples,
        "seed": args.seed,
        "max_train_steps": args.max_train_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
        "num_devices": args.num_devices,
        "limits": {
            "train": args.limit_train,
            "valid": args.limit_valid,
            "test": args.limit_test,
        },
        "loss": {
            "type": "segmented_multinomial_count",
            "num_segments": 8,
            "segment_size_bp": args.window_size // 8,
            "positional_weight": 5.0,
            "count_weight": 1.0,
        },
        "target_scaling": "per_track_nonzero_mean",
        "backbone": "frozen",
        "frozen_parameter_gradients": "stopped_inside_loss",
    }


def make_learning_rate_schedule(peak, warmup_steps, total_steps, schedule):
    """Match PyTorch LambdaLR, including steps past its rounded-down horizon."""
    import jax.numpy as jnp

    def value(step):
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        decay = 1.0 if schedule == "constant" else 0.5 * (1 + jnp.cos(jnp.pi * progress))
        return peak * jnp.where(step < warmup_steps, step / max(1, warmup_steps), decay)

    return value


def resume_config(args: argparse.Namespace) -> dict:
    """Reject incompatible continuation settings before loading the model."""
    source = args.resume_from.expanduser().resolve()
    config = json.loads((source.parent / "run_config.json").read_text())
    saved = json.loads((source / "config.json").read_text())
    if not (source / "checkpoint" / "_METADATA").is_file():
        raise ValueError(f"Missing checkpoint metadata in {source}")
    if args.checkpoint_dir.expanduser().resolve() == source.parent:
        raise ValueError("Use a new output directory to preserve the source checkpoints")
    current = serializable_config(args)
    for key in (
        "head_name", "output_type", "fold", "window_size", "model_version",
        "batch_size", "epochs", "learning_rate", "weight_decay", "warmup_steps",
        "lr_schedule", "max_grad_norm", "lora_rank", "lora_alpha", "lora_targets",
        "gradient_checkpointing", "seed", "num_devices", "limits",
    ):
        if current[key] != config[key]:
            raise ValueError(f"{key} differs from the source run: {current[key]!r} != {config[key]!r}")
    if config.get("gradient_accumulation_steps", 1) != args.gradient_accumulation_steps:
        raise ValueError("gradient_accumulation_steps differs from the source run")
    # Track paths may change when staging identical data onto local storage.
    if [t["label"] for t in config["targets"]] != [label for label, _ in args.target]:
        raise ValueError("Target labels/order differ from the source run")
    if saved["custom_heads"] != [args.head_name] or not saved.get("backbone_lora"):
        raise ValueError("Expected the requested head and backbone LoRA in the checkpoint")
    means = config["track_nonzero_means"]
    if len(means) != len(args.target) or any(not math.isfinite(x) or x <= 0 for x in means):
        raise ValueError("Invalid saved track normalization")
    if args.max_train_steps is not None or config.get("max_train_steps") is not None:
        raise ValueError("Epoch-based continuation requires uncapped source and destination runs")
    return config


def infer_strand_pairs(labels: list[str]) -> list[tuple[int, int]]:
    """Pair ``*_forward``/``*_reverse`` tracks like the PyTorch CLI."""
    by_stem: dict[str, dict[str, int]] = {}
    for index, label in enumerate(labels):
        lowered = label.lower()
        for suffix, strand in (("_forward", "forward"), ("_reverse", "reverse")):
            if lowered.endswith(suffix):
                by_stem.setdefault(lowered[: -len(suffix)], {})[strand] = index
                break
    return [
        (indices["forward"], indices["reverse"])
        for indices in by_stem.values()
        if set(indices) == {"forward", "reverse"}
    ]


def print_preflight(args: argparse.Namespace) -> int:
    errors = validate_args(args)
    print(json.dumps(serializable_config(args), indent=2))
    modules = (
        "jax",
        "haiku",
        "optax",
        "pyBigWig",
        "alphagenome",
        "alphagenome_research",
        "alphagenome_ft",
    )
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("Python environment is missing: " + ", ".join(missing))
    if args.base_checkpoint is None:
        print(
            "Base model: Kaggle fold_1 download/cache (Kaggle credentials are required "
            "unless the model is already cached)."
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Preflight passed.")
    return 0


def load_local_splits(args: argparse.Namespace):
    import pandas as pd

    from alphagenome_ft.finetune.data import load_intervals_from_dataframe

    # Match GenomicDataset: center and skip invalid windows, never clamp them.
    chrom_sizes = {}
    with Path(f"{args.fasta}.fai").open() as handle:
        for line in handle:
            fields = line.split()
            chrom_sizes[fields[0]] = int(fields[1])
    frames = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(
            args.split_dir / f"{split}.bed",
            sep=r"\s+",
            comment="#",
            header=None,
            usecols=(0, 1, 2),
            names=("chromosome", "start", "end"),
        )
        center = (frame["start"] + frame["end"]) // 2
        frame["start"] = center - args.window_size // 2
        frame["end"] = frame["start"] + args.window_size
        sizes = frame["chromosome"].map(chrom_sizes)
        keep = sizes.notna() & (frame["start"] >= 0) & (frame["end"] <= sizes)
        print(f"{split}: skipped {int((~keep).sum())} out-of-bounds/unknown FASTA intervals")
        frame = frame.loc[keep].copy()
        frame["split"] = split
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return load_intervals_from_dataframe(
        combined,
        window_size=None,
        limit_train=args.limit_train,
        limit_valid=args.limit_valid,
        limit_test=args.limit_test,
    )


def run(args: argparse.Namespace) -> int:
    errors = validate_args(args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model

    from alphagenome_ft import (
        create_model_with_heads,
        load_checkpoint,
        lora,
        parameter_utils,
    )
    from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
    from alphagenome_ft.finetune.data import (
        BigWigDataModule,
        compute_track_nonzero_means,
        prepare_intervals_from_fold,
    )
    from alphagenome_ft.finetune.train import register_predefined_heads, train
    from alphagenome_ft.finetune.evaluation import make_track_loss, create_track_evaluator

    if args.split_dir is not None:
        intervals = load_local_splits(args)
    else:
        intervals = prepare_intervals_from_fold(
            fold=args.fold,
            window_size=args.window_size,
            organism="HOMO_SAPIENS",
            limit_train=args.limit_train,
            limit_valid=args.limit_valid,
            limit_test=args.limit_test,
        )
    for split in ("train", "valid", "test"):
        print(f"{split}: {len(intervals.get(split, ())):,} interval(s)")

    labels = [label for label, _path in args.target]
    strand_pairs = infer_strand_pairs(labels)
    source_config = resume_config(args) if args.resume_from else None
    if source_config is not None:
        import numpy as np
        track_means = np.asarray(source_config["track_nonzero_means"], dtype=np.float32)
    else:
        track_means = compute_track_nonzero_means(
            [path for _label, path in args.target],
            intervals["train"],
            max_samples=args.track_means_samples,
            strand_pair_groups=strand_pairs or None,
        )
    print(f"Track nonzero means: {track_means.tolist()}")

    # Reuse the upstream prediction head, but pass an explicit fine-tuning
    # objective below: the upstream loss fixes segment size at 131072 bp.
    targets_config = {
        "heads": [
            {
                "id": args.head_name,
                "source": "predefined",
                "kind": args.output_type,
                "resolutions": [1],
                "apply_squashing": args.output_type == "rna_seq",
                "targets": [
                    {
                        "label": label,
                        "path": str(path),
                        "nonzero_mean": float(nonzero_mean),
                    }
                    for (label, path), nonzero_mean in zip(args.target, track_means)
                ],
            }
        ]
    }
    head_specs = prepare_head_specs(
        load_targets_config(targets_config), organism="HOMO_SAPIENS"
    )
    register_predefined_heads(head_specs)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if (checkpoint_dir / "best").exists() or (checkpoint_dir / "last").exists():
        raise FileExistsError(
            f"checkpoint output already exists under {checkpoint_dir}; choose a new RUN_TAG or "
            "CHECKPOINT_DIR"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = serializable_config(args)
    run_config["track_nonzero_means"] = track_means.tolist()
    run_config["strand_pair_groups"] = [list(pair) for pair in strand_pairs]
    (checkpoint_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )

    model_kwargs = {
        "head_init_scheme": args.head_init_scheme,
        "heads": [args.head_name],
        "detach_backbone": True,
        # Keep both organisms so checkpoint shapes match, but point human sequence
        # access at the local FASTA and disable unused remote annotation resources.
        "organism_settings": {
            ag_dna_model.Organism.HOMO_SAPIENS: research_dna_model.OrganismSettings(
                fasta_path=str(args.fasta.resolve())
            ),
            ag_dna_model.Organism.MUS_MUSCULUS: research_dna_model.OrganismSettings(),
        },
    }
    if args.base_checkpoint is not None:
        model_kwargs["checkpoint_path"] = str(args.base_checkpoint.resolve())
    if args.resume_from:
        model = load_checkpoint(
            args.resume_from,
            base_model_version=args.model_version,
            base_checkpoint_path=args.base_checkpoint,
            organism_settings=model_kwargs["organism_settings"],
        )
        print("Restored head + LoRA weights. Adam optimizer state starts fresh.")
    else:
        model = create_model_with_heads(args.model_version, **model_kwargs)

    total_params = parameter_utils.count_parameters(model._params)
    lora_params = lora.count_lora_parameters(model._params)
    if lora_params != 0:
        raise ValueError("Probing must not contain LoRA parameters")
    print(f"Total parameters: {total_params:,}")
    print(f"Transformer LoRA parameters: {lora_params:,} ({lora_params / total_params:.6%})")
    from alphagenome_ft.finetune.train import _label_params_for_heads
    import jax

    param_labels = _label_params_for_heads(model._params, [args.head_name], include_lora=False)
    trainable_params = sum(
        parameter.size
        for parameter, label in zip(
            jax.tree_util.tree_leaves(model._params),
            jax.tree_util.tree_leaves(param_labels),
        )
        if label == "head"
    )
    print(f"Trainable probing-head parameters: {trainable_params:,}")
    if args.expected_trainable_params is not None and trainable_params != args.expected_trainable_params:
        raise ValueError(f"Trainable parameter mismatch: {trainable_params} != {args.expected_trainable_params}")

    data_module = BigWigDataModule(
        intervals=intervals,
        fasta_path=args.fasta.resolve(),
        head_specs=head_specs,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.num_devices > 1,
    )
    microbatches_per_epoch = (
        len(data_module._intervals["train"]) // args.batch_size
        if data_module._drop_last
        else math.ceil(len(data_module._intervals["train"]) / args.batch_size)
    )
    steps_per_epoch = math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * args.epochs
    # PyTorch schedules using floor(total microbatches / accumulation), but
    # flushes a partial accumulation group at each epoch boundary.
    schedule_steps = (microbatches_per_epoch * args.epochs) // args.gradient_accumulation_steps
    if args.expected_total_steps is not None and schedule_steps != args.expected_total_steps:
        raise ValueError(f"Planned step mismatch: {schedule_steps} != {args.expected_total_steps}")
    total_steps = min(planned_steps, args.max_train_steps or planned_steps)
    completed_steps = steps_per_epoch * args.completed_epochs
    if source_config is not None:
        if steps_per_epoch != source_config["steps_per_epoch"] or total_steps != source_config["total_steps"]:
            raise ValueError("Training steps differ from the source run")
    schedule_steps = min(schedule_steps, args.max_train_steps or schedule_steps)
    if args.warmup_steps >= schedule_steps and args.warmup_steps:
        raise ValueError("Warmup must be shorter than the LR schedule")
    learning_rate = make_learning_rate_schedule(
        args.learning_rate, args.warmup_steps, schedule_steps, args.lr_schedule
    )

    if completed_steps and callable(learning_rate):
        original_schedule = learning_rate
        learning_rate = lambda step: original_schedule(step + completed_steps)
    if source_config is not None:
        print(f"Continuing after epoch {args.completed_epochs}/{args.epochs}, step {completed_steps}/{total_steps}.")
        print("Epoch labels below count epochs in this continuation; shuffle seeds and LR retain the original offset.")

    run_config.update(
        n_trainable_params=trainable_params,
        n_lora_params=lora_params,
        split_counts={split: len(rows) for split, rows in data_module._intervals.items()},
        microbatches_per_epoch=microbatches_per_epoch,
        schedule_steps=schedule_steps,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        completed_steps=completed_steps,
        remaining_epochs=args.epochs - args.completed_epochs,
        remaining_steps=total_steps - completed_steps,
    )
    (checkpoint_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    loss_fns = {args.head_name: make_track_loss(
        track_means, apply_squashing=args.output_type == "rna_seq",
        **{key: run_config["loss"][key] for key in (
            "num_segments", "positional_weight", "count_weight")},
    )}
    train(
        model,
        data_module,
        head_specs,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs - args.completed_epochs,
        seed=args.seed + args.completed_epochs,
        max_train_steps=args.max_train_steps,
        max_grad_norm=args.max_grad_norm,
        heads_only=True,
        checkpoint_dir=checkpoint_dir,
        organism="HOMO_SAPIENS",
        best_metric="valid_loss",
        best_metric_mode="min",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        verbose=True,
        num_devices=args.num_devices,
        train_lora=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_fns=loss_fns,
    )
    if args.output_type != "procap":
        return 0
    # Evaluate held-out data with the validation-selected checkpoint, not the
    # final epoch weights. Registration above preserves target order and means.
    if not (checkpoint_dir / "best").exists():
        raise RuntimeError("No finite validation checkpoint was saved; cannot evaluate best model")
    del model
    model = load_checkpoint(
        checkpoint_dir / "best", base_model_version=args.model_version,
        base_checkpoint_path=args.base_checkpoint,
        organism_settings=model_kwargs["organism_settings"],
    )
    evaluator = create_track_evaluator(model, loss_fns, num_devices=args.num_devices)
    for split in ("valid", "test"):
        if not data_module._intervals.get(split):
            continue
        metrics = evaluator(data_module, split)
        report = {"checkpoint": "best", "split": split,
                  "targets": labels, "loss_config": run_config["loss"],
                  "heads": metrics}
        (checkpoint_dir / f"best_{split}_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"Best checkpoint {split} metrics: {json.dumps(metrics)}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    args.fasta = args.fasta.expanduser().resolve()
    args.split_dir = args.split_dir.expanduser().resolve() if args.split_dir else None
    args.base_checkpoint = (
        args.base_checkpoint.expanduser().resolve() if args.base_checkpoint else None
    )
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.preflight:
        return print_preflight(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
