"""
Benchmark and/or train every registered LID architecture, and produce the
side-by-side comparison the work-let's Expectations slide asks for:
"Detailed report on performance comparison, total parameters, evaluation
time latency, training loss convergence."

Two modes:

  --mode bench (default, fast, no data required)
      Builds each architecture fresh and reports parameter count, model
      size, and CPU inference latency. Useful for an early architecture
      sanity check before committing to full training runs.

  --mode train (needs --train_dir/--val_dir/--test_dir)
      Trains each architecture end-to-end (via engine.train_single_model),
      evaluates it on the held-out test set, and adds val/test accuracy to
      the comparison alongside params/latency. Each model gets its own
      checkpoint + training_log.csv under checkpoints/<model>/ and
      reports/<model>/, in addition to the combined comparison table.

Usage:
    python compare.py --mode bench
    python compare.py --mode train --train_dir data/train --val_dir data/val \
        --test_dir data/test --epochs 20

Outputs:
    reports/comparison_table.csv
    reports/comparison_table.md
    reports/comparison_report.json
"""
import argparse
import json
import os

import config
from engine import build_dataloaders, train_single_model
from evaluate import evaluate, load_model_from_checkpoint
from models import MODEL_REGISTRY, get_model, feature_type_for
from utils import count_parameters, measure_latency, save_json, set_seed

from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Compare all registered LID architectures")
    p.add_argument("--mode", choices=["bench", "train"], default="bench")
    p.add_argument("--train_dir", type=str, default=None)
    p.add_argument("--val_dir", type=str, default=None)
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--models", type=str, nargs="*", default=list(MODEL_REGISTRY),
                   help="Subset of registered models to compare (default: all)")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--report_dir", type=str, default=config.REPORT_DIR)
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--device", type=str, default=config.DEVICE)
    return p.parse_args()


def bench_one(name, device):
    """Params + latency only, on a freshly-initialized (untrained) model."""
    feature_type = feature_type_for(name)
    input_channels = config.N_MFCC if feature_type == "mfcc" else None
    model = get_model(name, num_languages=10, input_channels=input_channels)  # dummy class count

    param_info = count_parameters(model)
    if feature_type == "raw":
        input_shape = (config.SEGMENT_SAMPLES,)
    else:
        seg_frames = config.SEGMENT_SAMPLES // config.HOP_LENGTH + 1
        input_shape = (input_channels, seg_frames)
    latency = measure_latency(model, input_shape, device=device)

    return {
        "model": name,
        "description": MODEL_REGISTRY[name]["description"],
        "feature_type": feature_type,
        "total_params": param_info["total_params"],
        "size_mb": param_info["size_mb"],
        "latency_mean_ms": latency["mean_ms"],
        "latency_p95_ms": latency["p95_ms"],
    }


def train_and_eval_one(name, args):
    feature_type = feature_type_for(name)
    print(f"\n=== {name} ({feature_type}) ===")

    train_dataset, val_dataset, train_loader, val_loader, normalizer = build_dataloaders(
        args.train_dir, args.val_dir, feature_type, args.batch_size, args.num_workers)
    num_languages = len(train_dataset.label_to_idx)

    input_channels = config.N_MFCC if feature_type == "mfcc" else (
        config.N_MELS if feature_type == "melspec" else None)
    model = get_model(name, num_languages=num_languages, input_channels=input_channels)

    model_ckpt_dir = os.path.join(args.checkpoint_dir, name)
    model_report_dir = os.path.join(args.report_dir, name)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(model_report_dir, exist_ok=True)
    train_dataset.save_label_map(os.path.join(model_ckpt_dir, config.LABEL_MAP_NAME))

    extra_fields = {
        "model_name": name,
        "label_to_idx": train_dataset.label_to_idx,
        "feature_type": feature_type,
        "input_channels": input_channels,
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
    }
    checkpoint_path = os.path.join(model_ckpt_dir, f"best_{name}.pth")
    log_csv_path = os.path.join(model_report_dir, "training_log.csv")

    best_val_loss, best_val_acc, param_info = train_single_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device,
        checkpoint_path=checkpoint_path, log_csv_path=log_csv_path,
        extra_checkpoint_fields=extra_fields,
    )

    test_acc, latency = None, None
    if args.test_dir:
        eval_model, label_to_idx, ft, ic, eval_normalizer, ckpt = load_model_from_checkpoint(
            checkpoint_path, args.device)
        idx_to_label = {i: l for l, i in label_to_idx.items()}
        test_dataset = __import__("data.dataset", fromlist=["LIDDataset"]).LIDDataset(
            args.test_dir, feature_type=ft, transform=eval_normalizer, label_to_idx=label_to_idx)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        metrics = evaluate(eval_model, test_loader, args.device, idx_to_label)
        test_acc = metrics["accuracy"]

        if ft == "raw":
            input_shape = (config.SEGMENT_SAMPLES,)
        else:
            seg_frames = config.SEGMENT_SAMPLES // config.HOP_LENGTH + 1
            input_shape = (ic, seg_frames)
        latency = measure_latency(eval_model, input_shape, device="cpu")

    return {
        "model": name,
        "description": MODEL_REGISTRY[name]["description"],
        "feature_type": feature_type,
        "total_params": param_info["total_params"],
        "size_mb": param_info["size_mb"],
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "latency_mean_ms": latency["mean_ms"] if latency else None,
        "latency_p95_ms": latency["p95_ms"] if latency else None,
        "checkpoint": checkpoint_path,
        "training_log": log_csv_path,
    }


def write_markdown_table(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    # keep the table readable: drop long path columns from the markdown view
    display_cols = [c for c in cols if c not in ("checkpoint", "training_log", "description")]
    lines = ["| " + " | ".join(display_cols) + " |",
             "|" + "|".join(["---"] * len(display_cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in display_cols) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    set_seed(config.SEED)
    os.makedirs(args.report_dir, exist_ok=True)

    if args.mode == "train" and not (args.train_dir and args.val_dir):
        raise ValueError("--mode train requires --train_dir and --val_dir")

    results = []
    for name in args.models:
        if args.mode == "bench":
            print(f"Benchmarking {name}...")
            results.append(bench_one(name, args.device))
        else:
            results.append(train_and_eval_one(name, args))

    csv_path = os.path.join(args.report_dir, "comparison_table.csv")
    md_path = os.path.join(args.report_dir, "comparison_table.md")
    json_path = os.path.join(args.report_dir, "comparison_report.json")

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    write_markdown_table(results, md_path)
    save_json({"mode": args.mode, "results": results}, json_path)

    print("\n=== Comparison summary ===")
    for r in results:
        acc_str = f", val_acc={r['best_val_acc']:.2f}%" if "best_val_acc" in r else ""
        test_str = f", test_acc={r['test_acc']*100:.2f}%" if r.get("test_acc") is not None else ""
        latency_str = f"{r['latency_mean_ms']:.2f}ms" if r.get("latency_mean_ms") is not None else "n/a"
        print(f"{r['model']:15s} params={r['total_params']:>10,} "
              f"size={r['size_mb']:>7.2f}MB latency={latency_str:>9s}{acc_str}{test_str}")

    print(f"\nSaved: {csv_path}, {md_path}, {json_path}")


if __name__ == "__main__":
    main()
