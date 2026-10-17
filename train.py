"""
Train any registered LID architecture.

Usage:
    python train.py --model ecapa_tdnn --train_dir data/train --val_dir data/val
    python train.py --model rawnet     --train_dir data/train --val_dir data/val
    python train.py --model streaming_lid --train_dir data/train --val_dir data/val

Run `python -c "from models import MODEL_REGISTRY; print(list(MODEL_REGISTRY))"`
to see all available --model choices.

Outputs (under --checkpoint_dir, default checkpoints/<model>/):
    best_<model>.pth      - best weights + label map + normalizer + metadata
    label_map.json        - language name -> index mapping
Outputs (under --report_dir, default reports/<model>/):
    training_log.csv       - per-epoch loss/accuracy (for convergence plots)
    model_size.json         - parameter count / size report
"""
import argparse
import os

import config
from engine import build_dataloaders, train_single_model
from models import MODEL_REGISTRY, get_model, feature_type_for
from utils import set_seed, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Train a registered LID architecture")
    p.add_argument("--model", type=str, default="ecapa_tdnn", choices=list(MODEL_REGISTRY))
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Defaults to checkpoints/<model>/")
    p.add_argument("--report_dir", type=str, default=None,
                   help="Defaults to reports/<model>/")
    p.add_argument("--device", type=str, default=config.DEVICE)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    checkpoint_dir = args.checkpoint_dir or os.path.join(config.CHECKPOINT_DIR, args.model)
    report_dir = args.report_dir or os.path.join(config.REPORT_DIR, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    feature_type = feature_type_for(args.model)
    print(f"Model: {args.model} | feature_type: {feature_type} | device: {args.device}")

    train_dataset, val_dataset, train_loader, val_loader, normalizer = build_dataloaders(
        args.train_dir, args.val_dir, feature_type, args.batch_size, args.num_workers)
    num_languages = len(train_dataset.label_to_idx)
    print(f"Languages ({num_languages}): {list(train_dataset.label_to_idx.keys())}")

    input_channels = config.N_MFCC if feature_type == "mfcc" else (
        config.N_MELS if feature_type == "melspec" else None)
    model = get_model(args.model, num_languages=num_languages, input_channels=input_channels)

    train_dataset.save_label_map(os.path.join(checkpoint_dir, config.LABEL_MAP_NAME))

    extra_fields = {
        "model_name": args.model,
        "label_to_idx": train_dataset.label_to_idx,
        "feature_type": feature_type,
        "input_channels": input_channels,
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"best_{args.model}.pth")
    log_csv_path = os.path.join(report_dir, "training_log.csv")

    best_val_loss, best_val_acc, param_info = train_single_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience, device=args.device,
        checkpoint_path=checkpoint_path, log_csv_path=log_csv_path,
        extra_checkpoint_fields=extra_fields,
    )

    save_json(param_info, os.path.join(report_dir, "model_size.json"))
    print(f"Training complete. Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Logs: {log_csv_path}")


if __name__ == "__main__":
    main()
