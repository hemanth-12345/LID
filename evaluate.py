"""
Evaluate a trained checkpoint from any registered architecture.

Supports both:
  - Same-corpora evaluation (test split from the corpus the model trained on)
  - Cross-corpora evaluation (a different corpus, e.g. train on VoxLingua107,
    evaluate on Mozilla Common Voice or Fleurs), using the label map saved
    at training time so language indices line up across corpora.

Usage:
    python evaluate.py --test_dir data/test \
        --checkpoint checkpoints/ecapa_tdnn/best_ecapa_tdnn.pth --tag same_corpora

    python evaluate.py --test_dir /path/to/other_corpus/test \
        --checkpoint checkpoints/ecapa_tdnn/best_ecapa_tdnn.pth --tag cross_corpora_fleurs

Outputs reports/<tag>_report.json with accuracy, per-class precision/
recall/F1, a confusion matrix, parameter count, and inference latency.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config
from data.dataset import LIDDataset, FeatureNormalize
from models import get_model
from utils import count_parameters, measure_latency, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained LID checkpoint")
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--report_dir", type=str, default=config.REPORT_DIR)
    p.add_argument("--tag", type=str, default="eval", help="Name used for the output report file")
    p.add_argument("--device", type=str, default=config.DEVICE)
    return p.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name = ckpt["model_name"]
    label_to_idx = ckpt["label_to_idx"]
    feature_type = ckpt["feature_type"]
    input_channels = ckpt["input_channels"]

    model = get_model(model_name, num_languages=len(label_to_idx), input_channels=input_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    normalizer = None
    if ckpt.get("normalizer") is not None:
        normalizer = FeatureNormalize()
        normalizer.load_state_dict(ckpt["normalizer"])

    return model, label_to_idx, feature_type, input_channels, normalizer, ckpt


@torch.no_grad()
def evaluate(model, loader, device, idx_to_label):
    all_preds, all_labels = [], []
    for features, labels in loader:
        features = features.to(device)
        outputs = model(features)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=sorted(idx_to_label)).tolist()

    return {"accuracy": acc, "classification_report": report,
            "confusion_matrix": cm, "labels_order": target_names}


def main():
    args = parse_args()
    print(f"Device: {args.device}")

    model, label_to_idx, feature_type, input_channels, normalizer, ckpt = \
        load_model_from_checkpoint(args.checkpoint, args.device)
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    model_name = ckpt["model_name"]

    test_dataset = LIDDataset(args.test_dir, feature_type=feature_type,
                               transform=normalizer, label_to_idx=label_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    print(f"Model: {model_name} | Evaluating on {len(test_dataset)} files from {args.test_dir}")
    metrics = evaluate(model, test_loader, args.device, idx_to_label)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")

    param_info = count_parameters(model)
    if feature_type == "raw":
        input_shape = (config.SEGMENT_SAMPLES,)
    else:
        seg_frames = config.SEGMENT_SAMPLES // config.HOP_LENGTH + 1
        input_shape = (input_channels, seg_frames)
    latency_cpu = measure_latency(model, input_shape, device="cpu")

    report = {
        "tag": args.tag,
        "model_name": model_name,
        "test_dir": args.test_dir,
        "checkpoint": args.checkpoint,
        "num_test_files": len(test_dataset),
        "trained_val_acc": ckpt.get("val_acc"),
        **metrics,
        "param_info": param_info,
        "latency": latency_cpu,
    }

    os.makedirs(args.report_dir, exist_ok=True)
    out_path = os.path.join(args.report_dir, f"{args.tag}_report.json")
    save_json(report, out_path)
    print(f"Report saved to {out_path}")
    print(f"Params: {param_info['total_params']:,} | "
          f"Size: {param_info['size_mb']} MB | "
          f"CPU latency (mean): {latency_cpu['mean_ms']} ms/sample")


if __name__ == "__main__":
    main()
