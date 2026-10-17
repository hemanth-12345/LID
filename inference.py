"""
Run inference on a single audio file with a trained ECAPA-TDNN LID model.

Usage:
    python inference.py --audio path/to/clip.wav \
        --checkpoint checkpoints/best_ecapa_lid.pth --topk 3
"""
import argparse

import torch
import torch.nn.functional as F

import config
from data.dataset import load_and_preprocess_audio, extract_features
from evaluate import load_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="LID inference on a single audio file")
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=f"{config.CHECKPOINT_DIR}/{config.BEST_MODEL_NAME}")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--device", type=str, default=config.DEVICE)
    return p.parse_args()


@torch.no_grad()
def predict(audio_path, checkpoint_path, device, topk=3):
    model, label_to_idx, feature_type, _, normalizer, _ = load_model_from_checkpoint(checkpoint_path, device)
    idx_to_label = {i: l for l, i in label_to_idx.items()}

    waveform = load_and_preprocess_audio(audio_path)
    features = extract_features(waveform, feature_type)
    if normalizer is not None:
        features = normalizer(features)
    features = features.unsqueeze(0).to(device)  # (1, C, T)

    logits = model(features)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    top_probs, top_idxs = probs.topk(min(topk, len(idx_to_label)))
    return [(idx_to_label[i.item()], p.item()) for p, i in zip(top_probs, top_idxs)]


def main():
    args = parse_args()
    results = predict(args.audio, args.checkpoint, args.device, args.topk)
    print(f"Predictions for {args.audio}:")
    for lang, prob in results:
        print(f"  {lang:15s} {prob*100:5.2f}%")


if __name__ == "__main__":
    main()
