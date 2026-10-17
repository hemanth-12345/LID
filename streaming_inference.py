"""
Demonstrate streaming (chunk-by-chunk) inference with the novel
StreamingLID architecture, and compare it against full-sequence inference
on the same audio -- this is the practical check for the work-let's
closure-phase "streaming support" requirement.

Usage:
    python streaming_inference.py --audio path/to/clip.wav \
        --checkpoint checkpoints/streaming_lid/best_streaming_lid.pth --chunks 6
"""
import argparse
import time

import torch
import torch.nn.functional as F

import config
from data.dataset import load_and_preprocess_audio, extract_features
from evaluate import load_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Streaming vs full-sequence LID inference")
    p.add_argument("--audio", type=str, required=True)
    p.add_argument("--checkpoint", type=str,
                   default=f"{config.CHECKPOINT_DIR}/streaming_lid/best_streaming_lid.pth")
    p.add_argument("--chunks", type=int, default=6, help="Number of chunks to split the 3s clip into")
    p.add_argument("--device", type=str, default=config.DEVICE)
    return p.parse_args()


@torch.no_grad()
def run(audio_path, checkpoint_path, num_chunks, device):
    model, label_to_idx, feature_type, _, normalizer, _ = load_model_from_checkpoint(checkpoint_path, device)
    idx_to_label = {i: l for l, i in label_to_idx.items()}

    waveform = load_and_preprocess_audio(audio_path)
    features = extract_features(waveform, feature_type)
    if normalizer is not None:
        features = normalizer(features)
    features = features.unsqueeze(0).to(device)  # (1, C, T)

    # ---- full-sequence inference ----
    t0 = time.perf_counter()
    full_logits = model(features)
    full_ms = (time.perf_counter() - t0) * 1000
    full_probs = F.softmax(full_logits, dim=1).squeeze(0).cpu()

    # ---- chunked streaming inference ----
    chunks = torch.chunk(features, num_chunks, dim=2)
    state = model.init_stream(batch_size=1, device=device)
    per_chunk_ms = []
    stream_logits = None
    for chunk in chunks:
        t0 = time.perf_counter()
        stream_logits, state = model.forward_chunk(chunk, state)
        per_chunk_ms.append((time.perf_counter() - t0) * 1000)
    stream_probs = F.softmax(stream_logits, dim=1).squeeze(0).cpu()

    return idx_to_label, full_probs, full_ms, stream_probs, per_chunk_ms


def main():
    args = parse_args()
    idx_to_label, full_probs, full_ms, stream_probs, per_chunk_ms = run(
        args.audio, args.checkpoint, args.chunks, args.device)

    print(f"Full-sequence inference: {full_ms:.2f} ms (whole 3s clip at once)")
    print("Full-sequence predictions:")
    for i, p in sorted(enumerate(full_probs.tolist()), key=lambda x: -x[1])[:3]:
        print(f"  {idx_to_label[i]:15s} {p*100:5.2f}%")

    print(f"\nStreaming inference: {sum(per_chunk_ms):.2f} ms total across "
          f"{len(per_chunk_ms)} chunks (avg {sum(per_chunk_ms)/len(per_chunk_ms):.2f} ms/chunk)")
    print("Streaming (final, after all chunks) predictions:")
    for i, p in sorted(enumerate(stream_probs.tolist()), key=lambda x: -x[1])[:3]:
        print(f"  {idx_to_label[i]:15s} {p*100:5.2f}%")

    max_diff = (full_probs - stream_probs).abs().max().item()
    print(f"\nMax probability difference (streaming vs full): {max_diff:.4f}")
    print("(Non-zero due to causal-conv chunk-boundary effects -- see StreamingLID docstring.)")


if __name__ == "__main__":
    main()
