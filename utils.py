"""
Shared utilities: reproducibility, checkpointing, parameter counting,
and inference-latency benchmarking (needed for the "total parameters /
evaluation time latency" reporting requirement).
"""
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # assuming float32
    return {"total_params": total, "trainable_params": trainable, "size_mb": round(size_mb, 3)}


@torch.no_grad()
def measure_latency(model: torch.nn.Module, input_shape: tuple, device: str = "cpu",
                     n_warmup: int = 10, n_runs: int = 50) -> dict:
    """
    Measures per-sample inference latency (batch size = 1) for on-device
    deployment estimates.

    input_shape: (C, T) feature shape, e.g. (40, 301) for 3s MFCC.
    """
    model.eval().to(device)
    dummy = torch.randn(1, *input_shape, device=device)

    for _ in range(n_warmup):
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    times = np.array(times)
    return {
        "device": device,
        "mean_ms": round(float(times.mean()), 3),
        "std_ms": round(float(times.std()), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "n_runs": n_runs,
    }


def save_checkpoint(model, path: str, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)  # raw state_dict (old-style save)
    return model, ckpt


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
