# Beyond ECAPA-TDNN: On-Device LID

Full implementation for the work-let *"Developing efficient deep learning
architectures beyond ECAPA-TDNN for low-latency on-device spoken language
identification (LID)"* — baseline, the recent architectures benchmarked
against it, a novel streamable architecture, and the comparison tooling
to report on all of them.

## Project structure

```
lid_project/
├── config.py                # all hyperparameters/paths in one place
├── data/
│   └── dataset.py            # audio loading, MFCC/mel/raw features, Dataset, normalization
├── models/
│   ├── __init__.py            # MODEL_REGISTRY + get_model() -- pick any architecture by name
│   ├── ecapa_lid.py            # MS1 baseline: dilated TDNN + SE + attentive stats pooling
│   ├── ecapa2.py                # MS2: depthwise-separable macro/micro blocks (efficiency)
│   ├── aasist.py                 # MS2: spectral/temporal graph-attention + master-node fusion
│   ├── rawnet.py                  # MS2: learnable sinc front end on raw waveform + GRU
│   └── streaming_lid.py            # Closure: novel streamable architecture (see below)
├── engine.py                 # shared dataloader-building + train/eval loop
├── utils.py                  # seeding, checkpointing, param count, latency benchmark
├── train.py                    # train any one registered model
├── evaluate.py                   # same-corpora & cross-corpora evaluation, metrics report
├── compare.py                      # bench or train+eval ALL registered models, comparison report
├── inference.py                     # single-file prediction
├── streaming_inference.py             # chunked streaming vs full-sequence inference demo
├── requirements.txt
├── checkpoints/                       # saved weights (created at train time, one subfolder/model)
└── reports/                             # training logs, evaluation & comparison reports
```

## Architectures implemented

| Model key       | Idea                                                              | Feature input |
|-----------------|--------------------------------------------------------------------|----------------|
| `ecapa_tdnn`    | MS1 baseline: multiscale dilated TDNN, SE, attentive stats pooling | MFCC           |
| `ecapa2`        | Depthwise-separable "macro/micro" blocks — fewer params, same idea | MFCC           |
| `aasist`        | Spectral + temporal graphs, attention message passing, master node fusion | MFCC     |
| `rawnet`        | Learnable sinc filterbank directly on the waveform + GRU            | raw waveform   |
| `streaming_lid` | **Novel**: causal depthwise-separable TDNN + incremental attentive pooling — the only one with a working chunked-inference API | MFCC |

Run `python -c "from models import MODEL_REGISTRY; print(list(MODEL_REGISTRY))"`
to see this list from code. Every model shares one interface —
`model(x) -> logits`, optional `return_embedding=True` — so `train.py`,
`evaluate.py`, and `compare.py` treat them interchangeably by name.

These are compact, from-scratch reimplementations of the *ideas* behind
each published architecture (not verbatim ports of any reference codebase),
sized down for on-device experimentation — see each model file's docstring
for what was simplified and why.

## Data layout

One folder per language, audio files inside (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`):

```
data/train/english/*.wav
data/train/hindi/*.wav
data/val/...
data/test/...
```

For cross-corpora evaluation, point `--test_dir` at a differently-sourced
corpus (e.g. train on VoxLingua107, evaluate on Common Voice or Fleurs)
using the same language folder names — `evaluate.py` reuses the label map
saved during training so indices stay consistent across corpora.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train one architecture:**
```bash
python train.py --model ecapa_tdnn --train_dir data/train --val_dir data/val
python train.py --model rawnet     --train_dir data/train --val_dir data/val
python train.py --model streaming_lid --train_dir data/train --val_dir data/val --epochs 40
```
Saves `checkpoints/<model>/best_<model>.pth` (weights + label map +
normalizer + metadata bundled together) and
`reports/<model>/training_log.csv` (per-epoch loss/accuracy, for
convergence plots).

**Evaluate (same-corpora or cross-corpora):**
```bash
python evaluate.py --test_dir data/test --checkpoint checkpoints/ecapa_tdnn/best_ecapa_tdnn.pth --tag same_corpora
python evaluate.py --test_dir /path/to/other_corpus/test --checkpoint checkpoints/ecapa_tdnn/best_ecapa_tdnn.pth --tag cross_corpora_fleurs
```
Produces `reports/<tag>_report.json`: accuracy, per-language
precision/recall/F1, confusion matrix, parameter count, model size, CPU
inference latency (mean/std/p95 ms per sample).

**Compare every architecture at once (this is the Expectations-slide report):**
```bash
# quick: params + latency only, no training needed
python compare.py --mode bench

# full: train + evaluate every architecture, produce a combined table
python compare.py --mode train --train_dir data/train --val_dir data/val --test_dir data/test --epochs 30
```
Outputs `reports/comparison_table.csv`, `reports/comparison_table.md`, and
`reports/comparison_report.json` — model, description, params, size,
latency, val/test accuracy, side by side.

**Inference on one file:**
```bash
python inference.py --audio path/to/clip.wav --checkpoint checkpoints/ecapa_tdnn/best_ecapa_tdnn.pth --topk 3
```

**Streaming inference (the closure-phase deliverable):**
```bash
python streaming_inference.py --audio path/to/clip.wav --checkpoint checkpoints/streaming_lid/best_streaming_lid.pth --chunks 6
```
Splits the clip into chunks, feeds them one at a time through
`model.forward_chunk`, and compares the result to ordinary full-sequence
inference — they agree almost exactly (residual difference is from
conv chunk-boundary effects, documented in `models/streaming_lid.py`).

## What "streamable" means here, precisely

`StreamingLID` is the only architecture with a chunked-inference API
(`init_stream` / `forward_chunk`), enabled by two design choices:

1. **Causal convolutions** — every conv only looks left (past frames), via
   left-only padding, so no future frame is ever needed to produce the
   current output.
2. **Incremental attentive statistics pooling** — the pooling module tracks
   three running sums (`sum_w`, `sum_wx`, `sum_wx2`) across chunks instead
   of needing the whole utterance buffered; the final mean/std is
   mathematically identical to running standard attentive pooling once
   over the full concatenated sequence (see `StreamingStatsPool`
   docstring for the derivation), not an approximation.

**Known gap, honestly documented, not hidden:** `forward_chunk` currently
recomputes each causal conv over just the current chunk — it doesn't yet
cache each block's last `dilation × (kernel_size − 1)` frames to carry into
the next chunk. That's why streaming and full-sequence predictions differ
by a small amount (~0.001–0.003 in the smoke test) rather than being
bit-identical. Closing that gap is a small, well-scoped next step: add a
`state` buffer per `CausalDilatedBlock` alongside the pooling state.

## Suggested next steps for the work-let

- **MS2 wrap-up:** run `compare.py --mode train` on the real VoxLingua107 /
  Common Voice / Fleurs data to get the actual performance/parameter/
  latency table for the report and publication draft.
- **Closure:** add the conv-state caching described above, then re-run
  `streaming_inference.py` and confirm streaming/full-sequence predictions
  converge to bit-identical (modulo floating-point order-of-operations).
- Consider quantization (int8) and ONNX export for the final on-device
  latency numbers, since the current benchmark is plain PyTorch on CPU.
