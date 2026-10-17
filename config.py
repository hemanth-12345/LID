"""
Central configuration for the LID (Language Identification) project.
Import this everywhere instead of hard-coding constants.
"""
import torch

# ----------------------------
# Audio / preprocessing
# ----------------------------
SAMPLE_RATE = 16000
SEGMENT_LENGTH_SEC = 3
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_LENGTH_SEC  # 48,000 samples

N_MFCC = 40
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160

# ----------------------------
# Training
# ----------------------------
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 30
PATIENCE = 5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# ----------------------------
# Model
# ----------------------------
FEATURE_TYPE = "mfcc"          # "mfcc" or "melspec"
INPUT_CHANNELS = N_MFCC        # switch to N_MELS if FEATURE_TYPE == "melspec"
CHANNELS = 512
SE_REDUCTION = 8

# ----------------------------
# Paths
# ----------------------------
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_NAME = "best_ecapa_lid.pth"
LABEL_MAP_NAME = "label_map.json"
REPORT_DIR = "reports"

# ----------------------------
# Misc
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
