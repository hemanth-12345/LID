"""
Audio preprocessing, feature extraction, and Dataset classes for LID.

Expected data layout (one folder per language, audio files inside):

    data/train/
        english/*.wav
        hindi/*.wav
        tamil/*.wav
        ...
    data/val/
        english/*.wav
        ...

Works for cross-corpora evaluation too: point a second LIDDataset at a
different root (e.g. a different corpus) as long as the language folder
names line up with the label map produced during training.
"""
import json
from pathlib import Path

import librosa
import torch
import torchaudio
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


# ----------------------------
# Preprocessing
# ----------------------------
def load_and_preprocess_audio(file_path: str) -> torch.Tensor:
    """Load, mono-convert, resample, peak-normalize, and fix-length an audio file.

    Uses librosa for decoding (mono conversion + resampling in one call),
    which avoids torchaudio's load/save codec-backend churn across versions.
    """
    waveform_np, _ = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)  # (1, T)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / (peak + 1e-8)

    num_samples = waveform.shape[1]
    if num_samples < config.SEGMENT_SAMPLES:
        padding = config.SEGMENT_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :config.SEGMENT_SAMPLES]

    return waveform.squeeze(0)  # (T,)


# ----------------------------
# Feature extraction
# ----------------------------
_mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=config.SAMPLE_RATE,
    n_mfcc=config.N_MFCC,
    melkwargs={"n_fft": config.N_FFT, "hop_length": config.HOP_LENGTH, "n_mels": config.N_MELS},
)
_melspec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
)
_amp_to_db = torchaudio.transforms.AmplitudeToDB()


def extract_features(waveform: torch.Tensor, feature_type: str = "mfcc") -> torch.Tensor:
    """Extract MFCC, log-Mel-spectrogram, or raw waveform from a waveform.

    Returns (C, T). For "raw", C=1 and T is the raw sample count -- used by
    waveform-native models (e.g. RawNet) instead of hand-crafted features.
    """
    if feature_type == "mfcc":
        return _mfcc_transform(waveform)
    elif feature_type == "melspec":
        return _amp_to_db(_melspec_transform(waveform))
    elif feature_type == "raw":
        return waveform.unsqueeze(0)  # (1, T)
    raise ValueError("feature_type must be 'mfcc', 'melspec', or 'raw'")


# ----------------------------
# Feature normalization
# ----------------------------
class FeatureNormalize:
    """Per-channel global mean/std normalization, fit once on the training set."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, dataloader):
        total_sum, total_sq_sum, total_count = 0.0, 0.0, 0
        for batch, _ in dataloader:
            total_sum += batch.sum(dim=(0, 2))
            total_sq_sum += (batch ** 2).sum(dim=(0, 2))
            total_count += batch.size(0) * batch.size(2)
        self.mean = total_sum / total_count
        self.std = torch.sqrt((total_sq_sum / total_count - self.mean ** 2).clamp(min=1e-8))
        return self

    def __call__(self, x):
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean[:, None]) / (self.std[:, None] + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, state):
        self.mean = torch.tensor(state["mean"])
        self.std = torch.tensor(state["std"])


# ----------------------------
# Dataset
# ----------------------------
class LIDDataset(Dataset):
    """
    Loads audio files organized in per-language subfolders.

    If `label_to_idx` is provided (e.g. from a saved label map), it is reused
    as-is instead of being rebuilt from the folders present. This is what
    makes cross-corpora evaluation consistent: the label indices from
    training carry over to a second, differently-composed corpus.
    """

    def __init__(self, data_dir: str, feature_type: str = "mfcc",
                 transform=None, label_to_idx: dict = None):
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.transform = transform
        self.file_paths = []
        self.labels = []

        available_langs = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        if label_to_idx is None:
            self.label_to_idx = {lang: i for i, lang in enumerate(available_langs)}
        else:
            self.label_to_idx = label_to_idx
            missing = set(available_langs) - set(self.label_to_idx)
            if missing:
                print(f"[LIDDataset] Warning: languages present on disk but absent "
                      f"from label map, skipping: {sorted(missing)}")

        self.idx_to_label = {i: lang for lang, i in self.label_to_idx.items()}

        for lang in available_langs:
            if lang not in self.label_to_idx:
                continue
            lang_dir = self.data_dir / lang
            for ext in AUDIO_EXTENSIONS:
                for file in lang_dir.glob(f"*{ext}"):
                    self.file_paths.append(str(file))
                    self.labels.append(self.label_to_idx[lang])

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No audio files found under {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform = load_and_preprocess_audio(file_path)
        features = extract_features(waveform, self.feature_type)

        if self.transform:
            features = self.transform(features)

        return features, label

    def save_label_map(self, path: str):
        with open(path, "w") as f:
            json.dump(self.label_to_idx, f, indent=2)

    @staticmethod
    def load_label_map(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
