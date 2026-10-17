"""
Central model registry. Every architecture in this project exposes the same
interface -- `model(x) -> logits (B, num_languages)`, with an optional
`return_embedding=True` -- so train.py / evaluate.py / compare.py can treat
them interchangeably by name.
"""
from models.ecapa_lid import ECAPA_TDNN_LID
from models.ecapa2 import ECAPA2LID
from models.aasist import AASISTLid
from models.rawnet import RawNetLID
from models.streaming_lid import StreamingLID

MODEL_REGISTRY = {
    "ecapa_tdnn": {
        "cls": ECAPA_TDNN_LID,
        "feature_type": "mfcc",
        "needs_input_channels": True,
        "description": "Baseline (MS1): dilated TDNN + SE + attentive stats pooling.",
    },
    "ecapa2": {
        "cls": ECAPA2LID,
        "feature_type": "mfcc",
        "needs_input_channels": True,
        "description": "MS2: depthwise-separable macro/micro blocks for parameter efficiency.",
    },
    "aasist": {
        "cls": AASISTLid,
        "feature_type": "mfcc",
        "needs_input_channels": True,
        "description": "MS2: spectral/temporal graph-attention with master-node fusion.",
    },
    "rawnet": {
        "cls": RawNetLID,
        "feature_type": "raw",
        "needs_input_channels": False,
        "description": "MS2: learnable sinc front end directly on raw waveform + GRU.",
    },
    "streaming_lid": {
        "cls": StreamingLID,
        "feature_type": "mfcc",
        "needs_input_channels": True,
        "description": "Closure: novel streamable architecture combining the above.",
    },
}


def get_model(name: str, num_languages: int, input_channels: int = None, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    cls = entry["cls"]
    if entry["needs_input_channels"]:
        return cls(num_languages=num_languages, input_channels=input_channels, **kwargs)
    return cls(num_languages=num_languages, **kwargs)


def feature_type_for(name: str) -> str:
    return MODEL_REGISTRY[name]["feature_type"]
