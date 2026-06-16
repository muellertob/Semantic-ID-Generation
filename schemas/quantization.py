from torch import Tensor
from enum import Enum
from dataclasses import dataclass, field

class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2

class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2

@dataclass
class QuantizeOutput:
    embeddings: Tensor
    ids: Tensor
    loss: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)