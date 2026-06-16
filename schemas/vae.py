from dataclasses import dataclass, field
from torch import Tensor

@dataclass
class VaeOutput:
    loss: Tensor
    reconstruction_loss: Tensor
    quantization_loss: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)
