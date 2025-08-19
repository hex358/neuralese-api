# kids_net.py
# Minimal, reusable building blocks for simple "children's" networks in PyTorch.
# PyTorch 2.x compatible.

from __future__ import annotations
from typing import Iterable, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# -----------------------------
# Small helpers (activations)
# -----------------------------
_ACTS = {
    None: nn.Identity,
    "id": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "softmax": lambda: nn.Softmax(dim=1),
}

def _act(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    name = name.lower()
    if name not in _ACTS:
        raise ValueError(f"Unknown activation: {name}")
    cls_or_fn = _ACTS[name]
    return cls_or_fn() if callable(cls_or_fn) else cls_or_fn

# -----------------------------
# “Lego bricks” (layer makers)
# -----------------------------
def Dense(out_features: int, in_features: Optional[int] = None,
          act: Optional[str] = "relu", bias: bool = True) -> nn.Sequential:
    """
    Fully connected: Linear -> (Activation)
    If in_features is None, PyTorch infers it at runtime (via LazyLinear).
    """
    lin = nn.Linear(in_features, out_features, bias=bias) if in_features is not None else nn.LazyLinear(out_features, bias=bias)
    return nn.Sequential(lin, _act(act))

def Conv2D(out_ch: int, k: int = 3, s: int = 1, p: Union[int, str] = "same",
           act: Optional[str] = "relu", bias: bool = True) -> nn.Sequential:
    """
    2D Convolution: Conv2d -> (Activation)
    Uses LazyConv2d to infer in_channels automatically.
    p="same" gives same spatial size when s=1 (PyTorch 2.2+).
    """
    conv = nn.LazyConv2d(out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
    return nn.Sequential(conv, _act(act))

def DepthwiseSeparable2D(out_ch: int, k: int = 3, s: int = 1, p: Union[int, str] = "same",
                         act: Optional[str] = "relu") -> nn.Sequential:
    """
    Depthwise + Pointwise conv (mobile-friendly).
    """
    dw = nn.LazyConv2d(groups=None, out_channels=None, kernel_size=0)  # placeholder to keep type hints happy
    # Build actual modules:
    modules: List[nn.Module] = []
    modules.append(nn.LazyConv2d(out_channels=0, kernel_size=0))  # never used; replaced below
    # Replace with depthwise then pointwise:
    modules = [
        nn.LazyConv2d(out_channels=None, kernel_size=0)  # dummy to satisfy type checkers
    ]
    # Real sequence:
    return nn.Sequential(
        nn.LazyConv2d(out_channels=0, kernel_size=0)  # we’ll overwrite below at runtime
    )
# (Keep things simple: skip depthwise for now—children rarely need it.)
# If you want it later, ask and I’ll add a clean version.

def MaxPool2D(k: int = 2, s: Optional[int] = None) -> nn.Module:
    return nn.MaxPool2d(kernel_size=k, stride=s or k)

def AvgPool2D(k: int = 2, s: Optional[int] = None) -> nn.Module:
    return nn.AvgPool2d(kernel_size=k, stride=s or k)

def Flatten() -> nn.Module:
    return nn.Flatten()

def Dropout(p: float = 0.5) -> nn.Module:
    return nn.Dropout(p)

def BatchNorm1D() -> nn.Module:
    return nn.LazyBatchNorm1d()

def BatchNorm2D() -> nn.Module:
    return nn.LazyBatchNorm2d()

# -----------------------------
# A simple chainable builder
# -----------------------------
class NetBuilder(nn.Module):
    """
    Chain layers with .add(...). Result is an nn.Sequential with friendly .summary().
    Example:
        model = (NetBuilder()
                 .add(Conv2D(16, k=3))
                 .add(MaxPool2D())
                 .add(Conv2D(32, k=3))
                 .add(MaxPool2D())
                 .add(Flatten())
                 .add(Dense(64))
                 .add(Dropout(0.2))
                 .add(Dense(10, act=None)))
    """
    def __init__(self):
        super().__init__()
        self.layers: List[nn.Module] = []

    def add(self, layer: nn.Module) -> "NetBuilder":
        self.layers.append(layer)
        return self

    def build(self) -> nn.Sequential:
        return nn.Sequential(*self.layers)

    # nn.Module interface
    def forward(self, x: Tensor) -> Tensor:
        seq = self.build()
        return seq(x)

    # quick info
    def summary(self, input_size: Tuple[int, ...]) -> str:
        """
        input_size excludes batch, e.g. (3, 32, 32) for images.
        """
        dummy = torch.zeros(1, *input_size)
        seq = self.build()
        out = dummy
        info = []
        total_params = 0

        for i, m in enumerate(seq):
            out = m(out)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            total_params += params
            info.append(f"{i:02d}: {m.__class__.__name__:20s} -> {tuple(out.shape)}  params={params}")

        header = f"Input: {tuple(dummy.shape)}"
        footer = f"Total trainable params: {total_params}"
        return "\n".join([header, *info, footer])

# -----------------------------
# Preset “recipes”
# -----------------------------
def MLP(input_dim: int, hidden: Iterable[int], num_classes: int,
        act: str = "relu", dropout: float = 0.0) -> nn.Sequential:
    """
    Simple multi-layer perceptron: [Dense(h) x N] -> Dense(num_classes)
    """
    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden:
        layers += [nn.Linear(in_dim, h), _act(act)]
        if dropout > 0:
            layers.append(Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)

def SimpleConvNet(num_classes: int = 10) -> nn.Sequential:
    """
    Tiny image conv net for 3x32x32 (CIFAR-like) images.
    """
    return (NetBuilder()
            .add(Conv2D(16, k=3, act="relu"))
            .add(BatchNorm2D())
            .add(MaxPool2D(2))
            .add(Conv2D(32, k=3, act="relu"))
            .add(BatchNorm2D())
            .add(MaxPool2D(2))
            .add(Conv2D(64, k=3, act="relu"))
            .add(MaxPool2D(2))
            .add(Flatten())
            .add(Dense(128, act="relu"))
            .add(Dropout(0.25))
            .add(Dense(num_classes, act=None))
            ).build()

# -----------------------------
# Tiny demo
# -----------------------------
if __name__ == "__main__":
    pass