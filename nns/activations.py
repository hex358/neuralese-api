import torch, torch.nn.functional as F
from typing import Dict, Any

_ACT = {
	"relu": torch.relu,
	"gelu": F.gelu,
	"sigmoid": torch.sigmoid,
	"tanh": torch.tanh,
	"none": lambda x: x,
}

def apply_act(x: torch.Tensor, props: Dict[str, Any], default: str = "none") -> torch.Tensor:
	cfg = props.get("config", {}) or {}
	name = str(cfg.get("activation", default)).lower()
	fn = _ACT.get(name, _ACT["none"])
	return fn(x)
