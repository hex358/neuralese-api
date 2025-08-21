import torch
from typing import Any, List
from .utils import to_tensor

def merge_inputs(vals: List[Any], device: torch.device) -> torch.Tensor:
	if not isinstance(vals, list):
		raise TypeError(f"merge_inputs expected list, got {type(vals).__name__}")
	if len(vals) == 0:
		raise ValueError("merge_inputs got empty list")

	def unpack(v):
		if isinstance(v, dict) and "tensor" in v:
			return v["tensor"].to(device, non_blocking=True)
		return to_tensor(v, device)

	if len(vals) == 1:
		return unpack(vals[0])

	base = unpack(vals[0]).clone()
	for v in vals[1:]:
		t = unpack(v)
		if t.shape != base.shape:
			raise ValueError(f"cannot sum inputs with shapes {base.shape} and {t.shape}")
		base.add_(t)
	return base
