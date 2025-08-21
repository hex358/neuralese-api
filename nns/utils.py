from __future__ import annotations
import os, json, torch
import torch.nn as nn
from typing import Dict, Any, Optional

def pick_device(ctx) -> torch.device:
	# cuda if possible else cpu
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stable_key(prefix: str, props: Dict[str, Any]) -> str:
	# stringify props for stable reuse
	try:
		body = json.dumps(props, sort_keys=True, default=str)
	except Exception:
		body = str(props)
	return f"{prefix}:{body}"

def _partial_load_linear_from_sd(dst: nn.Linear, sd: Dict[str, torch.Tensor], device: torch.device):
	with torch.no_grad():
		if "weight" in sd:
			w = sd["weight"].to(device)
			o = min(dst.weight.shape[0], w.shape[0])
			i = min(dst.weight.shape[1], w.shape[1])
			dst.weight[:o, :i].copy_(w[:o, :i])
		if dst.bias is not None and "bias" in sd:
			b = sd["bias"].to(device)
			o = min(dst.bias.shape[0], b.shape[0])
			dst.bias[:o].copy_(b[:o])

def _partial_load_conv2d_from_sd(dst: nn.Conv2d, sd: Dict[str, torch.Tensor], device: torch.device):
	with torch.no_grad():
		if "weight" in sd:
			w = sd["weight"].to(device)
			oc = min(dst.weight.shape[0], w.shape[0])
			ic = min(dst.weight.shape[1], w.shape[1])
			kh = min(dst.weight.shape[2], w.shape[2])
			kw = min(dst.weight.shape[3], w.shape[3])
			dst.weight[:oc, :ic, :kh, :kw].copy_(w[:oc, :ic, :kh, :kw])
		if dst.bias is not None and "bias" in sd:
			b = sd["bias"].to(device)
			oc = min(dst.bias.shape[0], b.shape[0])
			dst.bias[:oc].copy_(b[:oc])

def _partial_load_from_sd(dst: nn.Module, sd: Dict[str, torch.Tensor], device: torch.device):
	if isinstance(dst, nn.Linear):
		_partial_load_linear_from_sd(dst, sd, device)
	elif isinstance(dst, nn.Conv2d):
		_partial_load_conv2d_from_sd(dst, sd, device)

def count_params(m: Optional[nn.Module]) -> int:
	# num of trainable params
	if m is None:
		return 0
	return sum(p.numel() for p in m.parameters())

def pack_tensor(t: torch.Tensor, kind: str, m: Optional[nn.Module]) -> Dict[str, Any]:
	# standard pack for passing tensors around
	return {
		"tensor": t,
		"shape": tuple(t.shape),
		"dtype": str(t.dtype).replace("torch.", ""),
		"device": str(t.device),
		"layer_kind": kind,
		"params": count_params(m),
	}

def to_tensor(v: Any, device: torch.device) -> torch.Tensor:
	if isinstance(v, torch.Tensor):
		return v if v.device == device else v.to(device, non_blocking=True)
	t = torch.as_tensor(v, dtype=torch.float32)
	return t.to(device, non_blocking=True) if device.type == "cuda" else t


# --- extra helpers moved out of model_core ---
def layer_tag(props: Dict[str, Any], default: str) -> str:
	# cache_tag stabilizes identity, else fallback
	tag = props.get("cache_tag")
	return str(tag if tag is not None else default)

def inputs_all_from_kind(vals: list[Any], kind: str) -> bool:
	# check all dict entries have given layer_kind
	if not isinstance(vals, list) or not vals:
		return False
	for v in vals:
		if isinstance(v, dict):
			if v.get("layer_kind") != kind:
				return False
	return True

def transplant_linear(dst: nn.Linear, src: nn.Linear):
	# copy overlapping weights/bias
	with torch.no_grad():
		o = min(dst.out_features, src.out_features)
		i = min(dst.in_features, src.in_features)
		dst.weight[:o, :i].copy_(src.weight[:o, :i])
		if dst.bias is not None and src.bias is not None:
			dst.bias[:o].copy_(src.bias[:o])

def transplant_conv(dst: nn.Conv2d, src: nn.Conv2d):
	# copy overlapping conv2d weights if kernel matches
	with torch.no_grad():
		if dst.weight.shape[2:] != src.weight.shape[2:]:
			return
		oc = min(dst.out_channels, src.out_channels)
		ic = min(dst.in_channels, src.in_channels)
		kH, kW = dst.weight.shape[2], dst.weight.shape[3]
		dst.weight[:oc, :ic, :kH, :kW].copy_(src.weight[:oc, :ic, :kH, :kW])
		if dst.bias is not None and src.bias is not None:
			dst.bias[:oc].copy_(src.bias[:oc])
