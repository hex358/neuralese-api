from __future__ import annotations
from typing import Dict, Any, Callable, Optional, List
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_core import node, Context


def _device_from_ctx(ctx: Context) -> torch.device:
	# Reads the preferred device from the context; falls back to CPU.
	# Keeps all tensors and modules on the same device so math works and is fast.
	try:
		dev = ctx.extra.get("device", "cpu")
	except Exception:
		dev = "cpu"
	return torch.device(dev)


def _stable_cache_key(prefix: str, props: Dict[str, Any]) -> str:
	# Creates a stable string key from any dict of layer settings.
	# This key is used to fetch/reuse the same PyTorch layer between calls.
	try:
		body = json.dumps(props, sort_keys=True, default=str)
	except Exception:
		body = str(props)
	return f"{prefix}:{body}"


def _param_count(module: Optional[nn.Module]) -> int:
	# Counts how many trainable numbers (parameters) a layer has.
	# Helpful for showing kids that dense layers have more parameters than pooling.
	if module is None:
		return 0
	return sum(p.numel() for p in module.parameters())


def _make_pack(t: torch.Tensor, kind: str, module: Optional[nn.Module]) -> Dict[str, Any]:
	# Wraps a tensor into a tiny, consistent “datapack”.
	# Each node passes datapacks so graphs stay uniform and easy to debug.
	return {
		"tensor": t,
		"shape": tuple(t.shape),
		"dtype": str(t.dtype).replace("torch.", ""),
		"device": str(t.device),
		"layer_kind": kind,
		"params": _param_count(module),
	}


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
	# Turns Python numbers/lists into tensors on the right device.
	# If it’s already a tensor, it is moved to the right device.
	if isinstance(value, torch.Tensor):
		return value.to(device)
	return torch.tensor(value, dtype=torch.float32, device=device)


def _merge_inputs(vals: List[Any], device: torch.device) -> torch.Tensor:
	# Accepts a list of datapacks or raw values from a port and combines them.
	# All inputs must share the exact same shape; they are added element-wise.
	# Returning a single input as-is is a fast path when the list length is 1.
	if not isinstance(vals, list):
		raise TypeError(f"_merge_inputs expected list, got {type(vals).__name__}")
	if len(vals) == 0:
		raise ValueError("No inputs provided on this port (empty list).")

	def unpack_one(v) -> torch.Tensor:
		if isinstance(v, dict) and "tensor" in v:
			return v["tensor"].to(device)
		return _to_tensor(v, device)

	tensors = [unpack_one(v) for v in vals]
	base = tensors[0]
	for t in tensors[1:]:
		if t.shape != base.shape:
			raise ValueError(f"Cannot sum inputs with shapes {base.shape} and {t.shape}.")
	if len(tensors) == 1:
		return base
	return torch.stack(tensors, dim=0).sum(dim=0)


_ACT = {
	"relu": torch.relu,
	"sigmoid": torch.sigmoid,
	"tanh": torch.tanh,
	"none": lambda x: x,
}


def _apply_activation(x: torch.Tensor, props: Dict[str, Any]) -> torch.Tensor:
	# Applies the requested activation by name (“relu”, “sigmoid”, etc.).
	# If none is requested, the tensor passes through unchanged.
	cfg = props.get("config", {})
	act = str(cfg.get("activation", "none")).lower()
	fn = _ACT.get(act, _ACT["none"])
	return fn(x)


def _get_modules(ctx: Context) -> Dict[str, nn.Module]:
	# Returns a global cache of built layers, keyed by their settings.
	# Reusing modules means parameters persist between graph runs (important for training).
	try:
		return ctx.extra.setdefault("_module_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_module_cache"):
			ctx._local_module_cache = {}
		return ctx._local_module_cache


def _get_or_make_module(ctx: Context, key: str, maker: Callable[[], nn.Module]) -> nn.Module:
	# Fetches a module from cache or builds a new one if missing.
	# Ensures the same Linear/Conv2d instance is reused across forward passes.
	cache = _get_modules(ctx)
	if key not in cache:
		cache[key] = maker()
	return cache[key]


def _get_optim_cache(ctx: Context) -> Dict[str, torch.optim.Optimizer]:
	# Separate cache for optimizers so training steps can reuse them too.
	try:
		return ctx.extra.setdefault("_optim_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_optim_cache"):
			ctx._local_optim_cache = {}
		return ctx._local_optim_cache


def _gather_params(ctx: Context) -> List[nn.Parameter]:
	# Collects all trainable parameters from every cached module.
	# This lets the Training node update the whole network at once.
	params: List[nn.Parameter] = []
	for m in _get_modules(ctx).values():
		for p in m.parameters(recurse=True):
			if p.requires_grad:
				params.append(p)
	return params


def _set_all_modules_training(ctx: Context, training: bool) -> None:
	# Switches every cached module into train/eval mode.
	# Dropout and BatchNorm behave differently in training mode.
	for m in _get_modules(ctx).values():
		m.train(training)


def _get_or_make_optimizer(ctx: Context, cfg: Dict[str, Any]) -> Optional[torch.optim.Optimizer]:
	# Builds or reuses an optimizer over all current parameters using simple settings.
	# If parameters change (e.g., graph shape changes), the optimizer is rebuilt automatically.
	params = _gather_params(ctx)
	if not params:
		return None
	opt_name = str(cfg.get("optimizer", "sgd")).lower()
	lr = float(cfg.get("lr", 1e-2))
	weight_decay = float(cfg.get("weight_decay", 0.0))
	momentum = float(cfg.get("momentum", 0.0))
	betas = tuple(cfg.get("betas", (0.9, 0.999)))

	key = f"{opt_name}|lr={lr}|wd={weight_decay}|mom={momentum}|betas={betas}"
	cache = _get_optim_cache(ctx)
	opt = cache.get(key)

	def make():
		if opt_name == "adam":
			return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
		# default: SGD (kids can see momentum effect later)
		return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)

	if opt is None:
		opt = make()
		cache[key] = opt
	else:
		current = sum(len(g["params"]) for g in opt.param_groups)
		if current != len(params):
			cache[key] = make()
			opt = cache[key]
	return opt


@node("InputNode")
def input_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Makes the very first datapack in a graph from a user-provided value.
	# Kids can feed numbers, lists, or images (lists of lists) here.
	device = _device_from_ctx(ctx)
	payload = props.get("value", [0.0, 1.0, 1.0, 1.0])
	t = _to_tensor(payload, device)
	return {"input_out": _make_pack(t, "input", None)}


def _dense(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Fully-connected layer: turns a list of numbers into a new list with chosen size.
	# Respects activation (“relu”, “sigmoid”, etc.). Caches weights so they can learn.
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() == 1:
		x = x.unsqueeze(0)
	in_features = x.shape[-1]

	cfg = props.get("config", {}) or {}
	out_features = int(props.get("neuron_count", cfg.get("units", in_features)))
	use_bias = bool(cfg.get("bias", True))

	key = _stable_cache_key("dense", {"in": in_features, "out": out_features, "bias": use_bias})
	layer = _get_or_make_module(ctx, key, lambda: nn.Linear(in_features, out_features, bias=use_bias).to(device))
	layer.train(bool(cfg.get("training", False)))

	y = layer(x)
	y = _apply_activation(y, props)
	return _make_pack(y, "dense", layer)


def _conv2d(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# 2D convolution: looks at images with a sliding window to make new feature maps.
	# Expects [N,C,H,W]. Uses cache so filters survive across steps and can be trained.
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() != 4:
		raise ValueError(f"Conv2d expects a 4D tensor [N,C,H,W], got shape {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	in_ch = x.shape[1]
	out_ch = int(props.get("neuron_count", cfg.get("out_channels", in_ch)))
	k = int(cfg.get("kernel_size", 3))
	s = int(cfg.get("stride", 1))
	p = int(cfg.get("padding", 1))
	use_bias = bool(cfg.get("bias", True))

	key = _stable_cache_key("conv2d", {"in": in_ch, "out": out_ch, "k": k, "s": s, "p": p, "b": use_bias})
	layer = _get_or_make_module(ctx, key, lambda: nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=use_bias).to(device))
	layer.train(bool(cfg.get("training", False)))

	y = layer(x)
	y = _apply_activation(y, props)
	return _make_pack(y, "conv2d", layer)


def _maxpool2d(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Max pooling: shrinks images by keeping the biggest value in each small window.
	# Has no weights; just changes size to help later layers see the big picture.
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() != 4:
		raise ValueError(f"MaxPool2d expects [N,C,H,W], got {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	k = int(cfg.get("kernel_size", 2))
	s = int(cfg.get("stride", k))
	p = int(cfg.get("padding", 0))

	pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
	y = pool(x)
	return _make_pack(y, "maxpool2d", None)


def _flatten(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Flatten: turns images or feature maps into a simple list so Dense can read them.
	# Keeps the batch dimension so training works on many items at once.
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() == 1:
		x = x.unsqueeze(0)
	N = x.shape[0]
	y = x.view(N, -1)
	return _make_pack(y, "flatten", None)


def _dropout(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Dropout: randomly hides some numbers while training to reduce overfitting.
	# Does nothing when not training; great to teach randomness and robustness.
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	cfg = props.get("config", {}) or {}
	p = float(cfg.get("p", 0.5))
	training = bool(cfg.get("training", False))

	drop = nn.Dropout(p=p)
	drop.train(training)
	y = drop(x)
	return _make_pack(y, "dropout", None)


_LAYER_TABLE: Dict[str, Callable[[List[Any], Dict[str, Any], Context], Dict[str, Any]]] = {
	"dense": _dense,
	"linear": _dense,
	"conv2d": _conv2d,
	"maxpool2d": _maxpool2d,
	"flatten": _flatten,
	"dropout": _dropout,
}


@node("NeuronLayer")
def neuron_layer(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Chooses which layer to run based on props.config.type and returns a datapack.
	# Because all ports carry lists, every handler receives a list and merges them safely.
	cfg = props.get("config", {}) or {}
	layer_type = str(cfg.get("type", "dense")).lower()
	handler = _LAYER_TABLE.get(layer_type)
	if handler is None:
		raise ValueError(f"Unknown layer type '{layer_type}'. Available: {list(_LAYER_TABLE.keys())}")
	pack_out = handler(inputs.get("layer_in", []), props, ctx)
	return {"layer_out": pack_out}


@node("TrainingNode")
def training_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	# Performs one training step:
	# 1) reads predictions and targets,
	# 2) computes a loss,
	# 3) backpropagates,
	# 4) updates all cached layers with an optimizer,
	# 5) returns the loss as a datapack so kids can “see learning”.
	device = _device_from_ctx(ctx)
	cfg = props.get("config", {}) or {}

	y_pred = _merge_inputs(inputs.get("pred_in", []), device)
	y_true = _merge_inputs(inputs.get("target_in", []), device)

	loss_name = str(cfg.get("loss", "mse")).lower()
	if loss_name in ("ce", "crossentropy", "cross_entropy"):
		if y_true.dim() == 2 and y_true.shape == y_pred.shape:
			y_true = y_true.argmax(dim=1)
		y_true = y_true.long()
		criterion = nn.CrossEntropyLoss()
	else:
		criterion = nn.MSELoss()

	_set_all_modules_training(ctx, True)
	opt = _get_or_make_optimizer(ctx, cfg)

	if opt is not None and bool(cfg.get("zero_grad", True)):
		opt.zero_grad(set_to_none=True)

	loss = criterion(y_pred, y_true)
	loss.backward()

	max_grad_norm = cfg.get("max_grad_norm", None)
	if opt is not None:
		if max_grad_norm is not None:
			nn.utils.clip_grad_norm_(_gather_params(ctx), float(max_grad_norm))
		opt.step()

	return {"layer_out": _make_pack(loss.detach(), "train_step", None)}
