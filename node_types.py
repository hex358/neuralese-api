from __future__ import annotations
from typing import Dict, Any, Callable, Optional, List
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_core import node, Context, execute_graph
#rint(torch.cuda)


def _device_from_ctx(ctx: Context) -> torch.device:
	"""
	Reads the preferred device string from ctx.extra["device"] (e.g., "cpu", "cuda")
	and returns a torch.device. Keeps tensors/modules on the same device so math is valid.
	"""
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stable_cache_key(prefix: str, props: Dict[str, Any]) -> str:
	"""
	Builds a stable string key from layer settings. The key is used to cache and reuse
	the same PyTorch module across forward/training steps so its parameters persist.
	"""
	try:
		body = json.dumps(props, sort_keys=True, default=str)
	except Exception:
		body = str(props)
	return f"{prefix}:{body}"


def _param_count(module: Optional[nn.Module]) -> int:
	"""
	Returns the number of trainable parameters for display/debugging. None -> 0.
	"""
	if module is None:
		return 0
	return sum(p.numel() for p in module.parameters())


def _make_pack(t: torch.Tensor, kind: str, module: Optional[nn.Module]) -> Dict[str, Any]:
	"""
	Wraps a tensor and a few facts into a tiny 'datapack' dict. Every node returns
	a datapack so graphs can branch/merge uniformly and UIs can inspect shapes easily.
	"""
	return {
		"tensor": t,
		"shape": tuple(t.shape),
		"dtype": str(t.dtype).replace("torch.", ""),
		"device": str(t.device),
		"layer_kind": kind,
		"params": _param_count(module),
	}


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
	"""
	Converts Python numbers/lists or existing tensors to a torch.Tensor placed
	on the desired device. Preserves dtype=float32 by default for simplicity.
	"""
	if isinstance(value, torch.Tensor):
		return value.to(device)
	return torch.tensor(value, dtype=torch.float32, device=device)


def _merge_inputs(vals: List[Any], device: torch.device) -> torch.Tensor:
	"""
	Accepts a list of datapacks or raw values. Unpacks each to a tensor on the given
	device and elementwise-sums them. All inputs must share the exact same shape.
	Returns the single tensor unchanged when the list has length 1.
	"""
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
	"gelu": F.gelu,
	"sigmoid": torch.sigmoid,
	"tanh": torch.tanh,
	"none": lambda x: x,
}


def _apply_activation(x: torch.Tensor, props: Dict[str, Any], default_name: str = "none") -> torch.Tensor:
	"""
	Looks up props.config.activation (a lowercased string like "relu", "gelu", etc.)
	and applies it to the tensor. If not set, uses the provided default_name.
	Unknown names fall back to identity (no change).
	"""
	cfg = props.get("config", {}) or {}
	act = str(cfg.get("activation", default_name)).lower()
	fn = _ACT.get(act, _ACT["none"])
	return fn(x)


def _get_modules(ctx: Context) -> Dict[str, nn.Module]:
	"""
	Returns a dict-like cache (in ctx.extra) where each key is a layer signature and
	each value is a PyTorch module. This makes parameters persist across runs/training.
	"""
	try:
		return ctx.extra.setdefault("_module_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_module_cache"):
			ctx._local_module_cache = {}
		return ctx._local_module_cache


def _get_or_make_module(ctx: Context, key: str, maker: Callable[[], nn.Module]) -> nn.Module:
	"""
	Fetches a module from the cache by key, or creates and stores a new one by calling
	the provided 'maker'. Ensures the same module instance is reused later.
	"""
	cache = _get_modules(ctx)
	if key not in cache:
		cache[key] = maker()
	return cache[key]


def _get_optim_cache(ctx: Context) -> Dict[str, torch.optim.Optimizer]:
	"""
	Returns a dedicated optimizer cache from ctx.extra. We separate it from the module
	cache so different optimizer hyperparameters can coexist without conflicts.
	"""
	try:
		return ctx.extra.setdefault("_optim_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_optim_cache"):
			ctx._local_optim_cache = {}
		return ctx._local_optim_cache


def _gather_params(ctx: Context) -> List[nn.Parameter]:
	"""
	Collects all trainable parameters from every cached module. The TrainingNode uses
	this to update the whole network at once, regardless of branches/merges.
	"""
	params: List[nn.Parameter] = []
	for m in _get_modules(ctx).values():
		for p in m.parameters(recurse=True):
			if p.requires_grad:
				params.append(p)
	return params


def _set_all_modules_training(ctx: Context, training: bool) -> None:
	"""
	Sets every cached module to training or evaluation mode. This matters for Dropout
	and normalization layers; here we flip them to training during a train step.
	"""
	for m in _get_modules(ctx).values():
		m.train(training)


def _get_or_make_optimizer(ctx: Context, cfg: Dict[str, Any]) -> Optional[torch.optim.Optimizer]:
	"""
	Creates or reuses an optimizer over all current parameters using simple knobs:
	- optimizer: "sgd" or "adam"
	- lr: float (default 1e-2)
	- weight_decay: float (default 0.0)
	- momentum: float (only for sgd; default 0.0)
	- betas: pair (only for adam; default (0.9, 0.999))
	If the set of parameters changes (e.g., graph shape changed), the optimizer is rebuilt.
	"""
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
	"""
	Creates the first datapack from user data. Accepts numbers/lists/tensors and
	outputs a pack with the tensor, shape, dtype, and a friendly 'layer_kind' tag.
	Port: input_out
	"""
	device = _device_from_ctx(ctx)
	payload = props.get("value", [1.0, 1.0, 1.0, 1.0])
	t = _to_tensor(payload, device)
	return {"input_out": _make_pack(t, "input", None)}


def _dense(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Fully-connected layer for vectors. Expects [B, F] (adds batch when given [F]).
	Knobs:
	- neuron_count or config.units (int): output size
	- config.bias (bool): bias on/off (default True)
	- config.activation (str): "relu", "sigmoid", "tanh", "none" (default "none")
	"""
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
	y = _apply_activation(y, props, default_name="none")
	return _make_pack(y, "dense", layer)


def _conv2d(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Kids-friendly Conv2D. Expects [N, C, H, W]. Knobs (config):
	- finders (int): out_channels; default 16 (or 32 if step==2)
	- window (int): kernel_size; allowed 3 or 5; default 3
	- step (int): stride; allowed 1 or 2; default 1
	- keep_size (bool): if True and step==1 -> padding="same" (k//2); else padding=0
	- activation (str): "relu" (default here), "gelu", "tanh", "sigmoid", "none"
	Other fields are inferred (in_channels from input; bias=True by default).
	"""
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() != 4:
		raise ValueError(f"Conv2d expects [N,C,H,W], got {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	in_ch = int(x.shape[1])

	step = int(cfg.get("step", 1))
	k = int(cfg.get("window", 3))
	keep_size = bool(cfg.get("keep_size", step == 1))
	p = (k // 2) if (keep_size and step == 1) else 0

	default_finders = 32 if step == 2 else 16
	out_ch = int(cfg.get("finders", props.get("neuron_count", default_finders)))
	use_bias = True

	key = _stable_cache_key("conv2d", {"in": in_ch, "out": out_ch, "k": k, "s": step, "p": p, "b": use_bias})
	layer = _get_or_make_module(ctx, key, lambda: nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=step, padding=p, bias=use_bias).to(device))
	layer.train(bool(cfg.get("training", False)))

	y = layer(x)
	y = _apply_activation(y, props, default_name="relu")
	return _make_pack(y, "conv2d", layer)


def _maxpool2d(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Kids-friendly MaxPool2D. Expects [N, C, H, W]. Knobs (config):
	- shrink_by (int): sets kernel_size=stride={2 or 3}; default 2
	Padding, dilation, indices are hidden and kept at simple defaults.
	"""
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() != 4:
		raise ValueError(f"MaxPool2d expects [N,C,H,W], got {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	k = int(cfg.get("shrink_by", 2))
	if k not in (2, 3):
		raise ValueError("MaxPool2D 'shrink_by' must be 2 or 3.")
	pool = nn.MaxPool2d(kernel_size=k, stride=k, padding=0)
	y = pool(x)
	return _make_pack(y, "maxpool2d", None)


def _flatten(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Flattens any input to [N, -1] so Dense can follow CNN blocks. If given [F],
	it becomes [1, F]. Batch dimension is preserved as the first dimension.
	"""
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() == 1:
		x = x.unsqueeze(0)
	N = x.shape[0]
	y = x.view(N, -1)
	return _make_pack(y, "flatten", None)


def _dropout(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Dropout for regularization. Knobs (config):
	- p (float): probability to drop units during training; default 0.5
	- training (bool): when True, dropout is active; otherwise it is bypassed
	"""
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
	"convolution2d": _conv2d,
	"maxpool2d": _maxpool2d,
	"flatten": _flatten,
	"dropout": _dropout,
}


@node("NeuronLayer")
def neuron_layer(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Chooses a concrete layer implementation using props.config.type (case-insensitive).
	Supported types for kids-friendly CNNs include: "conv2d", "maxpool2d", plus "dense",
	"flatten", and "dropout". Expects every incoming port to be a list (often length-1).
	"""
	cfg = props.get("config", {}) or {}
	layer_type = str(cfg.get("type", "dense")).lower()
	handler = _LAYER_TABLE.get(layer_type)
	if handler is None:
		raise ValueError(f"Unknown layer type '{layer_type}'. Available: {list(_LAYER_TABLE.keys())}")
	pack_out = handler(inputs.get("layer_in", []), props, ctx)
	return {"layer_out": pack_out}


def _get_targets_from_config(props: Dict[str, Any], device: torch.device) -> torch.Tensor:
	cfg = props.get("config", {}) or {}
	if "target" in cfg:
		raw = cfg["target"]
	elif "targets" in cfg:
		raw = cfg["targets"]
	elif "labels" in cfg:
		raw = cfg["labels"]
	else:
		raise ValueError("TrainingNode requires targets in props.config under 'target', 'targets', or 'labels'.")
	return _to_tensor(raw, device)


def _align_targets(y_true: torch.Tensor, y_pred: torch.Tensor, loss_name: str) -> torch.Tensor:
	"""
	Normalizes target shape to match prediction shape to avoid broadcasting warnings.
	Rules:
	- MSE:
	  * If target is scalar or has 1 element -> expand to y_pred.shape.
	  * If target.shape == y_pred.shape -> OK.
	  * If target is [F] and y_pred is [B,F] -> unsqueeze to [1,F].
	  * Else -> error with a clear message.
	- CrossEntropy: handled elsewhere (indices [N] or one-hot [N,C]).
	"""
	if loss_name not in ("ce", "crossentropy", "cross_entropy"):
		# MSE / regression path
		if y_true.dim() == 0 or y_true.numel() == 1:
			return y_true.expand_as(y_pred)
		if y_true.shape == y_pred.shape:
			return y_true
		# Common kid case: target is [F], pred is [B,F] -> make [1,F]
		if y_true.dim() == 1 and y_pred.dim() == 2 and y_true.shape[0] == y_pred.shape[1]:
			return y_true.unsqueeze(0)
		raise ValueError(
			f"MSE target shape {tuple(y_true.shape)} is incompatible with prediction shape {tuple(y_pred.shape)}. "
			"Provide a scalar (will broadcast), a [F] vector (for [1,F] preds), or an exact match."
		)
	return y_true

@node("TrainInput")
def training_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = _device_from_ctx(ctx)
	cfg = props.get("config", {}) or {}

	y_pred = _merge_inputs(inputs.get("pred_in", []), device)
	y_true = _get_targets_from_config(props, device)

	loss_name = str(cfg.get("loss", "mse")).lower()

	if loss_name in ("ce", "crossentropy", "cross_entropy"):
		if y_pred.dim() != 2:
			raise ValueError(f"CrossEntropy expects logits of shape [N,C], got {tuple(y_pred.shape)}")
		if y_true.dim() == 2 and y_true.shape == y_pred.shape:
			y_true = y_true.argmax(dim=1)
		y_true = y_true.long()
		criterion = nn.CrossEntropyLoss()
	else:
		# Align shapes explicitly to avoid broadcasting warnings
		y_true = _align_targets(y_true, y_pred, loss_name)
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

	pack = _make_pack(loss.detach(), "train_step", None)
	print(pack)
	return {"layer_out": pack}

def train(pack: dict, epochs: int = 1, context: Context = None):
	for i in range(epochs):
		execute_graph(pack, context)