from __future__ import annotations

import os.path
from typing import Dict, Any, Callable, Optional, List
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_core import node, Context, execute_graph


def _device_from_ctx(ctx: Context) -> torch.device:
	"""
	Choose CUDA if available, else CPU. Keeps tensors/modules aligned.
	"""
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stable_cache_key(prefix: str, props: Dict[str, Any]) -> str:
	"""
	Stable string key from layer settings; used to cache/reuse modules.
	"""
	try:
		body = json.dumps(props, sort_keys=True, default=str)
	except Exception:
		body = str(props)
	return f"{prefix}:{body}"


def _param_count(module: Optional[nn.Module]) -> int:
	"""
	Number of trainable parameters (for UI/debug). None -> 0.
	"""
	if module is None:
		return 0
	return sum(p.numel() for p in module.parameters())


def _make_pack(t: torch.Tensor, kind: str, module: Optional[nn.Module]) -> Dict[str, Any]:
	"""
	Uniform datapack wrapper for graph passing and UI inspection.
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
	if isinstance(value, torch.Tensor):
		# if it’s already on correct device, return as-is
		return value if value.device == device else value.to(device, non_blocking=True)
	# as_tensor avoids an extra copy when possible
	t = torch.as_tensor(value, dtype=torch.float32)
	return t.to(device, non_blocking=True) if device.type == "cuda" else t


def _merge_inputs(vals: List[Any], device: torch.device) -> torch.Tensor:
	if not isinstance(vals, list):  # keep your guardrails
		raise TypeError(f"_merge_inputs expected list, got {type(vals).__name__}")
	if len(vals) == 0:
		raise ValueError("No inputs provided on this port (empty list).")

	def unpack_one(v) -> torch.Tensor:
		if isinstance(v, dict) and "tensor" in v:
			return v["tensor"].to(device, non_blocking=True)
		return _to_tensor(v, device)

	if len(vals) == 1:
		return unpack_one(vals[0])

	base = unpack_one(vals[0]).clone()  # clone so we can add in-place
	for v in vals[1:]:
		t = unpack_one(v)
		if t.shape != base.shape:
			raise ValueError(f"Cannot sum inputs with shapes {base.shape} and {t.shape}.")
		base.add_(t)  # in-place add (no extra buffer)
	return base


# Activations (kid-friendly names)
_ACT = {
	"relu": torch.relu,
	"gelu": F.gelu,
	"sigmoid": torch.sigmoid,
	"tanh": torch.tanh,
	"none": lambda x: x,
}

def _apply_activation(x: torch.Tensor, props: Dict[str, Any], default_name: str = "none") -> torch.Tensor:
	"""
	Apply activation by name from props.config.activation (default fallback).
	"""
	cfg = props.get("config", {}) or {}
	act = str(cfg.get("activation", default_name)).lower()
	fn = _ACT.get(act, _ACT["none"])
	return fn(x)


# Caches (modules and optimizers)
def _get_modules(ctx: Context) -> Dict[str, nn.Module]:
	"""
	Module cache in ctx.extra: keeps parameters persistent across runs.
	"""
	try:
		return ctx.extra.setdefault("_module_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_module_cache"):
			ctx._local_module_cache = {}
		return ctx._local_module_cache

def set_eval(ctx: Context):
	for m in _get_modules(ctx).values():
		m.eval()

def _get_or_make_module(ctx: Context, key: str, maker: Callable[[], nn.Module]) -> nn.Module:
	"""
	Fetch or construct a module by stable key.
	"""
	cache = _get_modules(ctx)
	if key not in cache:
		cache[key] = maker()
	return cache[key]

def _get_optim_cache(ctx: Context) -> Dict[str, torch.optim.Optimizer]:
	"""
	Optimizer cache in ctx.extra (separate from module cache).
	"""
	try:
		return ctx.extra.setdefault("_optim_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_optim_cache"):
			ctx._local_optim_cache = {}
		return ctx._local_optim_cache

def _gather_params(ctx: Context) -> List[nn.Parameter]:
	"""
	Collect all trainable parameters from cached modules for a global optimizer.
	"""
	params: List[nn.Parameter] = []
	for m in _get_modules(ctx).values():
		for p in m.parameters(recurse=True):
			if p.requires_grad:
				params.append(p)
	return params

def _set_all_modules_training(ctx: Context, training: bool) -> None:
	"""
	Set train/eval mode for every cached module (affects Dropout/Norm).
	"""
	for m in _get_modules(ctx).values():
		m.train(training)

def _get_or_make_optimizer(ctx: Context, cfg: Dict[str, Any]) -> Optional[torch.optim.Optimizer]:
	params = _gather_params(ctx)
	if not params:
		return None

	opt_name = str(cfg.get("optimizer", "sgd")).lower()
	weight_decay = float(cfg.get("weight_decay", 0.0))

	if opt_name == "adam":
		# Use Adam's true defaults
		lr = float(cfg.get("lr", 1e-3))                # <-- was 1e-2
		betas = tuple(cfg.get("betas", (0.9, 0.999)))
		eps = float(cfg.get("eps", 1e-8))
		key = f"adam|lr={lr}|wd={weight_decay}|betas={betas}|eps={eps}"
		cache = _get_optim_cache(ctx)
		opt = cache.get(key)

		def make():
			return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

	else:
		# SGD path keeps its typical defaults
		lr = float(cfg.get("lr", 1e-2))
		momentum = float(cfg.get("momentum", 0.0))
		key = f"sgd|lr={lr}|wd={weight_decay}|mom={momentum}"
		cache = _get_optim_cache(ctx)
		opt = cache.get(key)

		def make():
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



# Training helpers
def _get_targets_from_config(props: Dict[str, Any], device: torch.device) -> torch.Tensor:
	"""
	Read ground-truth labels from props.config: 'target'|'targets'|'labels'.
	"""
	cfg = props.get("config", {}) or {}
	raw = None
	if "target" in cfg:
		raw = cfg["target"]
	return _to_tensor(raw, device)

def _align_targets(y_true: torch.Tensor, y_pred: torch.Tensor, loss_name: str) -> torch.Tensor:
	"""
	Align target shape to pred shape to avoid broadcasting warnings (MSE path).
	[ F ] + [B,F] -> [1,F]; scalar/size-1 -> expand to pred shape; else exact match required.
	"""
	if loss_name not in ("ce", "crossentropy", "cross_entropy"):
		if y_true.dim() == 0 or y_true.numel() == 1:
			return y_true.expand_as(y_pred)
		if y_true.shape == y_pred.shape:
			return y_true
		if y_true.dim() == 1 and y_pred.dim() == 2 and y_true.shape[0] == y_pred.shape[1]:
			return y_true.unsqueeze(0)
		raise ValueError(
			f"MSE target shape {tuple(y_true.shape)} incompatible with prediction shape {tuple(y_pred.shape)}. "
			"Provide scalar, [F] for [1,F], or exact match."
		)
	return y_true




@node("InputNode")
def input_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Create an initial datapack from user values (numbers/lists/tensors).
	Config: props.raw_values
	Port out: input_out
	"""
	device = _device_from_ctx(ctx)
	payload = props.get("raw_values", None)
	if payload is None: raise ValueError("input payload is empty")
	t = _to_tensor(payload, device)
	return {"input_out": _make_pack(t, "input", None)}

def _inputs_all_from_kind(vals: List[Any], kind: str) -> bool:
	if not isinstance(vals, list) or not vals:
		return False
	ok = True
	for v in vals:
		if isinstance(v, dict):
			ok = ok and (v.get("layer_kind") == kind)
	return ok

torch.backends.cudnn.benchmark = True           # fastest convs for fixed shapes
if hasattr(torch, "set_float32_matmul_precision"):
	torch.set_float32_matmul_precision("high")

def _dense(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Dense layer with stable identity via props.cache_tag and weight transplant on size edits.
	Kids‑friendly: accepts any input rank from InputNode; flattens to [1,F] if needed.
	"""
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)

	# Ensure [B,F]; special handling when fed directly by InputNode(s).
	if x.dim() == 1:
		x = x.unsqueeze(0)
	if _inputs_all_from_kind(pack_in, "input"):
		if x.dim() == 0:
			x = x.view(1, 1)
		elif x.dim() == 1:
			x = x.unsqueeze(0)
		else:
			x = x.view(1, -1)  # flatten anything else to [1,F]
	else:
		if x.dim() == 1:
			x = x.unsqueeze(0)

	in_features = x.shape[-1]

	cfg = props.get("config", {}) or {}
	out_features = int(props.get("neuron_count", cfg.get("units", in_features)))
	use_bias = bool(cfg.get("bias", True))

	# Stable, tag‑based cache key (doesn't include sizes)
	tag = _layer_tag(props, default="dense")
	cache_key = f"dense|tag={tag}"

	layer = _get_modules(ctx).get(cache_key)
	# Rebuild if layer missing or shape/bias changed
	if not isinstance(layer, nn.Linear) or \
	   layer.in_features != in_features or \
	   layer.out_features != out_features or \
	   ((layer.bias is not None) != use_bias):
		layer = _rebuild_linear_with_transplant(ctx, cache_key, in_features, out_features, use_bias, device)

	layer.train(bool(cfg.get("training", False)))

	y = layer(x.contiguous())
	y = _apply_activation(y, props, default_name="none")
	return _make_pack(y, "dense", layer)


def _conv2d(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Kids‑friendly Conv2D with stable cache_tag and weight transplant on channel edits.
	Config (UI‑friendly):
	  - finders: out_channels (default 16, or 32 if step==2)
	  - window:  kernel_size in {3,5} (default 3)
	  - step:    stride in {1,2}     (default 1)
	  - keep_size: if True and step==1 -> padding="same" (k//2), else 0
	  - activation: "relu"(default), "gelu", "tanh", "sigmoid", "none"
	Expects input [N,C,H,W]; in_channels inferred from x.
	"""
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() != 4:
		raise ValueError(f"Conv2d expects [N,C,H,W], got {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	in_ch = int(x.shape[1])

	step = int(cfg.get("step", 1))
	k    = int(cfg.get("window", 3))
	keep = bool(cfg.get("keep_size", step == 1))
	p    = (k // 2) if (keep and step == 1) else 0

	default_finders = 32 if step == 2 else 16
	out_ch = int(cfg.get("finders", props.get("neuron_count", default_finders)))
	use_bias = True  # kept simple for kids

	# Stable, tag‑based cache key
	tag = _layer_tag(props, default="conv2d")
	cache_key = f"conv2d|tag={tag}"

	layer = _get_modules(ctx).get(cache_key)
	# Check whether we need to rebuild (shape/stride/pad/bias change)
	need_rebuild = True
	if isinstance(layer, nn.Conv2d):
		same = (
			layer.in_channels == in_ch and
			layer.out_channels == out_ch and
			layer.kernel_size == (k, k) and
			layer.stride == (step, step) and
			layer.padding == (p, p) and
			((layer.bias is not None) == use_bias)
		)
		need_rebuild = not same

	if need_rebuild:
		layer = _rebuild_conv2d_with_transplant(ctx, cache_key, in_ch, out_ch, k, step, p, use_bias, device)

	layer.train(bool(cfg.get("training", False)))

	y = layer(x)
	y = _apply_activation(y, props, default_name="relu")
	return _make_pack(y, "conv2d", layer)

def _maxpool2d(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	MaxPool2D with stable identity; no weights to transplant.
	Config:
	  - shrink_by in {2,3} sets kernel_size=stride (default 2)
	Expects [N,C,H,W].
	"""
	device = _device_from_ctx(ctx)
	x = _merge_inputs(pack_in, device)
	if x.dim() != 4:
		raise ValueError(f"MaxPool2d expects [N,C,H,W], got {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	k = int(cfg.get("shrink_by", 2))
	if k not in (2, 3):
		raise ValueError("MaxPool2D 'shrink_by' must be 2 or 3.")

	# we keep a tiny module purely to have a stable entry in the cache (optional)
	tag = _layer_tag(props, default="maxpool2d")
	cache_key = f"maxpool2d|tag={tag}"

	cache = _get_modules(ctx)
	pool = cache.get(cache_key)
	if not isinstance(pool, nn.MaxPool2d) or pool.kernel_size != (k, k) or pool.stride != (k, k) or pool.padding != (0, 0):
		pool = nn.MaxPool2d(kernel_size=k, stride=k, padding=0)
		cache[cache_key] = pool

	y = pool(x)
	return _make_pack(y, "maxpool2d", None)


def _flatten(pack_in: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Flatten to [N,-1]. If given [F], becomes [1,F].
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
	Dropout regularization: config.p (default 0.5), config.training (bool).
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


def _squish(x: torch.Tensor):
	if x.dim() == 0:
		x = x.view(1, 1)
	elif x.dim() == 1:
		x = x.unsqueeze(0)
	else:
		x = x.view(1, -1)
	return x


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
	Router node: picks implementation by props.config.type and emits its datapack.
	Base safeguard: if this is the FIRST Dense fed directly by Input nodes and the
	input feature count changed since last run, crop/pad Input tensors to match the
	previously seen feature count so cached weights remain usable.
	"""
	device = _device_from_ctx(ctx)
	cfg = props.get("config", {}) or {}
	layer_type = str(cfg.get("type", "dense")).lower()
	handler = _LAYER_TABLE.get(layer_type)
	if handler is None:
		raise ValueError(f"Unknown layer type '{layer_type}'. Available: {list(_LAYER_TABLE.keys())}")

	pack_list = inputs.get("layer_in", [])

	# Execute concrete layer
	pack_out = handler(pack_list, props, ctx)
	return {"layer_out": pack_out}


def _best_slot(ctx: Context):
    return ctx.extra.setdefault("_best", {"loss": 999.0, "state": None, "step": -1})

def _snapshot_state(ctx: Context):
    # copy state_dicts of all cached modules
    return {k: m.state_dict() for k, m in _get_modules(ctx).items()}

def _restore_state(ctx: Context, state: Dict[str, Any]):
    if not state:
        return
    cache = _get_modules(ctx)
    for k, sd in state.items():
        if k in cache:
            cache[k].load_state_dict(sd)

def _maybe_update_best(ctx: Context, loss_val: float):
    best = _best_slot(ctx)
    if loss_val < best["loss"]:
        best["loss"] = loss_val
        best["state"] = _snapshot_state(ctx)
        best["step"] = ctx.extra.get("global_step", 0)

def _normalize_ce_targets(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Make targets valid for CrossEntropyLoss.
    Accepts:
      - scalar -> [1]
      - [N] class indices -> ok
      - [C] one-hot for a single sample with logits [1,C] -> argmax -> [1]
      - [N,C] one-hot -> argmax -> [N]
    Errors on ambiguous shapes.
    """
    # Ensure logits are [N, C]
    if y_pred.dim() == 1:
        # allow logits like [C] for single sample
        y_pred = y_pred.unsqueeze(0)
    if y_pred.dim() != 2:
        raise ValueError(f"CrossEntropy expects logits [N,C], got {tuple(y_pred.shape)}")
    N, C = y_pred.shape

    # scalar label
    if y_true.dim() == 0:
        return y_true.view(1).long()

    # [N] indices -> ok
    if y_true.dim() == 1:
        if y_true.numel() == N:
            return y_true.long()
        # special case: [C] one-hot for single sample with logits [1,C]
        if N == 1 and y_true.numel() == C:
            return y_true.argmax(dim=0, keepdim=True).long()
        raise ValueError(
            f"CE target 1-D has length {y_true.numel()} which doesn't match batch {N} "
            f"and isn't single-sample one-hot (C={C})."
        )

    # [N,C] one-hot -> indices
    if y_true.dim() == 2 and y_true.shape == (N, C):
        return y_true.argmax(dim=1).long()

    raise ValueError(
        f"Unsupported CE target shape {tuple(y_true.shape)} for logits shape {tuple(y_pred.shape)}."
    )


@node("SoftmaxNode")
def softmax_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = _device_from_ctx(ctx)
	inputed = _merge_inputs(inputs.get("layer_in", []), device)
	minimum = inputed.min(); maximum = inputed.max(); diff = maximum-minimum
	softmaxed = torch.softmax(inputed, 1)
	print(softmaxed)
	return {"soft_out": _make_pack(softmaxed, "softmax_node", None)}


@node("TrainInput")
def training_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	"""
	Single training step: merge predictions, read targets from config, compute loss,
	backprop, (optional) clip, optimizer step. Returns scalar loss datapack.
	Config: loss ("mse"/"crossentropy"), optimizer ("sgd"/"adam"), lr, weight_decay,
	momentum (sgd), betas (adam), zero_grad (bool), max_grad_norm (float|None),
	target|targets|labels (numbers/lists/tensor).
	"""
	device = _device_from_ctx(ctx)
	cfg = props.get("config", {}) or {}
	if not is_training: return {}
	global train_target_tensor
	cfg["target"] = train_target_tensor

	y_pred = _merge_inputs(inputs.get("pred_in", []), device)
	#print(y_pred)

	y_true = _get_targets_from_config(props, device)

	loss_name = str(cfg.get("loss", "mse")).lower()
	if loss_name in ("ce", "crossentropy", "cross_entropy"):
		# logits can be [C] or [N,C]; normalize inside helper
		y_true = _normalize_ce_targets(y_true, y_pred)
		# ensure logits are [N,C] after possible unsqueeze
		if y_pred.dim() == 1:
			y_pred = y_pred.unsqueeze(0)
		criterion = nn.CrossEntropyLoss()
	else:
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

	res_pack = _make_pack(loss.detach(), "train_step", None)
	#print(loss)
	train_target_tensor = res_pack
	return {"layer_out": res_pack}

def _get_optimizers(ctx: Context) -> Dict[str, torch.optim.Optimizer]:
	"""
	Returns the optimizer cache (same object used by _get_or_make_optimizer).
	This lets save_model/load_model serialize/restore all optimizers by key.
	"""
	try:
		return ctx.extra.setdefault("_optim_cache", {})
	except Exception:
		# Local fallback if ctx.extra is unavailable (mirrors _get_optim_cache)
		if not hasattr(ctx, "_local_optim_cache"):
			ctx._local_optim_cache = {}
		return ctx._local_optim_cache

def _get_modules(ctx: Context) -> Dict[str, nn.Module]:
	try:
		return ctx.extra.setdefault("_module_cache", {})
	except Exception:
		# If ctx.extra isn't there for some reason, make a local fallback
		if not hasattr(ctx, "_local_module_cache"):
			ctx._local_module_cache = {}
		return ctx._local_module_cache


def save_model(ctx: Context, path: str):
    """
    Collects all layer modules from the cache and saves their weights+optimizers.
    """
    modules = _get_modules(ctx)   # your helper that returns {layer_name: nn.Module}
    optims  = _get_optimizers(ctx)  # if you cache them the same way

    state = {
        "modules": {k: m.state_dict() for k, m in modules.items()},
        "optimizers": {k: o.state_dict() for k, o in optims.items()},
    }
    torch.save(state, path)
    print(f"Saved model to {path}")


def _layer_tag(props: Dict[str, Any], default: str) -> str:
    # Let users set props.cache_tag to stabilize identity across edits.
    # Fall back to default name if not provided.
    tag = props.get("cache_tag")
    if tag is None:
        # you can also derive from props.get("name") if your UI provides it
        tag = default
    return str(tag)

def _layer_tag(props: Dict[str, Any], default: str) -> str:
	"""
	Stable identity for a logical layer, independent of its current shape.
	Set props.cache_tag in your node JSON (e.g., "fc1", "conv3").
	If not provided, falls back to the default layer name.
	"""
	tag = props.get("cache_tag")
	return str(tag if tag is not None else default)


def _transplant_linear(dst: nn.Linear, src: nn.Linear) -> None:
	"""
	Copy overlapping weights/bias from src -> dst when in/out sizes change.
	"""
	with torch.no_grad():
		o = min(dst.out_features, src.out_features)
		i = min(dst.in_features,  src.in_features)
		dst.weight[:o, :i].copy_(src.weight[:o, :i])
		if dst.bias is not None and src.bias is not None:
			dst.bias[:o].copy_(src.bias[:o])


def _rebuild_linear_with_transplant(
	ctx: Context,
	cache_key: str,
	in_features: int,
	out_features: int,
	bias: bool,
	device: torch.device,
) -> nn.Linear:
	"""
	Rebuild a Linear layer under the same cache key and transplant from the old one if present.
	"""
	cache = _get_modules(ctx)
	old = cache.get(cache_key)
	new_layer = nn.Linear(in_features, out_features, bias=bias).to(device)
	if isinstance(old, nn.Linear):
		try:
			_transplant_linear(new_layer, old)
		except Exception:
			pass
	cache[cache_key] = new_layer
	return new_layer


def _transplant_conv2d(dst: nn.Conv2d, src: nn.Conv2d) -> None:
	"""
	Copy overlapping Conv2d weights when channel counts change.
	Requires same kernel size; if kernel differs, skip transplant.
	"""
	with torch.no_grad():
		# kernel mismatch -> give up (too ambiguous)
		if dst.weight.shape[2:] != src.weight.shape[2:]:
			return
		oc = min(dst.out_channels, src.out_channels)
		ic = min(dst.in_channels,  src.in_channels)
		kH, kW = dst.weight.shape[2], dst.weight.shape[3]
		dst.weight[:oc, :ic, :kH, :kW].copy_(src.weight[:oc, :ic, :kH, :kW])
		if dst.bias is not None and src.bias is not None:
			dst.bias[:oc].copy_(src.bias[:oc])


def _rebuild_conv2d_with_transplant(
	ctx: Context,
	cache_key: str,
	in_ch: int,
	out_ch: int,
	k: int,
	s: int,
	p: int,
	bias: bool,
	device: torch.device,
) -> nn.Conv2d:
	"""
	Rebuild a Conv2d under the same cache key and transplant overlapping weights if possible.
	"""
	cache = _get_modules(ctx)
	old = cache.get(cache_key)
	new_layer = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias).to(device)
	if isinstance(old, nn.Conv2d):
		try:
			_transplant_conv2d(new_layer, old)
		except Exception:
			pass
	cache[cache_key] = new_layer
	return new_layer


def _safe_load_linear(module: nn.Linear, sd: Dict[str, torch.Tensor]) -> None:
	"""Copy overlapping weights/bias from state_dict into a Linear of different size."""
	with torch.no_grad():
		if "weight" in sd:
			ws = sd["weight"]; wd = module.weight
			o = min(wd.shape[0], ws.shape[0])
			i = min(wd.shape[1], ws.shape[1])
			wd[:o, :i].copy_(ws[:o, :i])
		if module.bias is not None and "bias" in sd and sd["bias"] is not None:
			bs = sd["bias"]; bd = module.bias
			o = min(bd.shape[0], bs.shape[0])
			bd[:o].copy_(bs[:o])

def _safe_load_conv2d(module: nn.Conv2d, sd: Dict[str, torch.Tensor]) -> None:
	"""
	Copy overlapping channels for Conv2d if kernel sizes match. If kernel differs,
	skip (ambiguous how to map).
	"""
	with torch.no_grad():
		if "weight" in sd:
			ws = sd["weight"]; wd = module.weight
			# require same kernel size to transplant channels safely
			if ws.shape[2:] == wd.shape[2:]:
				oc = min(wd.shape[0], ws.shape[0])
				ic = min(wd.shape[1], ws.shape[1])
				kH, kW = wd.shape[2], wd.shape[3]
				wd[:oc, :ic, :kH, :kW].copy_(ws[:oc, :ic, :kH, :kW])
		if module.bias is not None and "bias" in sd and sd["bias"] is not None:
			bs = sd["bias"]; bd = module.bias
			oc = min(bd.shape[0], bs.shape[0])
			bd[:oc].copy_(bs[:oc])

def _safe_load_generic(module: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
	"""
	Try a forgiving load first; fall back to strict=False; finally ignore mismatches.
	"""
	try:
		module.load_state_dict(sd, strict=False)
	except Exception:
		# last resort: copy overlapping tensors param-by-param
		with torch.no_grad():
			for name, param in module.state_dict().items():
				if name in sd:
					src = sd[name]
					if param.shape == src.shape:
						param.copy_(src)  # exact copy
					# else: give up on this tensor (unknown mapping)

def _safe_load_module(module: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
	"""Dispatch to the best compatible loader per module type."""
	if isinstance(module, nn.Linear):
		_safe_load_linear(module, sd)
	elif isinstance(module, nn.Conv2d):
		_safe_load_conv2d(module, sd)
	else:
		_safe_load_generic(module, sd)

def _try_load_optimizers(ctx: Context, opt_state: Dict[str, Any]) -> None:
	"""
	Best-effort optimizer state restore: load when param-group sizes match; otherwise skip.
	"""
	if not opt_state:
		return
	# You already keep optimizers in a cache; reuse that here
	try:
		optim_cache = ctx.extra.get("_optim_cache", {})
	except Exception:
		optim_cache = {}
	if not isinstance(optim_cache, dict):
		return

	for key, opt in list(optim_cache.items()):
		sd = opt_state.get(key)
		if sd is None:
			continue
		# Only load if param group counts match; otherwise skip (structure changed)
		try:
			if "param_groups" in sd and len(sd["param_groups"]) == len(opt.param_groups):
				opt.load_state_dict(sd)
		except Exception:
			# silently skip incompatible optimizer states
			pass



def load_model(ctx: Context, path: str):
	"""
	Shape-tolerant loader:
	- For each cached module, copy overlapping weights (Linear/Conv2d) or load with strict=False.
	- Optimizers are restored only when param-group counts match; else skipped.
	"""
	if not os.path.exists(path):
		return
	device = _device_from_ctx(ctx)
	state = torch.load(path, map_location=device)

	modules = _get_modules(ctx)           # existing, already-constructed modules
	sd_mods = state.get("modules", {})
	for k, m in modules.items():
		ckpt_sd = sd_mods.get(k)
		if ckpt_sd is None:
			continue
		try:
			_safe_load_module(m, ckpt_sd)
		except Exception:
			# Never crash on load; the worst case is keeping current init
			pass

	# Optimizers (optional)
	_try_load_optimizers(ctx, state.get("optimizers", {}))

	print(f"Loaded (shape-tolerant) from {path}")


import db.lset as lset
train_target_tensor = None
is_training: str = ""
def train(pack: dict, context: Context, epochs: int, dataset: str):
	"""
	Run the provided graph 'pack' repeatedly for 'epochs' times using execute_graph.
	"""
	if not pack or not pack.get("pages"):
		return
	global train_target_tensor
	global is_training; is_training = dataset
	loss = 1.0
	best = _best_slot(context)  # initializes
	set_eval(context)
	prev_best = best.get(loss, 999.0)
	for _ in range(epochs):
		print("Epoch", _)
		loss = 0.0; length = 0
		for (x,y) in lset.DatasetIterable(dataset):
			page_pack = pack["pages"]["0"]
			key = list(page_pack.keys())[0]
			page_pack[key]["props"]["raw_values"] = x
			train_target_tensor = y
			exc = execute_graph(pack, context)
			loss += train_target_tensor["tensor"].item(); length += 1
		if best["loss"] < prev_best:
			print(best["loss"])

		loss_val = loss / length
		context.extra["last_loss"] = loss_val
		context.extra["global_step"] = context.extra.get("global_step", 0) + 1
		_maybe_update_best(context, loss_val)

	set_eval(context)
	if best["state"] is not None:
		print(f"Restoring best @ {best['step']} with loss {best['loss']}")
		_restore_state(context, best["state"])
	print("Saving model...")
	save_model(context, "digit")
	is_training = ""; train_target_tensor = None
