import os, torch, torch.nn as nn
from typing import Any, Dict, List
from .graph_core import node, Context, execute_graph
from .utils import (
	pick_device, pack_tensor, to_tensor,
	layer_tag, inputs_all_from_kind,
	transplant_linear, transplant_conv,
	_partial_load_linear_from_sd, _partial_load_conv2d_from_sd,
	_partial_load_from_sd
)
from .merge import merge_inputs
from .activations import apply_act
from .optim import (
	_modules, _optims, all_params,
	set_train_mode, set_eval_mode, get_or_make_optim
)
import db.lset as lset

# global state used by training node
train_target_tensor = None
is_training: str = ""



# ----- input node -----
@node("InputNode")
def input_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	payload = props.get("raw_values", None)
	if payload is None:
		raise ValueError("input payload is empty")
	t = to_tensor(payload, device)
	return {"input_out": pack_tensor(t, "input", None)}



def _rebuild_linear(ctx: Context, key: str, in_f: int, out_f: int, bias: bool, device: torch.device) -> nn.Linear:
	cache = _modules(ctx)
	old = cache.get(key)
	layer = nn.Linear(in_f, out_f, bias=bias).to(device)

	# prefer restoring from checkpoint; else transplant from old
	ckpt = ctx.extra.get("_ckpt", {}).get(key)
	if isinstance(ckpt, dict):
		_partial_load_from_sd(layer, ckpt, device)
	elif isinstance(old, nn.Linear):
		try: transplant_linear(layer, old)
		except Exception: pass

	cache[key] = layer
	return layer

def dense_layer(inputs: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	x = merge_inputs(inputs, device)
	if x.dim() == 1: x = x.unsqueeze(0)
	if inputs_all_from_kind(inputs, "input"):
		if x.dim() == 0: x = x.view(1,1)
		elif x.dim() == 1: x = x.unsqueeze(0)
		else: x = x.view(1,-1)
	else:
		if x.dim() == 1: x = x.unsqueeze(0)

	in_f = x.shape[-1]
	cfg = props.get("config", {}) or {}
	out_f = int(props.get("neuron_count", cfg.get("units", in_f)))
	use_bias = bool(cfg.get("bias", True))

	tag = layer_tag(props, "dense")
	key = f"dense|tag={tag}"
	layer = _modules(ctx).get(key)
	if not isinstance(layer, nn.Linear) or \
	   layer.in_features != in_f or \
	   layer.out_features != out_f or \
	   ((layer.bias is not None) != use_bias):
		layer = _rebuild_linear(ctx, key, in_f, out_f, use_bias, device)

	layer.train(bool(cfg.get("training", False)))
	y = layer(x.contiguous())
	y = apply_act(y, props, "none")
	return pack_tensor(y, "dense", layer)


# ----- conv2d -----
def _rebuild_conv(ctx: Context, key: str, in_c: int, out_c: int, k: int, s: int, p: int, bias: bool, device: torch.device) -> nn.Conv2d:
	cache = _modules(ctx)
	old = cache.get(key)
	layer = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias).to(device)

	ckpt = ctx.extra.get("_ckpt", {}).get(key)
	if isinstance(ckpt, dict):
		_partial_load_from_sd(layer, ckpt, device)
	elif isinstance(old, nn.Conv2d):
		try: transplant_conv(layer, old)
		except Exception: pass

	cache[key] = layer
	return layer

def conv2d_layer(inputs: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	x = merge_inputs(inputs, device)
	if x.dim() != 4:
		raise ValueError(f"conv2d expects [N,C,H,W], got {tuple(x.shape)}")

	cfg = props.get("config", {}) or {}
	in_ch = int(x.shape[1])
	step = int(cfg.get("step", 1))
	k = int(cfg.get("window", 3))
	keep = bool(cfg.get("keep_size", step == 1))
	p = (k // 2) if (keep and step == 1) else 0
	default_finders = 32 if step == 2 else 16
	out_ch = int(cfg.get("finders", props.get("neuron_count", default_finders)))

	tag = layer_tag(props, "conv2d")
	key = f"conv2d|tag={tag}"
	layer = _modules(ctx).get(key)
	need_rebuild = True
	if isinstance(layer, nn.Conv2d):
		same = (
			layer.in_channels == in_ch and
			layer.out_channels == out_ch and
			layer.kernel_size == (k,k) and
			layer.stride == (step,step) and
			layer.padding == (p,p)
		)
		need_rebuild = not same
	if need_rebuild:
		layer = _rebuild_conv(ctx, key, in_ch, out_ch, k, step, p, True, device)

	layer.train(bool(cfg.get("training", False)))
	y = layer(x)
	y = apply_act(y, props, "relu")
	return pack_tensor(y, "conv2d", layer)


# ----- maxpool -----
def maxpool2d_layer(inputs: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	x = merge_inputs(inputs, device)
	if x.dim() != 4:
		raise ValueError(f"maxpool2d expects [N,C,H,W], got {tuple(x.shape)}")
	k = int(props.get("config", {}).get("shrink_by", 2))
	if k not in (2,3):
		raise ValueError("maxpool2d shrink_by must be 2 or 3")
	tag = layer_tag(props, "maxpool2d")
	key = f"maxpool2d|tag={tag}"
	cache = _modules(ctx)
	pool = cache.get(key)
	if not isinstance(pool, nn.MaxPool2d) or pool.kernel_size != (k,k):
		pool = nn.MaxPool2d(kernel_size=k, stride=k, padding=0)
		cache[key] = pool
	y = pool(x)
	return pack_tensor(y, "maxpool2d", None)


# ----- flatten -----
def flatten_layer(inputs: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	x = merge_inputs(inputs, device)
	if x.dim() == 1: x = x.unsqueeze(0)
	N = x.shape[0]
	y = x.view(N, -1)
	return pack_tensor(y, "flatten", None)


# ----- dropout -----
def dropout_layer(inputs: List[Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	x = merge_inputs(inputs, device)
	cfg = props.get("config", {}) or {}
	p = float(cfg.get("p", 0.5))
	training = bool(cfg.get("training", False))
	drop = nn.Dropout(p=p); drop.train(training)
	y = drop(x)
	return pack_tensor(y, "dropout", None)


# ----- layer router node -----
_LAYER_TABLE = {
	"dense": dense_layer,
	"linear": dense_layer,
	"conv2d": conv2d_layer,
	"convolution2d": conv2d_layer,
	"maxpool2d": maxpool2d_layer,
	"flatten": flatten_layer,
	"dropout": dropout_layer,
}

@node("NeuronLayer")
def neuron_layer(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	cfg = props.get("config", {}) or {}
	lt = str(cfg.get("type", "dense")).lower()
	fn = _LAYER_TABLE.get(lt)
	if fn is None:
		raise ValueError(f"unknown layer type '{lt}', available {list(_LAYER_TABLE.keys())}")
	packs = inputs.get("layer_in", [])
	out = fn(packs, props, ctx)
	return {"layer_out": out}


# ----- softmax node -----
@node("SoftmaxNode")
def softmax_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	device = pick_device(ctx)
	x = merge_inputs(inputs.get("layer_in", []), device)
	y = torch.softmax(x, 1)
	return {"soft_out": pack_tensor(y, "softmax_node", None)}


# ----- loss + training node -----
def _normalize_ce_targets(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
	if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(0)
	if y_pred.dim() != 2:
		raise ValueError(f"ce expects logits [N,C], got {tuple(y_pred.shape)}")
	N, C = y_pred.shape
	if y_true.dim() == 0: return y_true.view(1).long()
	if y_true.dim() == 1:
		if y_true.numel() == N: return y_true.long()
		if N == 1 and y_true.numel() == C:
			return y_true.argmax(dim=0, keepdim=True).long()
		raise ValueError("ce 1d target mismatch with batch or one-hot size")
	if y_true.dim() == 2 and y_true.shape == (N,C):
		return y_true.argmax(dim=1).long()
	raise ValueError(f"unsupported ce target shape {tuple(y_true.shape)} for logits {tuple(y_pred.shape)}")

@node("TrainInput")
def train_node(inputs: Dict[str, Any], props: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
	global train_target_tensor, is_training
	device = pick_device(ctx)
	cfg = props.get("config", {}) or {}
	if not is_training: return {}
	cfg["target"] = train_target_tensor
	y_pred = merge_inputs(inputs.get("pred_in", []), device)
	y_true = to_tensor(cfg.get("target"), device)

	loss_name = str(cfg.get("loss", "mse")).lower()
	if loss_name in ("ce","crossentropy","cross_entropy"):
		y_true = _normalize_ce_targets(y_true, y_pred)
		if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(0)
		crit = nn.CrossEntropyLoss()
	else:
		crit = nn.MSELoss()

	set_train_mode(ctx, True)
	opt = get_or_make_optim(ctx, cfg)
	if opt is not None and bool(cfg.get("zero_grad", True)):
		opt.zero_grad(set_to_none=True)

	loss = crit(y_pred, y_true)
	loss.backward()
	max_grad = cfg.get("max_grad_norm")
	if opt is not None:
		if max_grad is not None:
			nn.utils.clip_grad_norm_(all_params(ctx), float(max_grad))
		opt.step()

	res = pack_tensor(loss.detach(), "train_step", None)
	train_target_tensor = res
	return {"layer_out": res}


# ----- training orchestration -----
def _best_slot(ctx: Context):
	return ctx.extra.setdefault("_best", {"loss": 999.0, "state": None, "step": -1})

def _snapshot(ctx: Context):
	return {k: m.state_dict() for k,m in _modules(ctx).items()}

def _restore(ctx: Context, state: Dict[str, Any]):
	if not state: return
	cache = _modules(ctx)
	for k,sd in state.items():
		if k in cache: cache[k].load_state_dict(sd)

def _maybe_update_best(ctx: Context, loss_val: float):
	best = _best_slot(ctx)
	if loss_val < best["loss"]:
		best["loss"] = loss_val
		best["state"] = _snapshot(ctx)
		best["step"] = ctx.extra.get("global_step", 0)

def train(pack: dict, ctx: Context, epochs: int, dataset: str, output: str):
	global train_target_tensor, is_training
	if not pack or not pack.get("pages"): return
	is_training = dataset
	best = _best_slot(ctx)
	set_eval_mode(ctx)
	prev_best = best.get("loss", 999.0)
	for ep in range(epochs):
		print("epoch", ep)
		loss = 0.0; length = 0
		for (x,y) in lset.DatasetIterable(dataset):
			page_pack = pack["pages"]["0"]
			key = list(page_pack.keys())[0]
			page_pack[key]["props"]["raw_values"] = x
			train_target_tensor = y
			execute_graph(pack, ctx)
			loss += train_target_tensor["tensor"].item(); length += 1
		if best["loss"] < prev_best:
			print(best["loss"])
		val = loss/length
		ctx.extra["last_loss"] = val
		ctx.extra["global_step"] = ctx.extra.get("global_step", 0) + 1
		_maybe_update_best(ctx, val)
	set_eval_mode(ctx)
	if best["state"] is not None:
		print(f"restoring best @ {best['step']} loss {best['loss']}")
		_restore(ctx, best["state"])
	print("saving model...")
	save_model(ctx, output)
	is_training = ""; train_target_tensor = None


# ----- save/load -----
def save_model(ctx: Context, path: str):
	mods = _modules(ctx); opts = _optims(ctx)
	state = {"modules": {k:m.state_dict() for k,m in mods.items()},
	         "optimizers": {k:o.state_dict() for k,o in opts.items()}}
	torch.save(state, path)
	print(f"saved model to {path}")

def load_model(ctx: Context, path: str):
	if not os.path.exists(path): return
	device = pick_device(ctx)
	state = torch.load(path, map_location=device)

	# keep around for future builds
	ctx.extra["_ckpt"] = state.get("modules", {})

	# try restore into whatever already exists
	mods = _modules(ctx)
	for k, m in mods.items():
		sd = ctx.extra["_ckpt"].get(k)
		if isinstance(sd, dict):
			try:
				# best effort: partial copy
				_partial_load_from_sd(m, sd, device)
			except Exception:
				pass

	_try_load_opts(ctx, state.get("optimizers", {}))

def _try_load_opts(ctx: Context, opt_state: Dict[str, Any]):
	if not opt_state: return
	cache = _optims(ctx)
	for key,opt in list(cache.items()):
		sd = opt_state.get(key)
		if sd is None: continue
		try:
			if "param_groups" in sd and len(sd["param_groups"]) == len(opt.param_groups):
				opt.load_state_dict(sd)
		except Exception: pass
