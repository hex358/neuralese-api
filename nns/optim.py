import torch, torch.nn as nn
from typing import Any, Dict, List, Optional
from .graph_core import Context
from .utils import count_params

def _modules(ctx: Context) -> Dict[str, nn.Module]:
	try:
		return ctx.extra.setdefault("_module_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_module_cache"):
			ctx._local_module_cache = {}
		return ctx._local_module_cache

def _optims(ctx: Context) -> Dict[str, torch.optim.Optimizer]:
	try:
		return ctx.extra.setdefault("_optim_cache", {})
	except Exception:
		if not hasattr(ctx, "_local_optim_cache"):
			ctx._local_optim_cache = {}
		return ctx._local_optim_cache

def all_params(ctx: Context) -> List[nn.Parameter]:
	params: List[nn.Parameter] = []
	for m in _modules(ctx).values():
		for p in m.parameters(recurse=True):
			if p.requires_grad:
				params.append(p)
	return params

def set_train_mode(ctx: Context, training: bool):
	for m in _modules(ctx).values():
		m.train(training)

def set_eval_mode(ctx: Context):
	for m in _modules(ctx).values():
		m.eval()

def get_or_make_optim(ctx: Context, cfg: Dict[str, Any]) -> Optional[torch.optim.Optimizer]:
	params = all_params(ctx)
	if not params: return None

	wd = float(cfg.get("weight_decay", 0.0))
	name = str(cfg.get("optimizer", "sgd")).lower()

	if name == "adam":
		# ADAM: lr, betas, eps, wd
		lr = float(cfg.get("lr", 1e-3))
		betas = tuple(cfg.get("betas", (0.9, 0.999)))
		eps = float(cfg.get("eps", 1e-8))
		key = f"adam|lr={lr}|wd={wd}|betas={betas}|eps={eps}"
		cache = _optims(ctx)
		opt = cache.get(key)

		def make():
			return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)

	else:
		# SGD: lr, momentum, wd
		lr = float(cfg.get("lr", 1e-2))
		mom = float(cfg.get("momentum", 0.0))
		key = f"sgd|lr={lr}|wd={wd}|mom={mom}"
		cache = _optims(ctx)
		opt = cache.get(key)

		def make():
			return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=mom)

	if opt is None:
		opt = make(); cache[key] = opt
	else:
		current = sum(len(g["params"]) for g in opt.param_groups)
		if current != len(params):
			cache[key] = make(); opt = cache[key]
	return opt
