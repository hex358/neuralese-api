from __future__ import annotations
from typing import Dict, Callable, Any, List, Tuple, Set
from dataclasses import dataclass, field

NodeFn = Callable[[Dict[str, Any], Dict[str, Any], "Context"], Dict[str, Any]]
NODE_REGISTRY: Dict[str, NodeFn] = {}

def node(type_name: str):
	def deco(fn: NodeFn):
		NODE_REGISTRY[type_name] = fn
		fn.__node_type__ = type_name
		return fn
	return deco

def run_node(type_name: str, *args, **kwargs):
	if not type_name in NODE_REGISTRY: return
	NODE_REGISTRY[type_name](*args, **kwargs)

@dataclass
class Context:
	run_id: str = "run-0"
	extra: Dict[str, Any] = field(default_factory=dict)

def _ensure_dict(d) -> Dict[str, Any]:
	return d if isinstance(d, dict) else {}

def _sorted_page_keys(graph_pages: Dict[str, Any]) -> List[str]:
	try:
		return [k for _, k in sorted((int(k), k) for k in graph_pages.keys())]
	except ValueError:
		return sorted(graph_pages.keys())


def _cap_append(global_inbox: Dict[str, Dict[str, List[Any]]],
                expect: Dict[str, Dict[str, int]],
                node_id: str, port: str, value: Any) -> None:
	want = expect.get(node_id, {}).get(port, 0)
	if want == 0:
		return
	box = global_inbox.setdefault(node_id, {}).setdefault(port, [])
	if want <= 1:
		if not box:
			box.append(value)
	else:
		if len(box) < want:
			box.append(value)

def _deliver_scheduled_to_global(inbox_by_page: Dict[str, Dict[str, Dict[str, Any]]],
                                 page_k: str,
                                 global_inbox: Dict[str, Dict[str, List[Any]]],
                                 expect: Dict[str, Dict[str, int]]) -> None:
	for node_id, ports in _ensure_dict(inbox_by_page.get(page_k)).items():
		for port, v in _ensure_dict(ports).items():
			if isinstance(v, list):
				for item in v:
					_cap_append(global_inbox, expect, node_id, port, item)
			else:
				_cap_append(global_inbox, expect, node_id, port, v)

def _is_ready(node_id: str,
              expect: Dict[str, Dict[str, int]],
              global_inbox: Dict[str, Dict[str, List[Any]]]) -> bool:
	exp_ports = expect.get(node_id, {})
	for port, need in exp_ports.items():
		have = len(global_inbox.get(node_id, {}).get(port, []))
		if have < need:
			return False
	return True

def _prepare_inputs(node_id: str,
                    expect: Dict[str, Dict[str, int]],
                    global_inbox: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Any]:
	res: Dict[str, Any] = {}
	exp_ports = expect.get(node_id, {})
	for port, need in exp_ports.items():
		vals = global_inbox.get(node_id, {}).get(port, [])
		if need <= 1:
			res[port] = vals
		else:
			res[port] = vals[:need]
	return res

@dataclass
class ExecutionResult:
	last_inbox: Dict[str, Dict[str, Any]]
	trace: Dict[Tuple[str, str], Dict[str, Any]]
	inbox_by_page: Dict[str, Dict[str, Dict[str, Any]]]


def execute_graph(pack: Dict[str, Any],
                             context: Context | None = None) -> ExecutionResult:
	context = context or Context()
	pages: Dict[str, Any] = _ensure_dict(pack.get("pages"))
	expect: Dict[str, Dict[str, int]] = _ensure_dict(pack.get("expect"))

	page_keys = _sorted_page_keys(pages)

	node_defs: Dict[str, Tuple[str, Dict[str, Any]]] = {}
	for page_k in page_keys:
		for nid, blob in _ensure_dict(pages[page_k]).items():
			node_defs[nid] = (page_k, blob)

	inbox_by_page: Dict[str, Dict[str, Dict[str, Any]]] = {}
	global_inbox: Dict[str, Dict[str, List[Any]]] = {}
	trace: Dict[Tuple[str, str], Dict[str, Any]] = {}
	active_nodes: Set[str] = set()
	executed: Set[str] = set()

	for idx, page_k in enumerate(page_keys):
		for nid in _ensure_dict(pages[page_k]).keys():
			active_nodes.add(nid)

		_deliver_scheduled_to_global(inbox_by_page, page_k, global_inbox, expect)

		ready_now = [nid for nid in active_nodes if nid not in executed and _is_ready(nid, expect, global_inbox)]
		for nid in ready_now:
			def_page, blob = node_defs[nid]
			type_name = blob.get("type")
			if type_name not in NODE_REGISTRY:
				raise KeyError(f"unknown node type {type_name} (node {nid})")
			props = _ensure_dict(blob.get("props"))
			fn = NODE_REGISTRY[type_name]

			inputs = _prepare_inputs(nid, expect, global_inbox)
			outputs = fn(inputs, props, context)
			trace[(def_page, nid)] = outputs

			if idx + 1 < len(page_keys):
				next_page_k = page_keys[idx + 1]
				emit = _ensure_dict(blob.get("emit"))
				for out_port, fanouts in emit.items():
					val = outputs.get(out_port)
					if val is None:
						continue
					for tgt_id, tgt_ports in _ensure_dict(fanouts).items():
						for tgt_port in (tgt_ports or []):
							slot = inbox_by_page.setdefault(next_page_k, {}).setdefault(tgt_id, {})
							if tgt_port not in slot:
								slot[tgt_port] = [val]
							else:
								slot[tgt_port].append(val)

			executed.add(nid)

	last_page_k = page_keys[-1] if page_keys else "0"
	last_inbox = inbox_by_page.get(last_page_k, {})
	return ExecutionResult(
		last_inbox=last_inbox,
		trace=trace,
		inbox_by_page=inbox_by_page,
	)
