from __future__ import annotations
from typing import Dict, Callable, Any, List, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------
# Node registry (decorator lives in engine; nodes import from here)
# ---------------------------------------------------------
NodeFn = Callable[[Dict[str, Any], Dict[str, Any], "Context"], Dict[str, Any]]
NODE_REGISTRY: Dict[str, NodeFn] = {}

def node(type_name: str):
	def deco(fn: NodeFn):
		NODE_REGISTRY[type_name] = fn
		fn.__node_type__ = type_name
		return fn
	return deco

# ---------------------------------------------------------
# Execution context
# ---------------------------------------------------------
@dataclass
class Context:
	run_id: str = "run-0"
	extra: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _sorted_page_keys(graph: Dict[str, Any]) -> List[str]:
	try:
		return [k for k, _ in sorted(((int(k), k) for k in graph.keys()))]
	except ValueError:
		return sorted(graph.keys())


def _ensure_dict(d) -> Dict[str, Any]:
	return d if isinstance(d, dict) else {}


def _merge_inbox(inbox: Dict[str, Dict[str, Dict[str, Any]]], page_key: str, node_id: str, port: str, value: Any):
	"""
	Collect inbound values for node inputs.
	Policy: if multiple upstreams write to the same input port, store a list.
	"""
	slot = inbox.setdefault(page_key, {}).setdefault(node_id, {})
	if port not in slot:
		slot[port] = value
	else:
		if not isinstance(slot[port], list):
			slot[port] = [slot[port]]
		slot[port].append(value)

# ---------------------------------------------------------
# Core executor
# ---------------------------------------------------------

@dataclass
class ExecutionResult:
	last_inbox: Dict[str, Dict[str, Any]]
	trace: Dict[Tuple[str, str], Dict[str, Any]]
	inbox_by_page: Dict[str, Dict[str, Dict[str, Any]]]


def execute_paged_graph(graph: Dict[str, Any], initial_inputs: Dict[str, Dict[str, Any]] | None = None, context: Context | None = None) -> ExecutionResult:
	"""
	graph: pages -> nodes -> {type, props, emit}
	initial_inputs: optional seed inputs: page->node_id->input_port->value
	context: optional Context
	"""
	context = context or Context()
	inbox_by_page: Dict[str, Dict[str, Dict[str, Any]]] = {}
	trace: Dict[Tuple[str, str], Dict[str, Any]] = {}

	if initial_inputs:
		for page_k, nodes in initial_inputs.items():
			for node_id, ports in nodes.items():
				for port, val in ports.items():
					_merge_inbox(inbox_by_page, page_k, node_id, port, val)

	page_keys = _sorted_page_keys(graph)

	for idx, page_k in enumerate(page_keys):
		page = _ensure_dict(graph.get(page_k))
		next_page_k = page_keys[idx + 1] if idx + 1 < len(page_keys) else None

		page_inbox = inbox_by_page.get(page_k, {})

		for node_id, node_blob in page.items():
			node_blob = _ensure_dict(node_blob)
			type_name = node_blob.get("type")
			props = _ensure_dict(node_blob.get("props"))
			emit = _ensure_dict(node_blob.get("emit"))

			if type_name not in NODE_REGISTRY:
				raise KeyError(f"Unknown node type: {type_name} (node {node_id} on page {page_k})")

			node_inputs: Dict[str, Any] = _ensure_dict(page_inbox.get(node_id))

			fn = NODE_REGISTRY[type_name]
			outputs: Dict[str, Any] = fn(node_inputs, props, context)
			trace[(page_k, node_id)] = outputs

			if next_page_k and emit:
				for out_port, fanouts in emit.items():
					val = outputs.get(out_port)
					if val is None:
						continue
					for target_node_id, target_ports in _ensure_dict(fanouts).items():
						for target_port in (target_ports or []):
							_merge_inbox(inbox_by_page, next_page_k, target_node_id, target_port, val)

	last_page_k = page_keys[-1] if page_keys else "0"
	last_inbox = inbox_by_page.get(last_page_k, {})

	return ExecutionResult(
		last_inbox=last_inbox,
		trace=trace,
		inbox_by_page=inbox_by_page,
	)
