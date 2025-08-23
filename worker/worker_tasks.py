# worker_tasks.py
from __future__ import annotations
import time
from typing import Any, Dict, Callable

Emit = Callable[[Dict[str, Any]], None]

def train_task(emit: Emit, graph: Dict[str, Any]) -> Dict[str, Any]:
	emit({"phase": "start", "mode": "train"})
	epochs = 100
	print("training")
	for e in range(1, epochs + 1):
		print(e)
		time.sleep(0.5)
		emit({"phase": "train", "epoch": e, "epochs": epochs, "percent": int(100 * e / epochs)})
		call_external_module_training_step()
	return {"status": "ok", "mode": "train", "metrics": {"acc": 0.87}}

def infer_task(emit: Emit, graph: Dict[str, Any]) -> Dict[str, Any]:
	emit({"phase": "start", "mode": "infer"})
	steps = 6
	for i in range(1, steps + 1):
		time.sleep(0.4)
		emit({"phase": "infer", "step": i, "steps": steps, "percent": int(100 * i / steps)})
	return {"status": "ok", "mode": "infer", "outputs": {"prediction": 3}}
