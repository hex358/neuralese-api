# tasks.py
import os
import time
from celery import Celery

# no external services
BROKER_URL = "memory://"
RESULT_BACKEND = "cache+memory://"

celery_app = Celery("nn_tasks", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
	task_serializer="json",
	accept_content=["json"],
	result_serializer="json",
	task_track_started=True,   # enables STARTED state
	task_acks_late=False,      # fine for in-process demo
	worker_send_task_events=False,
)

@celery_app.task(bind=True, name="nn.run_graph")
def run_graph(self, graph: dict, train: bool = False):
	total_steps = int(graph.get("total_steps", 100))
	phase = "train" if train else "inference"

	print(f"[worker] run_graph start phase={phase} total_steps={total_steps}")
	self.update_state(state="PROGRESS", meta={"phase": phase, "step": 0, "total_steps": total_steps})

	for step in range(1, total_steps + 1):
		# ---- your heavy work goes here ----
		time.sleep(0.05)
		# -----------------------------------
		if step % 5 == 0 or step == total_steps:
			self.update_state(state="PROGRESS", meta={"phase": phase, "step": step, "total_steps": total_steps})
			print(f"[worker] step {step}/{total_steps}")

	print("[worker] run_graph DONE")
	return {"ok": True, "phase": phase, "total_steps": total_steps}
