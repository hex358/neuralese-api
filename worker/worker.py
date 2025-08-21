# worker.py
# celery worker with a single task that emulates training and publishes progress via redis

import json
import random
import time
from typing import Dict, Any

import redis
from celery import Celery, states
from celery.exceptions import Ignore

celery = Celery("nn_tasks")
celery.config_from_object("celeryconfig")

# progress storage: pub/sub channel + a hash for last-known status
def _mk_redis() -> redis.Redis:
	return redis.Redis(host="127.0.0.1", port=6379, db=2, decode_responses=True)

def _channel(job_id: str) -> str:
	return f"progress:{job_id}"

def _status_key(job_id: str) -> str:
	return f"status:{job_id}"

def _publish_progress(r: redis.Redis, job_id: str, payload: Dict[str, Any]) -> None:
	r.hset(_status_key(job_id), mapping=payload)
	r.publish(_channel(job_id), json.dumps(payload))

@celery.task(bind=True, name="train_graph")
def train_graph(self, graph: Dict[str, Any], model_name: str, epochs: int, dataset_path: str) -> Dict[str, Any]:
	job_id = self.request.id
	r = _mk_redis()

	try:
		# initial status
		status = {"state": "STARTED", "epoch": 0, "loss": 1.0, "acc": 0.10}
		_publish_progress(r, job_id, status)

		# emulate training curve
		for e in range(1, max(1, epochs) + 1):
			time.sleep(0.4)  # emulate work

			# simple synthetic curve with a bit of noise
			acc = min(0.98, 0.10 + (0.86 * e / epochs) + random.gauss(0, 0.01))
			loss = max(0.03, 1.00 - (0.90 * e / epochs) + random.gauss(0, 0.01))

			status = {"state": "PROGRESS", "epoch": e, "loss": round(float(loss), 4), "acc": round(float(acc), 4)}
			_publish_progress(r, job_id, status)

		# final
		final = {"state": "SUCCESS", "epoch": epochs, "final_loss": status["loss"], "final_acc": status["acc"]}
		_publish_progress(r, job_id, final)
		r.expire(_status_key(job_id), 3600)

		return {"ok": True, **final}

	except Exception as exc:
		err = {"state": "FAILURE", "error": str(exc)}
		_publish_progress(r, job_id, err)
		self.update_state(state=states.FAILURE, meta=err)
		raise
