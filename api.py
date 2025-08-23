from __future__ import annotations
from sanic import Sanic
from sanic.request import Request
from sanic.response import json
from sanic.log import logger
import asyncio, gzip, json as _json

import worker.worker_pebble as wp
import worker.worker_tasks as wt

app = Sanic("neuralese_api")

# ---------------------- lifecycle --------------------------------------------
@app.before_server_start
async def bind_loop(app, loop):
	# let worker schedule async callbacks on this loop (when task finishes)
	wp.bind_loop_callsoon(loop.call_soon_threadsafe)

# ---------------------- helpers ----------------------------------------------
async def stream_progress(ws, job_id: str):
	"""
	centralized progress→websocket pump.
	uses worker's per-job queue and emits a final 'done'/'error'.
	"""
	q = wp.progress_queue(job_id)
	await ws.send(_json.dumps({"job_id": job_id, "phase": "connected"}))
	try:
		while True:
			# drain progress without blocking the event loop
			try:
				update = await asyncio.get_running_loop().run_in_executor(None, q.get, True, 0.25)
				await ws.send(_json.dumps(update))
			except Exception:
				pass

			done, result, err = wp.poll_result(job_id)
			if done:
				if err:
					await ws.send(_json.dumps({"job_id": job_id, "phase": "error"}))
				else:
					await ws.send(_json.dumps({"job_id": job_id, "phase": "done", "result": result}))
				break
	finally:
		try:
			await ws.close()
		except Exception:
			pass

# ---------------------- HTTP endpoints (separated) ---------------------------
@app.post("/train")
async def start_train(request: Request):
	"""
	starts training via HTTP and returns a job_id. clients may:
	- call /await/<job_id> once (no polling), or
	- connect to /ws/<job_id> for live progress (optional),
	- or just rely on your server-side on_done callback.
	"""
	try:
		graph = _json.loads(gzip.decompress(request.body))
	except Exception:
		graph = request.json or {}

	job_id = wp.submit(wt.train_task, graph)

	# immediate server-side completion hook (no client polling)
	async def _notify(jid: str, result, error):
		if error:
			logger.error("train %s failed: %r", jid, error)
		else:
			logger.info("train %s done: %s", jid, result)

	wp.on_done(job_id, _notify)
	return json({"job_id": job_id, "mode": "train"})

@app.post("/infer")
async def start_infer(request: Request):
	"""
	starts inference via HTTP, returns a job_id (same options as /train).
	"""
	try:
		graph = _json.loads(gzip.decompress(request.body))
	except Exception:
		graph = request.json or {}

	job_id = wp.submit(wt.infer_task, graph)

	async def _notify(jid: str, result, error):
		if error:
			logger.error("infer %s failed: %r", jid, error)
		else:
			logger.info("infer %s done: %s", jid, result)

	wp.on_done(job_id, _notify)
	return json({"job_id": job_id, "mode": "infer"})

# ---------------------- OPTIONAL: classic progress WS by job_id --------------
@app.websocket("/ws/<job_id>")
async def ws_progress(request: Request, ws, job_id: str):
	await stream_progress(ws, job_id)

# ---------------------- WS train (graph from request payload) ----------------
@app.websocket("/ws/train")
async def ws_train(request, ws):
    # first frame from client is treated as "initial payload"
    raw = await ws.recv()
    try:
        graph = json.loads(raw)
    except Exception:
        graph = {}

    job_id = wp.submit(wt.train_task, graph)
    await stream_progress(ws, job_id)

# ---------------------- one-shot await (no polling) --------------------------
@app.get("/await/<job_id>")
async def await_once(request: Request, job_id: str):
	timeout = request.args.get("timeout", None)
	timeout = float(timeout) if timeout is not None else None

	# block in a worker thread on a threading.Event (doesn't block event loop)
	finished, result, err = await asyncio.to_thread(wp.wait_done, job_id, timeout)
	if not finished:
		return json({"job_id": job_id, "done": False, "timeout": True}, status=202)
	if err:
		return json({"job_id": job_id, "done": True, "error": repr(err)}, status=500)
	return json({"job_id": job_id, "done": True, "result": result})

# -----------------------------------------------------------------------------
if __name__ == "__main__":
	app.run(host="::", port=8000, debug=True)
