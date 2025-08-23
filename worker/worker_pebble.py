# worker_pebble.py
from __future__ import annotations
import uuid, threading, queue, inspect
from typing import Any, Callable, Dict, Tuple, Optional, List
from pebble import ThreadPool

Progress = Dict[str, Any]
OnDone = Callable[[str, Optional[Any], Optional[BaseException]], Any]  # (job_id, result, error) -> None/awaitable

_pool = ThreadPool(max_workers=4)
_lock = threading.Lock()

_progress: Dict[str, "queue.Queue[Progress]"] = {}
_done: Dict[str, bool] = {}
_results: Dict[str, Any] = {}
_errors: Dict[str, BaseException] = {}
_done_ev: Dict[str, threading.Event] = {}
_on_done: Dict[str, List[OnDone]] = {}

# optional: set by server to schedule async callbacks on the Sanic/asyncio loop
_loop_call_soon_threadsafe: Optional[Callable[[Callable, Any], None]] = None

def bind_loop_callsoon(call_soon_threadsafe: Callable[[Callable, Any], None]) -> None:
	"""Server injects loop.call_soon_threadsafe to safely schedule async callbacks from worker threads."""
	global _loop_call_soon_threadsafe
	_loop_call_soon_threadsafe = call_soon_threadsafe

def _emit_for(job_id: str) -> Callable[[Progress], None]:
	def emit(payload: Progress):
		payload = dict(payload)
		payload.setdefault("job_id", job_id)
		with _lock:
			ch = _progress.get(job_id)
		if ch:
			ch.put(payload)
	return emit

def on_done(job_id: str, cb: OnDone) -> None:
	"""Register a server-side callback fired exactly once when the job completes."""
	with _lock:
		_on_done.setdefault(job_id, []).append(cb)

def _fire_on_done(job_id: str, result: Optional[Any], error: Optional[BaseException]) -> None:
	# snapshot callbacks to run outside the lock
	with _lock:
		callbacks = list(_on_done.get(job_id, []))
		_on_done.pop(job_id, None)

	for cb in callbacks:
		try:
			ret = cb(job_id, result, error)
			if inspect.isawaitable(ret) and _loop_call_soon_threadsafe:
				# schedule coroutine on the main event loop
				_loop_call_soon_threadsafe(_run_coro, ret)
		except Exception:
			# ignore secondary callback errors (or log)
			pass

def _run_coro(coro):
	# executed on the event loop thread via call_soon_threadsafe
	import asyncio
	asyncio.create_task(coro)

def submit(fn: Callable[[Callable[[Progress], None], Any], Any], *args, **kwargs) -> str:
	job_id = uuid.uuid4().hex
	with _lock:
		_progress[job_id] = queue.Queue()
		_done[job_id] = False
		_done_ev[job_id] = threading.Event()

	def _runner():
		emit = _emit_for(job_id)
		return fn(emit, *args, **kwargs)

	fut = _pool.schedule(_runner)

	def _on_done_f(f):
		try:
			res = f.result()
			with _lock:
				_results[job_id] = res
				_done[job_id] = True
				ev = _done_ev.get(job_id)
			if ev: ev.set()
			_emit_for(job_id)({"phase": "done"})
			_fire_on_done(job_id, res, None)
		except BaseException as e:
			with _lock:
				_errors[job_id] = e
				_done[job_id] = True
				ev = _done_ev.get(job_id)
			if ev: ev.set()
			_emit_for(job_id)({"phase": "error", "error": str(e)})
			_fire_on_done(job_id, None, e)

	fut.add_done_callback(_on_done_f)
	return job_id

def progress_queue(job_id: str) -> "queue.Queue[Progress]":
	with _lock:
		return _progress.setdefault(job_id, queue.Queue())

def poll_result(job_id: str) -> Tuple[bool, Optional[Any], Optional[str]]:
	with _lock:
		if not _done.get(job_id, False):
			return False, None, None
		err = _errors.get(job_id)
		if err:
			return True, None, repr(err)
		return True, _results.get(job_id), None

def wait_done(job_id: str, timeout: Optional[float] = None) -> Tuple[bool, Optional[Any], Optional[BaseException]]:
	"""Blocking wait (threading) for completion. Returns (finished, result, error)."""
	with _lock:
		ev = _done_ev.get(job_id)
	if ev is None:
		return True, _results.get(job_id), _errors.get(job_id)
	ok = ev.wait(timeout)
	if not ok:
		return False, None, None
	with _lock:
		return True, _results.get(job_id), _errors.get(job_id)
