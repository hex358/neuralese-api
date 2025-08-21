# redis_launcher.py
# spawns a local redis-server process with a tiny config

import os
import sys
import time
import shutil
import socket
import subprocess

DEF_PORT = 6379
DATA_DIR = os.path.abspath("redis_data")
CONF_PATH = os.path.join(DATA_DIR, "redis.conf")

def _port_open(host: str, port: int) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.settimeout(0.25)
		try:
			s.connect((host, port))
			return True
		except OSError:
			return False

def _find_redis_server() -> str | None:
	return shutil.which("redis-server") or shutil.which("redis-server.exe")

def main():
	os.makedirs(DATA_DIR, exist_ok=True)
	if _port_open("127.0.0.1", DEF_PORT):
		print(f"redis already running on 127.0.0.1:{DEF_PORT}")
		return

	redis_exe = _find_redis_server()
	if not redis_exe:
		print(
			"redis-server not found in PATH.\n"
			"- install Redis and ensure `redis-server` is on PATH\n"
			"- on Windows, you can install via winget: winget install Redis\n"
			"- on macOS: brew install redis\n"
			"- on Linux: use your package manager",
			file=sys.stderr,
		)
		sys.exit(1)

	conf = (
		f"port {DEF_PORT}\n"
		f"dir {DATA_DIR}\n"
		f"save 900 1\n"
		f"save 300 10\n"
		f"save 60 10000\n"
		f"appendonly no\n"
	)
	with open(CONF_PATH, "w", encoding="utf-8") as f:
		f.write(conf)

	print(f"starting redis-server on port {DEF_PORT} ...")
	proc = subprocess.Popen([redis_exe, CONF_PATH])

	# small wait loop for readiness
	for _ in range(50):
		if _port_open("127.0.0.1", DEF_PORT):
			print(f"redis started (pid={proc.pid}) on 127.0.0.1:{DEF_PORT}")
			break
		time.sleep(0.1)
	else:
		print("failed to start redis in time", file=sys.stderr)
		sys.exit(2)

	try:
		print("press Ctrl+C to stop redis...")
		proc.wait()
	except KeyboardInterrupt:
		print("stopping redis...")
		proc.terminate()
		proc.wait(timeout=5)
		print("redis stopped")

if __name__ == "__main__":
	main()
