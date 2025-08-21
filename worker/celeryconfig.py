# celeryconfig.py
# minimal celery config using redis for broker+backend

broker_url = "redis://127.0.0.1:6379/0"
result_backend = "redis://127.0.0.1:6379/1"

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

timezone = "Asia/Almaty"
enable_utc = False

# safe defaults
task_ignore_result = False
result_expires = 3600  # seconds
