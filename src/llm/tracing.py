import json
import time
from pathlib import Path
from functools import wraps
from uuid import uuid4

TRACE_DIR = Path(__file__).parents[2] / "traces"
TRACE_DIR.mkdir(exist_ok=True)


def trace_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        run_id = str(uuid4())
        start = time.time()
        result = None
        error = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            error = repr(e)
            raise
        finally:
            end = time.time()
            record = {
                "run_id": run_id,
                "fn": fn.__name__,
                "args": str(args),
                "kwargs": kwargs,
                "result": str(result)[:2000],
                "error": error,
                "duration_sec": end - start,
                "ts": time.time(),
            }
            (TRACE_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2))
    return wrapper
