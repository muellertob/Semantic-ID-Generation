"""
The heartbeat of one run: a file in the run folder that says which process runs it and when it last showed life.

A thread rewrites the file every `interval_s` seconds while the run lives. A killed worker stops rewriting it, so a
reader on any host can tell a dead run from a live one by the age of the time in the file. The first beat is written
on entry, before any stage starts. A failed write is logged, not raised: a short NFS hiccup must not end a run.
"""
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

from search.json_files import replace_json

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Heartbeat:
    def __init__(self, path: str | Path, interval_s: float = 60):
        self._path = Path(path)
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)

    def __enter__(self) -> "Heartbeat":
        self._beat()
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()  # wakes the thread at once, it does not wait out the interval
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._beat()

    def _beat(self) -> None:
        beat = {"host": socket.gethostname(), "pid": os.getpid(), "time": _now().isoformat(timespec="seconds")}
        try:
            replace_json(self._path, beat)
        except Exception:
            logger.exception("heartbeat write failed: %s", self._path)
