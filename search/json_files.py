"""
JSON files that processes on several hosts read and write over the NFS home.

Both writers fill a temporary file first and then publish it in one step, so a reader sees the old file or the new
one, never half of one. The temporary file is named after host and process, so two writers never share it.
`create_json` also fails if the file exists: one process owns a trial folder, a queue job or a claim.
"""
import json
import os
import socket
from pathlib import Path


def create_json(path: str | Path, data: dict) -> None:
    """Write `data` to `path`, which must not exist yet (FileExistsError otherwise)."""
    path = Path(path)
    text = json.dumps(data, indent=2)  # serialized first, so bad data leaves no file
    tmp = _tmp_path(path)
    try:
        tmp.write_text(text)
        try:
            os.link(tmp, path)  # fails if `path` exists
        except OSError:
            # On NFS a lost server reply can make a successful link report an error; the link count of the
            # unique file tells (Linux open(2), O_EXCL)
            if tmp.stat().st_nlink != 2:
                raise
    finally:
        tmp.unlink(missing_ok=True)  # also after a failed write, e.g. a full quota


def replace_json(path: str | Path, data: dict) -> None:
    """Write `data` to `path`, replacing the file if it exists."""
    path = Path(path)
    tmp = _tmp_path(path)
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # gone after a successful replace; left only by a failed write


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
