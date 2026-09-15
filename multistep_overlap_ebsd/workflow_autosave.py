"""Atomic workflow checkpoints at completed processing boundaries."""
import os
from pathlib import Path
import tempfile


def save_checkpoint(session, path, ui_state):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".npz",
                                     dir=destination.parent)
    os.close(fd)
    try:
        session.save_workflow_state(temporary, ui_state=ui_state)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
