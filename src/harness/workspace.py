"""Per-run workspace creation and cleanup."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkspaceManager:
    def __init__(self, root: Path, *, keep: bool = False) -> None:
        self.root = Path(root).resolve()
        self.keep = keep

    def create(self, run_id: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=False)
        (path / ".tmp").mkdir(exist_ok=True)
        return path

    def cleanup(self, path: Path) -> None:
        if self.keep:
            return
        try:
            shutil.rmtree(path, ignore_errors=False)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not remove workspace %s", path, exc_info=True)
