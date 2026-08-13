"""Lightweight logging + metric record keeping."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["get_logger", "configure_logging", "MetricHistory"]

_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, logfile: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass(slots=True)
class MetricHistory:
    """Append-only store of scalar metrics, dumpable to JSONL/JSON."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def log(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))

    def last(self, key: str) -> Any | None:
        for record in reversed(self.records):
            if key in record:
                return record[key]
        return None

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, default=float) + "\n")

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2, default=float), encoding="utf-8")
