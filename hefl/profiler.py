from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator

import psutil
import torch


def memory_snapshot() -> Dict[str, float]:
    process = psutil.Process()
    row = {"cpu_rss_mb": process.memory_info().rss / 1024**2}
    if torch.cuda.is_available():
        row["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        row["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
    else:
        row["gpu_allocated_mb"] = 0.0
        row["gpu_reserved_mb"] = 0.0
    return row


@dataclass
class RoundProfiler:
    sections: Dict[str, float] = field(default_factory=dict)
    start_memory: Dict[str, float] = field(default_factory=memory_snapshot)
    end_memory: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def timeit(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.sections[name] = self.sections.get(name, 0.0) + elapsed

    def finish(self) -> None:
        self.end_memory = memory_snapshot()

    def to_row(self) -> Dict[str, float]:
        if not self.end_memory:
            self.finish()
        row = dict(self.sections)
        row.update({f"start_{k}": v for k, v in self.start_memory.items()})
        row.update({f"end_{k}": v for k, v in self.end_memory.items()})
        row["cpu_rss_delta_mb"] = row.get("end_cpu_rss_mb", 0.0) - row.get("start_cpu_rss_mb", 0.0)
        return row
