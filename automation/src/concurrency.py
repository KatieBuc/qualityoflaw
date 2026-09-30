from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass
class ConcurrencyLimiter:
    max_workers: int
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._semaphore = threading.Semaphore(self.max_workers)

    @property
    def semaphore(self) -> threading.Semaphore:
        return self._semaphore

    def run_parallel(self, tasks: Iterable[Callable[[], T]]) -> list[T | BaseException]:
        task_list = list(tasks)
        if not task_list:
            return []

        if not self.enabled or self.max_workers == 1:
            results: list[T | BaseException] = []
            for task in task_list:
                try:
                    results.append(task())
                except BaseException as exc:
                    results.append(exc)
            return results

        results: list[T | BaseException | None] = [None] * len(task_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._run_task, task): index
                for index, task in enumerate(task_list)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except BaseException as exc:
                    results[index] = exc

        return results  # type: ignore[return-value]

    def _run_task(self, task: Callable[[], T]) -> T:
        return task()
