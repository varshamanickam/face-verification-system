from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


def measure_latency_ms(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
	start = perf_counter()
	result = func(*args, **kwargs)
	elapsed_ms = (perf_counter() - start) * 1000.0
	return result, float(elapsed_ms)
