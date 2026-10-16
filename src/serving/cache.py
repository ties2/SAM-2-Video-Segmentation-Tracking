from collections import OrderedDict
from threading import RLock
from typing import Any, Protocol

from src.serving.config import settings


class ModelCache(Protocol):
    """
    Abstraction for Model Caching layer.
    """

    def get(self, model_name: str, version: str) -> Any: ...
    def set(self, model_name: str, version: str, model: Any) -> None: ...
    def contains(self, model_name: str, version: str) -> bool: ...
    def remove(self, model_name: str, version: str) -> None: ...
    def invalidate(self, model_name: str, version: str) -> None: ...
    def clear(self) -> None: ...


class InMemoryModelCache:
    """
    In-memory LRU implementation of ModelCache.

    The original version was an unbounded dict, which is fine for kilobyte-sized
    scikit-learn models but not for SAM 2 checkpoints (80 MB – 900 MB each): a few
    loaded versions would exhaust RAM.  ``max_items`` bounds the cache and evicts the
    least-recently-used model.  ``max_items=None`` keeps the old unbounded behaviour.
    A lock makes get/set safe under FastAPI's thread pool.
    """

    def __init__(self, max_items: int | None = None):
        self.max_items = max_items
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = RLock()
        self.evictions = 0

    def get(self, model_name: str, version: str) -> Any:
        with self._lock:
            key = (model_name, version)
            if key in self._cache:
                self._cache.move_to_end(key)
            return self._cache.get(key)

    def set(self, model_name: str, version: str, model: Any) -> None:
        with self._lock:
            key = (model_name, version)
            self._cache[key] = model
            self._cache.move_to_end(key)
            while self.max_items is not None and len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
                self.evictions += 1

    def contains(self, model_name: str, version: str) -> bool:
        with self._lock:
            return (model_name, version) in self._cache

    def remove(self, model_name: str, version: str) -> None:
        with self._lock:
            self._cache.pop((model_name, version), None)

    def invalidate(self, model_name: str, version: str) -> None:
        """
        Alias for remove. Used specifically during model promotion
        to explicitly state the domain intent of invalidating old cache.
        """
        self.remove(model_name, version)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# one sample (Singleton) for use in whole application
model_cache = InMemoryModelCache(max_items=settings.model_cache_max_items)
