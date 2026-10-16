import os
from typing import Protocol


# Custom Exceptions (Error Handling)
class StorageError(Exception):
    pass


class ArtifactNotFoundError(StorageError):
    pass


class InvalidURIError(StorageError):
    pass


class PathTraversalError(StorageError):
    pass


# Abstraction (Interface)
class ArtifactStorage(Protocol):
    def save(self, uri: str, content: bytes) -> None: ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...

    def delete(self, uri: str) -> None: ...


# Implementation
class LocalArtifactStorage:
    def __init__(self, base_dir: str):
        # change to Absolute Path for stronger security
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _resolve_path(self, uri: str) -> str:
        if not uri.startswith("local://"):
            raise InvalidURIError(f"Invalid URI scheme for LocalStorage: {uri}")

        relative_path = uri[len("local://") :].lstrip("/")

        full_path = os.path.abspath(os.path.join(self.base_dir, relative_path))

        # Security check: Path Traversal Protection
        if os.path.commonpath([full_path, self.base_dir]) != self.base_dir:
            raise PathTraversalError("Path traversal attempt detected")

        return full_path

    def save(self, uri: str, content: bytes) -> None:
        path = self._resolve_path(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    def get(self, uri: str) -> bytes:
        path = self._resolve_path(uri)

        if not os.path.exists(path):
            raise ArtifactNotFoundError(f"Artifact not found: {uri}")

        with open(path, "rb") as f:
            return f.read()

    def exists(self, uri: str) -> bool:
        path = self._resolve_path(uri)
        return os.path.exists(path)

    def delete(self, uri: str) -> None:
        path = self._resolve_path(uri)

        if not os.path.exists(path):
            raise ArtifactNotFoundError(f"Artifact not found: {uri}")

        os.remove(path)
