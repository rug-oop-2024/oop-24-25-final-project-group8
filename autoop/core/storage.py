from abc import ABC, abstractmethod
import os
from typing import List, Union
from glob import glob

class NotFoundError(Exception):
    """
    Class for throwing file not found exception
    """
    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path}")

class Storage(ABC):
    """Abstract base class for a storage system."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """Save data to a given path."""
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load data from a given path."""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete data at a given path."""
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """List all paths under a given path."""
        pass


class LocalStorage(Storage):
    """Concrete implementation of Storage that interacts with the local file system."""

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes LocalStorage with a base path, creating the base directory if it doesn't exist.

        Args:
            base_path (str): Base directory for storage. Defaults to './assets'.
        """
        self._base_path = base_path
        os.makedirs(self._base_path, exist_ok=True)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to the specified path under the base directory.

        Args:
            data (bytes): The data to save.
            key (str): The relative path where the data will be saved.
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                f.write(data)
            print(f"Data successfully written to {path}, size: {len(data)} bytes")
        except Exception as e:
            print(f"Error writing data to {path}: {e}")

    def load(self, key: str) -> bytes:
        """
        Load data from the specified path under the base directory.

        Args:
            key (str): The relative path to load data from.

        Returns:
            bytes: The data loaded from the file.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete the file at the specified path under the base directory.

        Args:
            key (str): The relative path to delete. Defaults to '/' (base path).
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        
        try:
            os.remove(path)
            print(f"Deleted file at {path}")
        except IsADirectoryError:
            raise NotFoundError(f"Path '{path}' is a directory, not a file")
        except Exception as e:
            print(f"Error deleting file at {path}: {e}")

    def list(self, prefix: str) -> List[str]:
        """
        List all files under the specified path, returning paths relative to the base directory.

        Args:
            prefix (str): The prefix path under which to list files.

        Returns:
            List[str]: A list of file paths relative to the base directory.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        
        keys = glob(os.path.join(path, '**', '*'), recursive=True)
        
        return [
            os.path.relpath(key, self._base_path).replace("\\", "/")
            for key in keys if os.path.isfile(key)
        ]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if the given path exists, raising NotFoundError if it does not.

        Args:
            path (str): Path to check for existence.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)
    
    def _join_path(self, path: str) -> str:
        """
        Join the base path with a given path and normalize to use forward slashes.

        Args:
            path (str): Relative path to join with the base path.

        Returns:
            str: The full path with consistent forward slashes.
        """
        return os.path.join(self._base_path, path).replace("\\", "/")
