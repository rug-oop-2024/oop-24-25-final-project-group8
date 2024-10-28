from abc import ABC, abstractmethod
import os
from typing import List, Union
from glob import glob

class NotFoundError(Exception):
    def __init__(self, path):
        super().__init__(f"Path not found: {path}")

class Storage(ABC):

    @abstractmethod
    def save(self, data: bytes, path: str):
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str):
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):

    def __init__(self, base_path: str="./assets"):
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str):
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str="/"):
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        
        # Use glob to list files
        keys = glob(path + "/**/*", recursive=True)

        # Convert full paths to relative paths and normalize to forward slashes
        relative_keys = [os.path.relpath(key, self._base_path).replace(os.sep, '/') for key in keys if os.path.isfile(key)]
        
        return relative_keys

    def _assert_path_exists(self, path: str):
        if not os.path.exists(path):
            raise NotFoundError(path)
    
    def _join_path(self, path: str) -> str:
        return os.path.join(self._base_path, path)


    