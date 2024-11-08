import json
from typing import Dict, Tuple, List, Union
from autoop.core.storage import Storage
import os

class Database:
    """
    A class representing a simple database stored in a file-based storage system.
    
    Attributes:
        storage (Storage): The storage backend to persist data.
        _data (Dict[str, Dict[str, dict]]): In-memory storage for database entries.
    """

    def __init__(self, storage: Storage) -> None:
        """
        Initialize the Database with a storage backend.
        
        Args:
            storage (Storage): The storage system to save and load data.
        """
        self._storage = storage
        self._data: Dict[str, Dict[str, dict]] = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """
        Store or update an entry in the database.
        
        Args:
            collection (str): The collection name.
            id (str): The unique ID for the entry.
            entry (dict): The data to store.

        Returns:
            dict: The stored entry.
        """
        assert isinstance(entry, dict), 'Data must be a dictionary'
        self._data.setdefault(collection, {})[id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """
        Retrieve an entry from the database by collection and ID.
        
        Args:
            collection (str): The collection name.
            id (str): The entry's unique ID.

        Returns:
            Union[dict, None]: The requested entry, or None if not found.
        """
        return self._data.get(collection, {}).get(id)

    def delete(self, collection: str, id: str) -> None:
        """
        Delete an entry from the database.
        
        Args:
            collection (str): The collection name.
            id (str): The entry's unique ID.
        """
        if self._data.get(collection, {}).pop(id, None) is not None:
            self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """
        List all entries in a collection.
        
        Args:
            collection (str): The collection name.

        Returns:
            List[Tuple[str, dict]]: A list of tuples (id, entry) for each entry in the collection.
        """
        return list(self._data.get(collection, {}).items())

    def refresh(self) -> None:
        """
        Refresh the in-memory data from the storage backend.
        """
        self._load()

    def _persist(self) -> None:
        """
        Persist the in-memory data to the storage.
        """
        # Save each entry in storage, organized by collection
        for collection, entries in self._data.items():
            for id, entry in entries.items():
                path = f"{collection}/{id}.json"
                self._storage.save(json.dumps(entry).encode(), path)
        
        # Remove items from storage if they no longer exist in the in-memory data
        for key in self._storage.list(''):
            collection, id = os.path.split(key)
            id = id.replace('.json', '')
            if not self._data.get(collection, {}).get(id):
                self._storage.delete(key)

    def _load(self) -> None:
        """
        Load data from storage into in-memory data structure.
        """
        self._data.clear()
        for key in self._storage.list(''):
            collection, id = os.path.split(key)
            id = id.replace('.json', '')
            try:
                data = self._storage.load(key)
                entry = json.loads(data.decode())
                if collection not in self._data:
                    self._data[collection] = {}
                self._data[collection][id] = entry
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {collection}/{id}")
            except Exception as e:
                print(f"Error loading key {collection}/{id}: {e}")
