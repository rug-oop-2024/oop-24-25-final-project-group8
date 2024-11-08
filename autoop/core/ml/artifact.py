import os
import base64
import pandas as pd
import pickle
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Dict
import json

class Artifact(BaseModel):
    """
    A class to handle artifacts with associated metadata, supporting multiple data types such as DataFrame, 
    bytes, and dictionaries, and allowing saving and loading from disk.
    """
    name: str = Field(..., description="Name of the artifact")
    id: str = Field(default=None, description="Unique identifier for the artifact")
    asset_path: str = Field(..., description="Path to asset storage directory")
    data: Optional[Any] = Field(None, description="Structure that stores artifact data")
    type: str = Field("unknown type", description="Data type of the artifact (dataset, model, pipeline, etc.)")
    version: str = Field("1.0.0", description="Version of the artifact")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags specified by the user for the artifact")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata specified by the user")

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def generate_id_and_defaults(cls, values:dict[str, Any]) -> dict[str, Any]:
        """
        Generate a unique ID, asset path, and filename based on given values.
        """
        values['name'] = values.get('name', 'default_name')
        values['version'] = values.get('version', '1.0.0')
        values['type'] = values.get('type', 'unknown type')
        values['asset_path'] = values.get('asset_path') or f"{values['type']}/{values['name']}_v{values['version']}"
        
        if not values.get('id'):
            values['id'] = cls._generate_id(values['asset_path'], values['version'])
        
        return values

    @staticmethod
    def _generate_id(asset_path: str, version: str) -> str:
        encoded_path = base64.urlsafe_b64encode(asset_path.encode()).decode()
        return f"{encoded_path}v{version}"

    def save(self) -> None:
        """
        Saves the artifact's data in `assets/objects`, along with a JSON file in `assets/artifacts`.
        """
        # Use a consistent naming convention: {name}_{id}.{extension}
        filename = f"{self.name}_{self.id}.{self.asset_path.split('.')[-1]}"

        # Construct the full path for data and metadata storage
        data_path = os.path.join("assets", "objects", filename)
        metadata_path = os.path.join("assets", "artifacts", f"{self.id}.json")

        # Normalize asset_path for storage in metadata
        self.asset_path = filename

        # Save the actual data file
        self._save_data(data_path)
        
        # Save metadata to the specified JSON file
        self._save_metadata(metadata_path)

    def _save_data(self, data_path: str) -> None:
        """
        Saves the artifact data to the specified path based on its data type.
        
        Args:
            data_path (str): Path to save the data.
        """
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        if isinstance(self.data, pd.DataFrame):
            self.data.to_csv(f"{data_path}.csv", index=False)
            self.asset_path = f"{data_path}.csv"
        elif isinstance(self.data, bytes):
            with open(f"{data_path}", 'wb') as f:
                f.write(self.data)
            self.asset_path = f"{data_path}"
        elif isinstance(self.data, dict):
            with open(f"{data_path}.json", 'w') as f:
                json.dump(self.data, f, indent=4)
            self.asset_path = f"{data_path}.json"
        else:
            with open(f"{data_path}.pkl", 'wb') as f:
                pickle.dump(self.data, f)
            self.asset_path = f"{data_path}.pkl"

    def _save_metadata(self, metadata_path: str) -> None:
        """
        Saves the metadata to a JSON file.
        
        Args:
            metadata_path (str): Path to save the metadata.
        """
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump({
                "name": self.name,
                "version": self.version,
                "asset_path": self.asset_path,
                "tags": self.tags,
                "metadata": self.metadata,
                "type": self.type,
                "id": self.id
            }, f, indent=4)

    def load(self, file_path: str) -> None:
        """
        Loads the artifact's data and metadata from the specified file path.
        
        Args:
            file_path (str): Path to the file to load the artifact from.
        
        Raises:
            FileNotFoundError: If the specified file path does not exist.
            ValueError: If the file format is unsupported.
        """
        file_path = os.path.normpath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.endswith(".csv"):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith(".bin") or file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        metadata_file = file_path.replace(os.path.splitext(file_path)[1], ".json")
        if os.path.exists(metadata_file):
            self._load_metadata(metadata_file)
        else:
            print(f"Metadata file not found for {file_path}")

    def _load_metadata(self, metadata_file: str) -> None:
        """
        Loads the metadata from a JSON file and updates the artifact's attributes.
        
        Args:
            metadata_file (str): Path to the JSON metadata file.
        """
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.type = metadata.get("type", "unknown")
            self.version = metadata.get("version", "unknown")
            self.tags = metadata.get("tags", {})
            self.metadata = metadata.get("metadata", {})
            self.asset_path = metadata.get("asset_path", "unknown")
            self.id = metadata.get("id", "unknown")

        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading metadata from {metadata_file}: {e}")
