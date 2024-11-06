import os
import base64
import pandas as pd
import pickle
from pydantic import BaseModel, Field, model_validator
from typing import Union,Optional, Any
import json

class Artifact(BaseModel):
    name: str = Field(None, description="Name of Artifact")
    id: str = Field(None, description="Unique identifier for the artifact")
    asset_path: str = Field(None, description="Path to asset storage directory")
    data: Optional[Any] = Field(None, description="Structure which stores artifacts data")
    type: str = Field("unknown type", description="Data type of the artifact (dataset, model, pipeline etc.)")
    version: str = Field("1.0.0", description="Version of artifact")
    tags: dict = Field(default_factory=dict, description="Set of tags that can be specified by the user for the dataset")
    metadata: dict = Field(default_factory=dict, description="Dictionary of meta data that can be specified by the user")

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like pd.DataFrame, torch.nn.Module

    @model_validator(mode="before")
    def generate_id_and_defaults(cls, values):
        """
        Generate a unique ID, asset path, and filename based on given values.
        """
        # Ensure 'name' and 'version' defaults
        values['name'] = values.get('name', 'default_name')
        values['version'] = values.get('version', '1.0.0')
        
        # Set default type if not provided
        values['type'] = values.get('type', 'unknown type')

        # Generate default filename if asset_path is not provided
        values['asset_path'] = values.get('asset_path') or f"{values['type']}/{values['name']}_v{values['version']}"

        # Generate and encode the ID
        if not values.get('id'):
            values['id'] = cls._generate_id(values['asset_path'], values['version'])
        
        return values
    
    @staticmethod
    def _generate_id(asset_path: str, version: str) -> str:
        encoded_path = base64.urlsafe_b64encode(asset_path.encode()).decode()
        return f"{encoded_path}v{version}"

    def save(self) -> None:
        """
        Saves the artifact's data in assets/objects, along with a JSON file in assets/artifacts.
        """
        # Create filename with the desired format: name_artifactId
        self.asset_path = f"{self.name}v{self.id}"
        data_path = os.path.join("assets", "objects", self.asset_path)
        metadata_path =f"assets/artifacts/{self.id}.json" 

        # Saving the data based on its type
        if isinstance(self.data, pd.DataFrame):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            self.data.to_csv(data_path + ".csv", index=False)
            self.asset_path = data_path + ".csv"
        elif isinstance(self.data, bytes):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path + ".bin", 'wb') as f:
                f.write(self.data)
            self.asset_path = data_path + ".bin"
        elif isinstance(self.data, dict):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path + ".json", 'w') as f:
                json.dump(self.data, f, indent=4)
            self.asset_path = data_path + ".json"
        else:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path + ".pkl", 'wb') as f:
                pickle.dump(self.data, f)
            self.asset_path = data_path + ".pkl"

        # Normalize asset_path to use forward slashes
        self.asset_path = self.asset_path.replace("\\", "/")

        # Save metadata to the JSON file with the updated asset_path
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

        print(f"Artifact saved with data file at {self.asset_path} and metadata at {metadata_path}")

    def load(self, file_path: str) -> None:
        """
        Loads the artifact's data from a file and loads the metadata from the corresponding JSON file.
        
        Args:
            file_path: The path to the file from which to load the artifact.
          
        Raises:
            FileNotFoundError: If the specified file path does not exist.
            ValueError: If the file format is unsupported.
        """
        # Infer full path if the file path is not absolute or does not start with assets/
        if not os.path.isabs(file_path) and not file_path.startswith(self.ASSETS_DIR):
            file_path = os.path.join(self.ASSETS_DIR, file_path)

        # Check if the file exists, raise an error if not found
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the file based on its extension
        if file_path.endswith(".csv"):
            # Load as a dataset (CSV)
            self.data = pd.read_csv(file_path)
        elif file_path.endswith(".bin") or file_path.endswith(".pkl"):
            # Load binary data or pickled object
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Try to load metadata from the corresponding JSON file
        metadata_file = file_path.replace(os.path.splitext(file_path)[1], ".json")
        if os.path.exists(metadata_file):
            self._load_metadata(metadata_file)
        else:
            print(f"Metadata file not found for {file_path}")
        
        print(f"Artifact loaded from {file_path}")

    def _load_metadata(self, metadata_file: str) -> None:
        """
        Loads the metadata from the corresponding JSON file and updates the artifact's attributes.
        
        Args:
            metadata_file: The path to the JSON metadata file.
        """
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Update the artifact's attributes with the loaded metadata
            self.type = metadata.get("type", "unknown")
            self.version = metadata.get("version", "unknown")
            self.tags = metadata.get("tags", [])
            self.metadata = metadata.get("metadata", {})
            self.asset_path = metadata.get("asset_path", "unknown")
            self.id = metadata.get("id", "unknown")

            print(f"Metadata loaded from {metadata_file}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading metadata from {metadata_file}: {e}")
