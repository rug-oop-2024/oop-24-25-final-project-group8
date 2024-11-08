from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List, Optional
import json
import pandas as pd
import pickle
from copy import deepcopy

import os


class ArtifactRegistry:
    _instance = None

    def __new__(cls, database: Database, storage: Storage):
        if cls._instance is None:
            cls._instance = super(ArtifactRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, database: Database, storage: Storage):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._database = database
        self._storage = storage
        self._initialized = True

    def register(self, artifact: Artifact):
        # Call the artifact's save method without arguments
        artifact.save() 
        
        # Add the artifact's metadata to the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)
    
    def list(self, type: str = None):
        """
        List all artifacts by reading JSON metadata files in assets/artifacts and filtering by type if specified.
        """
        artifacts_dir = "assets/artifacts"
        artifacts = []

        # Ensure the directory exists
        if not os.path.exists(artifacts_dir):
            print(f"Directory {artifacts_dir} does not exist.")
            return artifacts  # Return an empty list if the directory is missing

        # Iterate over all JSON files in assets/artifacts
        for file_name in os.listdir(artifacts_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(artifacts_dir, file_name)
                
                # Open and parse each JSON file
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)

                    # Filter by type, if specified
                    if type is None or metadata.get("type") == type:
                        artifact = Artifact(
                            id=metadata["id"],
                            name=metadata["name"],
                            version=metadata["version"],
                            asset_path=metadata["asset_path"],
                            tags=metadata.get("tags", {}),
                            metadata=metadata.get("metadata", {}),
                            data=None,  # Data will be loaded when needed
                            type=metadata["type"]
                        )
                        artifacts.append(artifact)
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error loading metadata from {file_path}: {e}")

        return artifacts
    
    def get(self, artifact_id: str) -> Optional[Artifact]:
        """
        Retrieves an artifact by loading its metadata from assets/dbo/artifacts
        and associated data from the path specified in the metadata.
        """
        # Path to the metadata JSON file based on artifact_id
        metadata_file = f"assets/artifacts/{artifact_id}.json"

        # Check if metadata file exists
        if not os.path.exists(metadata_file):
            print(f"Metadata file for artifact ID {artifact_id} not found.")
            return None

        try:
            # Load metadata from the JSON file
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Retrieve asset_path directly from the metadata
            data_file_path = metadata["asset_path"]
            loaded_data = None

            # Load the data based on file extension in asset_path
            if data_file_path.endswith(".bin"):
                with open(data_file_path, 'rb') as f:
                    loaded_data = f.read()  # Assume binary data is stored as bytes
            elif data_file_path.endswith(".csv"):
                loaded_data = pd.read_csv(data_file_path)
            elif data_file_path.endswith(".json"):
                with open(data_file_path, 'r') as f:
                    loaded_data = json.load(f)
            elif data_file_path.endswith(".pkl"):
                with open(data_file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
            else:
                print(f"Unsupported file format for {data_file_path}")

            # Determine whether to return a Dataset or a generic Artifact
            if metadata["type"] == "dataset":
                # Return a Dataset object if type is "dataset"
                return Dataset(
                    id=metadata["id"],
                    name=metadata["name"],
                    version=metadata["version"],
                    asset_path=metadata["asset_path"],
                    tags=metadata.get("tags", {}),
                    metadata=metadata.get("metadata", {}),
                    data=loaded_data,
                    type=metadata["type"]
                )
            else:
                # Return a generic Artifact object for other types
                return Artifact(
                    id=metadata["id"],
                    name=metadata["name"],
                    version=metadata["version"],
                    asset_path=metadata["asset_path"],
                    tags=metadata.get("tags", {}),
                    metadata=metadata.get("metadata", {}),
                    data=loaded_data,
                    type=metadata["type"]
                )

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading metadata or data for artifact ID {artifact_id}: {e}")
        except Exception as e:
            print(f"Unexpected error loading artifact ID {artifact_id}: {e}")

        return None
    
    def delete(self, artifact: Artifact):
        """
        Deletes the data file at the path specified in artifact.asset_path
        and removes the metadata JSON for the artifact.
        
        Args:
            artifact: The Artifact object containing the path to delete.
        """
        # Delete the data file
        if os.path.exists(artifact.asset_path):
            os.remove(artifact.asset_path)
        else:
            print(f"Data file '{artifact.asset_path}' not found.")
        
        # Delete the metadata JSON file in assets/dbo/artifacts directory
        metadata_file = f"assets/artifacts/{artifact.id}.json"
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        else:
            print(f"Metadata file '{metadata_file}' not found.")
        
        artifact = None

class AutoMLSystem:
    _instance = None  # Singleton instance attribute

    def __new__(cls, storage: LocalStorage, database: Database):
        if cls._instance is None:
            cls._instance = super(AutoMLSystem, cls).__new__(cls)
        return cls._instance

    def __init__(self, storage: LocalStorage, database: Database):
        # Initialize only once
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)
        self._initialized = True  # Flag to indicate initialization

    @staticmethod
    def get_instance():
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), 
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance
    
    @property
    def registry(self):
        return self._registry
    


