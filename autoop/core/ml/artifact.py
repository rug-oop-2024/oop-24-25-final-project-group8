from autoop import Validator

import os
import base64
import pandas as pd
import pickle
import torch
from pydantic import BaseModel, Field
from typing import Union, Optional, List
import json


class Artifact(BaseModel):
    ASSETS_DIR: str = Field("assets", description="Base directory where artifacts will be stored.")
    name: str = Field("name", description="Name of Artifact")
    asset_path: str = Field("asset path", description="Path to asset storage directory")
    data: Union[bytes, pd.DataFrame, torch.nn.Module, dict] = Field("data of artifact", description="Structure which stores artifacts data")
    version: str = Field("version", description="Version of artifact")

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like pd.DataFrame, torch.nn.Module

    def save(self, 
             data: Union[bytes, pd.DataFrame, torch.nn.Module, dict], 
             artifact_type: str, 
             filename: Optional[str] = None, 
             version: str = "1.0.0", 
             metadata: Optional[dict] = None, 
             tags: Optional[list] = None) -> None:
        """
        Saves the artifact's data in the assets directory, along with a full JSON structure.

        Args:
            data: The data to save (e.g., dataset, model).
            artifact_type: The type of data being saved (e.g., 'Pipeline', 'Dataset', 'Model', etc.).
            filename: The optional filename to save the artifact with.
            version: The version of the artifact (default is '1.0.0').
            metadata: A dictionary of metadata for the artifact (e.g., experiment_id, run_id).
            tags: Optional list of tags to describe the artifact.
        """
        
        # Validate input types
        validator = Validator()
        validator.validate(data, Union[bytes, pd.DataFrame, torch.nn.Module, dict])
        validator.validate(artifact_type, str)
        validator.validate(filename, str)
        validator.validate(version, str)
        validator.validate(metadata, dict)
        validator.validate(tags, list)

        # Initialize variables
        self.data = data
        self.artifact_type = artifact_type
        self.version = version
        self.asset_path = self._ensure_asset_directory()
        self.filename = filename or self._generate_filename()
        self.metadata = metadata or {}
        self.tags = tags or []
        
        file_path = os.path.join(self.asset_path, self.filename)
        
        # Saving the data based on its type
        if isinstance(data, pd.DataFrame):
            # Save the dataset as a CSV
            data.to_csv(file_path + ".csv", index=False)
        elif isinstance(data, bytes):
            # Save binary data (e.g., models, pickled objects)
            with open(file_path + ".bin", 'wb') as f:
                f.write(data)
        elif hasattr(data, "state_dict"):  # PyTorch model
            # Save PyTorch model
            torch.save(data, file_path + ".pth")
        elif isinstance(data, dict):
            # Save dictionaries (for model parameters/hyperparameters) as JSON
            with open(file_path + ".json", 'w') as f:
                json.dump(data, f, indent=4)
        else:
            # Save any other objects using pickle as a fallback
            with open(file_path + ".pkl", 'wb') as f:
                pickle.dump(data, f)
            
        # Save metadata to a JSON file with additional artifact details
        self._save_metadata(file_path + ".json")
        
        print(f"Artifact saved as {file_path}")

    def _save_metadata(self, json_path: str) -> None:
        """
        Saves the artifact's metadata (version, tags, and other details) to a JSON file.
        
        Args:
            json_path: The path to save the JSON metadata file.
        """
        # Generate an ID based on the asset_path and version
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()).decode()  # Base64-encode the asset path
        artifact_id = f"{encoded_path}:{self.version}"

        metadata = {
            "id": artifact_id,
            "asset_path": self.asset_path,
            "version": self.version,
            "type": self.artifact_type,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
        # Write the metadata to a JSON file
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def _ensure_asset_directory(self) -> str:
        """
        Ensures the assets directory exists and creates subdirectories based on the artifact type.
        
        Returns:
            str: The directory where the artifact will be saved.
        """
        # Create the base assets directory if it doesn't exist
        if not os.path.exists(self.ASSETS_DIR):
            os.makedirs(self.ASSETS_DIR)
        
        # Create subdirectory based on artifact type
        subdir = os.path.join(self.ASSETS_DIR, self.artifact_type + "s")  # "datasets", "models", etc.
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        
        return subdir

    def _generate_filename(self) -> str:
        """
        Generates a default filename if one is not provided.
        
        Returns:
            str: The generated filename.
        """
        base_name = self.artifact_type + "_v" + self.version  # e.g., "dataset_v1.0.0" or "model_v1.0.0"
        file_path = os.path.join(self.asset_path, base_name)
        
        counter = 1
        while os.path.exists(file_path):
            counter += 1
            base_name = f"{self.artifact_type}_v{counter}.0.0"
            file_path = os.path.join(self.asset_path, base_name)
        
        return base_name

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
        elif file_path.endswith(".pth"):
            # Load as a PyTorch model
            self.data = torch.load(file_path)
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
            self.artifact_type = metadata.get("type", "unknown")
            self.version = metadata.get("version", "unknown")
            self.tags = metadata.get("tags", [])
            self.metadata = metadata.get("metadata", {})
            self.asset_path = metadata.get("asset_path", "unknown")
            self.id = metadata.get("id", "unknown")

            print(f"Metadata loaded from {metadata_file}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading metadata from {metadata_file}: {e}")
