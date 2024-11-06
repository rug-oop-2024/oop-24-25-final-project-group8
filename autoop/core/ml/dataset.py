from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import pickle
import json
import io

class Dataset(Artifact):

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str="1.0.0"):
        """
        Create a Dataset artifact from a pandas DataFrame.
        
        Args:
            data: The pandas DataFrame to save as a CSV.
            name: The name of the dataset.
            asset_path: Path where the artifact will be stored.
            version: The version of the artifact (default is 1.0.0).
        """
        csv_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=csv_data,
            version=version,
            type="dataset"
        )
        
    def read(self) -> pd.DataFrame:
        """
        Reads the dataset from its stored format and returns it as a pandas DataFrame.
        """
        if self.asset_path.endswith(".csv"):
            # CSV data is directly readable by pandas
            return pd.read_csv(self.asset_path)
        elif self.asset_path.endswith(".json"):
            # JSON data needs to be loaded and then converted to a DataFrame if it's structured as such
            with open(self.asset_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif self.asset_path.endswith(".bin") or isinstance(self.data, bytes):
            # Assuming binary data represents a CSV encoded in bytes
            csv_string = self.data.decode() if isinstance(self.data, bytes) else open(self.asset_path, 'rb').read().decode()
            return pd.read_csv(io.StringIO(csv_string))
        elif self.asset_path.endswith(".pkl"):
            # Pickle files are loaded with pickle
            with open(self.asset_path, 'rb') as f:
                data = pickle.load(f)
            return pd.DataFrame(data) if isinstance(data, (pd.DataFrame, dict)) else data
        else:
            raise ValueError(f"Unsupported data format for asset path: {self.asset_path}")
    