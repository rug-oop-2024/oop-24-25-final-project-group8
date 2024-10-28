from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io

class Dataset(Artifact):

    def __init__(self, name: str, asset_path: str, data: pd.DataFrame, version: str) -> None:
        super().__init__(name=name, asset_path=asset_path, data=data, version=version)

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
        )
        
    def read(self) -> pd.DataFrame:
        """
        Reads the stored CSV data and returns it as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The dataset loaded from the CSV.
        """
        # `self.data` is expected to be the CSV content in bytes (loaded by Artifact class)
        csv_bytes = self.data
        csv_string = csv_bytes.decode()
        return pd.read_csv(io.StringIO(csv_string))
    
    def save(self, data: pd.DataFrame) -> bytes:
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
    