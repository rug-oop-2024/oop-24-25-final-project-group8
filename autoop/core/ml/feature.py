
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import pandas as pd

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    """
    Class for representing a feature in the dataset
    """
    name: str = Field(..., description="The name of the feature (column in the dataset).")
    feature_type: Literal['categorical', 'numerical'] = Field(..., description="The type of the feature: 'categorical' or 'numerical'.")