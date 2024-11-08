
from pydantic import BaseModel, Field
from typing import Literal

class Feature(BaseModel):
    """
    Class for representing a feature in the dataset
    """
    name: str = Field(..., description="The name of the feature (column in the dataset).")
    type: Literal['categorical', 'numerical'] = Field(..., description="The type of the feature: 'categorical' or 'numerical'.")