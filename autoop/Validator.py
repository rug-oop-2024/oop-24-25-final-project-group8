from typing import Any, Type, Union, get_args

class Validator:
    """
    Class to validate input types to functions and classes  
    """
    def validate(self, data: Any, expected_type: Type) -> None:
        """
        Validate if the input data matches the expected type.
        
        Args:
            data: The data to validate.
            expected_type: The expected type (e.g., pd.DataFrame, int, str).
        
        Raises:
            ValueError: If the data does not match the expected type.
        """
        # Handle Union types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            # Extract the types from Union[...]
            valid_types = get_args(expected_type)
            if not any(isinstance(data, t) for t in valid_types):
                expected_names = ", ".join([t.__name__ for t in valid_types])
                raise ValueError(f"Invalid data type. Expected one of ({expected_names}), "
                                 f"but got {type(data).__name__}.")
        # Handle regular types
        elif not isinstance(data, expected_type):
            raise ValueError(f"Invalid data type. Expected {expected_type.__name__}, "
                             f"but got {type(data).__name__}.")