import streamlit as st
import pandas as pd
from typing import List, Literal, get_args

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature_type_detector import FeatureTypeDetector
from autoop.core.ml.model import CLASSIFICATION_MODELS, REGRESSION_MODELS, get_model
from autoop.core.ml.metric import METRICS, get_metric
from autoop.core.ml.pipeline import Pipeline
import re

# Set page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

#Display page header and description
st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Access the singleton instance of AutoMLSystem
automl = AutoMLSystem.get_instance()

# Initialize task_type outside any block to ensure it's accessible in the training summary
task_type = "Unknown"

# Retrieve the selected dataset ID from session state
selected_dataset_id = st.session_state.get("selected_dataset")

# Fetch the full dataset object from the registry using the ID
selected_dataset = automl.registry.get(selected_dataset_id) if selected_dataset_id else None

# Initialize session state variables if not already set
if "training_mode" not in st.session_state:
    st.session_state["training_mode"] = False  # Tracks if we're in 'training' mode

# initialze session state variables
if "target_feature" not in st.session_state:
    st.session_state["target_feature"] = None
if "target_feature_object" not in st.session_state:
    st.session_state["target_feature_object"] = None
if "input_features" not in st.session_state:
    st.session_state["input_features"] = []
if "selected_metrics" not in st.session_state:
    st.session_state["selected_metrics"] = []
if "train_split" not in st.session_state:
    st.session_state["train_split"] = 70 
if "selected_model_hyperparameters" not in st.session_state:
    st.session_state["selected_model_hyperparameters"] = {}
if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = None
if "selected_model_instance" not in st.session_state:
    st.session_state["selected_model_instance"] = None


# Temporary list to track changes without triggering session state update
if "temp_input_features" not in st.session_state:
    st.session_state["temp_input_features"] = []

if not st.session_state["training_mode"]:
    # Display selected dataset information
    st.write("## Dataset Selected:")

    # Set the key for the Show/Hide Details button state
    details_key = "selected_dataset_details"

    # Initialize the session state for the Show/Hide Details toggle if not set
    if details_key not in st.session_state:
        st.session_state[details_key] = False

    if selected_dataset:
        ### FETCHING THE SELECTED DATASET ###
        col1, col2 = st.columns([2, 1])

        # Display dataset information in col1
        with col1:
            st.markdown(
                f"<div style='font-size: 20px; font-style: italic;'> {selected_dataset.name} | {selected_dataset.version}</div>", 
                unsafe_allow_html=True
            )
            
        # Show/Hide Details button in col2
        with col2:
            toggle_text = "Hide Details" if st.session_state[details_key] else "Show Details"
            if st.button(toggle_text, key="btn_show_selected_dataset"):
                st.session_state[details_key] = not st.session_state[details_key]
                st.rerun()  # Refresh page to update the display immediately

        # Display dataset head if "Show Details" is active
        if st.session_state[details_key]:
            dataset_instance = automl.registry.get(selected_dataset.id)
            if dataset_instance:
                st.write("Dataset Head:")
                st.dataframe(dataset_instance.read().head())

        ### FEATURE SELECTION AND TASK SELECTION ###
            
        # Detect features using the FeatureTypeDetector
        features: List[Feature] = FeatureTypeDetector(selected_dataset).detect_feature_types()

        # Separate feature names
        feature_names = [feature.name for feature in features]

        st.write("### Feature Selection")

        # Use st.session_state['target_feature'] to set the default selection
        target_feature_box = st.selectbox(
            "Select Target Feature",
            feature_names,
            index=feature_names.index(st.session_state["target_feature"]) if st.session_state["target_feature"] in feature_names else 0,
            key="target_feature_widget"  # Unique key to separate it from session state key
        )

        if st.session_state["target_feature"] != target_feature_box:
            st.session_state["input_features"] = []
            st.session_state["input_features_objects"] = []
            st.session_state["target_feature"] = target_feature_box
            st.session_state["target_feature_object"] = next(feature for feature in features if feature.name == target_feature_box)
            st.session_state["selected_metrics"] = []

        # Filter out the selected target feature from input features list
        input_features_options = [feature for feature in feature_names if feature != st.session_state.get('target_feature')]

        # Filter previously selected input features to ensure they exist in the current options
        filtered_default_input_features = [feature for feature in st.session_state['input_features'] if feature in input_features_options]

        def update_input_features():
            # Set selected input feature names in session state
            selected_feature_names = st.session_state["input_features_widget"]
            st.session_state["input_features"] = selected_feature_names
            # Map feature names to Feature objects and store them in session state
            st.session_state["input_features_objects"] = [feature for feature in features if feature.name in selected_feature_names]
        
        # Use a temporary variable to hold selected input features
        input_features_box = st.multiselect(
            "Select Input Features",
            options=[f for f in feature_names if f != st.session_state["target_feature"]],
            default=st.session_state["input_features"],
            key="input_features_widget",
            on_change=update_input_features
        )

        # Display task type based on the selected target feature's type
        if target_feature_box:
            # Find the feature type for task detection
            target_feature_type = next((feature.type for feature in features if feature.name == target_feature_box), None)
            
            # Determine task type based on the target feature's type
            if target_feature_type == "numerical":
                task_type = "Regression"
            elif target_feature_type == "categorical":
                task_type = "Classification"
            else:
                task_type = "Unknown"

            st.write(f"**Detected Task Type:** {task_type}")

            # Display the model section if a target and at least one input feature are selected
            if input_features_box:
                target_type = next(f.type for f in features if f.name == target_feature_box)
                task_type = "classification" if target_type == "categorical" else "regression"
                st.session_state['task_type'] = task_type
                
                st.write("## Select Model")
                
                # Model selection dropdown based on task type
                model_options = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS

                # Callback function to update the selected model and clear previous hyperparameters
                def update_selected_model():
                    # Clear any existing hyperparameters in the session state
                    if "selected_model_name" in st.session_state:
                        prev_model_name = st.session_state["selected_model_name"]
                        if f"{prev_model_name}_hyperparameters" in st.session_state:
                            del st.session_state[f"{prev_model_name}_hyperparameters"]
                    
                    # Update the selected model name in session state
                    st.session_state["selected_model_widget"] = st.session_state["selected_model_box"]


                selected_model_name_box = st.selectbox(
                    "Choose a model",
                    model_options,
                    index=model_options.index(st.session_state["selected_model_widget"]) if "selected_model_widget" in st.session_state and st.session_state["selected_model_widget"] in model_options else 0,
                    key="selected_model_box",
                    on_change=update_selected_model
                )

                # Instantiate the model if selected
                if selected_model_name_box:
                    model_instance = get_model(selected_model_name_box)
                    st.session_state["selected_model_instance"] = model_instance
                    st.session_state["selected_model_name"] = model_instance.name
                    
                    # Ensure hyperparameters are initialized once in session state
                    if f"{selected_model_name_box}_hyperparameters" not in st.session_state:
                        st.session_state[f"{selected_model_name_box}_hyperparameters"] = model_instance.hyperparameters

                    # Reference the hyperparameters directly in session state
                    hyperparameters = st.session_state[f"{selected_model_name_box}_hyperparameters"]
                    st.session_state["selected_model_hyperparameters"] = hyperparameters

                    st.write("#### Set Hyperparameters")
                    
                    # Create input fields for each hyperparameter, directly using session state keys
                    for param, value in hyperparameters.items():
                        # Get Pydantic field constraints from metadata
                        field = model_instance.__fields__[param]

                        min_value = None
                        max_value = None

                        EPSILON = 1e-10
                        pattern = r'(gt|ge|lt|le)=([0-9]+)(\.*[0-9]+)*'
                        for item in field.metadata:
                            matches = re.findall(pattern, str(item))
                            # Iterate through found matches
                            for constraint, int_value, decimal_value in matches:
                                # Combine integer and decimal parts
                                full_value = float(int_value + (decimal_value if decimal_value else ""))
                                
                                # Set min_value and max_value based on the constraints
                                if constraint == "gt":
                                    min_value = full_value + EPSILON
                                elif constraint == "ge":
                                    min_value = full_value  # ge means greater than or equal to
                                elif constraint == "lt":
                                    max_value = full_value - EPSILON 
                                elif constraint == "le":
                                    max_value = full_value  # le means less than or equal to

                        # Check if the field type is a Literal and extract allowed values for dropdown
                        field_type = field.annotation
                        literal_values = get_args(field_type) if hasattr(field_type, '__origin__') and field_type.__origin__ is Literal else None

                        # If the field is Literal, render a dropdown with allowed values
                        if literal_values:
                            updated_value = st.selectbox(
                                f"{param}",
                                options=literal_values,
                                index=literal_values.index(st.session_state.get(f"{selected_model_name_box}_{param}", value)),
                                key=f"{selected_model_name_box}_{param}"
                            )
                        # Determine the widget based on the type and handle None defaults
                        elif value is None:
                            # Generate the range based on min and max values for fields with None default
                            if min_value is not None and max_value is not None:
                                options = [None] + list(range(int(min_value), int(max_value) + 1))
                            elif min_value is not None:
                                options = [None] + list(range(int(min_value), int(min_value) + 20))  # Fallback range if only min_value is defined
                            elif max_value is not None:
                                options = [None] + list(range(int(max_value) - 20, int(max_value) + 1))  # Fallback range if only max_value is defined
                            else:
                                options = [None] + list(range(1, 21))  # General fallback if no constraints are defined

                            # Display a selectbox with `None` as an option
                            updated_value = st.selectbox(
                                f"{param} (Optional)",
                                options=options,
                                format_func=lambda x: "None" if x is None else x,
                                index=0 if st.session_state.get(f"{selected_model_name_box}_{param}", value) is None else options.index(st.session_state.get(f"{selected_model_name_box}_{param}", value)),
                            )
                        elif isinstance(value, bool):
                            # Checkbox for boolean values
                            updated_value = st.checkbox(
                                param,
                                value=st.session_state.get(f"{selected_model_name_box}_{param}", value),
                                key=f"{selected_model_name_box}_{param}",
                            )
                        elif isinstance(value, int):
                            # Render with constraints; if out of range, adjust to the nearest boundary
                            input_value = st.session_state.get(f"{selected_model_name_box}_{param}", value)
                            adjusted_value = max(min(input_value, max_value) if max_value is not None else input_value, min_value) if min_value is not None else input_value
                            updated_value = st.number_input(
                                param,
                                min_value=int(min_value) if min_value is not None else None,
                                max_value=int(max_value) if max_value is not None else None,
                                value=int(adjusted_value),
                                step=1,
                                key=f"{selected_model_name_box}_{param}"
                            )
                        elif isinstance(value, float):
                            # Render with constraints; if out of range, adjust to the nearest boundary
                            input_value = st.session_state.get(f"{selected_model_name_box}_{param}", value)
                            adjusted_value = max(min(input_value, max_value) if max_value is not None else input_value, min_value) if min_value is not None else input_value
                            updated_value = st.number_input(
                                param,
                                min_value=float(min_value) if min_value is not None else None,
                                max_value=float(max_value) if max_value is not None else None,
                                value=float(adjusted_value),
                                step=0.01,
                                key=f"{selected_model_name_box}_{param}"
                            )
                        elif isinstance(value, str):
                            updated_value = st.text_input(
                                param,
                                value=st.session_state.get(f"{selected_model_name_box}_{param}", value),
                                key=f"{selected_model_name_box}_{param}",
                            )

                    # Update the main hyperparameters dictionary with new values from widgetas
                    updated_hyperparameters = {
                        param: st.session_state[f"{selected_model_name_box}_{param}"]
                        for param in hyperparameters.keys()
                    }
                    # Assign the updated dictionary back to the main hyperparameters in session state
                    st.session_state[f"{selected_model_name_box}_hyperparameters"] = updated_hyperparameters

                    # Step: Train-Test Split Selection
                    st.write("#### Select Train-Test Split")

                    def update_split():
                        # Directly set selected_metrics in session state based on widget state
                        st.session_state["train_split"] = st.session_state["split_widget"]

                    # Use a slider to let the user choose the percentage for the training set
                    st.session_state["train_split"] = st.slider(
                        "Select Train Split Percentage",
                        min_value=1, max_value=99,
                        value=st.session_state["train_split"],
                        key="split_widget",
                        step=1,
                        help="Select the percentage of data to use for training. The rest will be used for testing.",
                        on_change=update_split
                    )

                    # Calculate the test split automatically
                    train_split = st.session_state["train_split"]
                    test_split = 100 - train_split

                    # Display the selected split
                    st.write(f"Training set: {train_split}%, Testing set: {test_split}%")

                    # Step: Select Metrics Step
                    task_type = "classification" if target_feature_type == "categorical" else "regression"

                    # Filter metrics based on task type
                    if task_type == "classification":
                        compatible_metrics = ["accuracy", "precision", "recall", "f1_score"]
                    elif task_type == "regression":
                        compatible_metrics = ["mean_squared_error", "root_mean_squared_error", "mean_absolute_error"]

                    st.write("### Select Evaluation Metrics")
                    # Define a callback function to update session state directly for selected metrics
                    def update_selected_metrics():
                        # Directly set selected_metrics in session state based on widget state
                        st.session_state["selected_metrics"] = st.session_state["metrics_widget"]

                    # Multiselect widget with an on_change callback for metrics selection
                    selected_metrics_box = st.multiselect(
                        "Select Evaluation Metrics",
                        options=compatible_metrics,
                        default=st.session_state["selected_metrics"],  # Initialize with current session state
                        key="metrics_widget",  # Unique key for the widget
                        on_change=update_selected_metrics  # Callback to update session state on each change
                    )
                    
                    if selected_metrics_box:
                        if st.button("Train Model") and selected_metrics_box:
                            st.session_state["training_mode"] = True  # Switch to training mode
                            st.rerun()
        else:
            write_helper_text("Please select a target feature to detect the task type.")
    
    else:
        st.write("None")
        write_helper_text("Please select a dataset on the Datasets page.")
else:
    # Ready to Train summary display
    selected_dataset_name = selected_dataset.name if selected_dataset else "None"
    selected_target_feature_object = st.session_state.get('target_feature_object', None)
    selected_target_feature = st.session_state.get('target_feature', None)
    selected_input_features_objects = st.session_state.get('input_features_objects', [])
    selected_input_features = st.session_state.get('input_features', [])
    train_split = st.session_state.get('train_split', 70)
    test_split = 100 - train_split
    train_test_split = f"{train_split}% - {test_split}%"
    selected_model_instance = st.session_state.get('selected_model_instance', 'None')
    selected_model_name = st.session_state.get("selected_model_name", 'None')
    hyperparameters = st.session_state.get(f"selected_model_hyperparameters", {})
    selected_metrics = st.session_state.get("selected_metrics", [])
    task_type = st.session_state.get("task_type", "None")

    st.write("## Overview:")
    st.markdown(f"""
    <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
        <h4 style="color: #4CAF50;">📂 Dataset Configuration</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Dataset:</strong> {selected_dataset_name}</li>
            <li><strong>Train-Test Split:</strong> {train_test_split}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
        <h4 style="color: #4CAF50;">🔍 Feature Selection</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Target Feature:</strong> {selected_target_feature}</li>
            <li><strong>Input Features:</strong> {", ".join(selected_input_features)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
        <h4 style="color: #4CAF50;">🤖 Model Configuration</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Task Type:</strong> {task_type.capitalize()}</li>
            <li><strong>Model:</strong> {selected_model_name}</li>
            <li><strong>Hyperparameters:</strong></li>
            <ul>
                { "".join([f"<li>{key}: {value}</li>" for key, value in hyperparameters.items()]) }
            </ul>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
        <h4 style="color: #4CAF50;">📏 Evaluation Metrics</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Metrics:</strong> {", ".join(selected_metrics)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    pipeline_name = st.text_input("Pipeline Name", value="My_Custom_Pipeline", help="Enter a custom name for the pipeline")
    pipeline_version = st.text_input("Pipeline Version", value="1.0.0", help="Enter a version for the pipeline")

    metrics = [get_metric(metric_name) for metric_name in selected_metrics]
    pipeline = Pipeline(
                metrics=metrics,
                dataset=selected_dataset,
                model=selected_model_instance,
                input_features=selected_input_features_objects,
                target_feature=selected_target_feature_object,
                split=train_split,
                name=pipeline_name, 
                version=pipeline_version
            )
    
    # Button to save the pipeline
    if st.button("Save Pipeline", help="Click to save the pipeline configuration."):
        pipeline.save()

    # Add a "Return" button to go back to the setup page
    if st.button("Return"):
        st.session_state["training_mode"] = False  # Reset training mode to go back to setup page
        st.rerun()  # Refresh the page to show setup content

    # Add a green "Train Model" button to initiate the training process
    if st.button("Train Model", help="Click to start training the model."):
        # Update session state to indicate training is in progress
        st.session_state["is_training"] = True

        # Display a message to indicate training has started
        st.write("🚀 Training the model...")

        # Simulate a delay for training time (replace this with actual training code)
        with st.spinner("Model is training..."):
            # Execute the pipeline
            results = pipeline.execute()

            # Display the results
            st.write("### Training Metrics:")
            for metric, result in results["train_metrics"]:
                st.write(f"{type(metric).__name__}: {result:.4f}")

            st.write("### Test Metrics:")
            for metric, result in results["test_metrics"]:
                st.write(f"{type(metric).__name__}: {result:.4f}")

        # Once training is complete, show a success message
        st.success("✅ Model training complete!")