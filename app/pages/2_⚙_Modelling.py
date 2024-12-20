import streamlit as st
from typing import List, Literal, get_args

from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.functional.feature_type_detector import FeatureTypeDetector
from autoop.core.ml.model import CLASSIFICATION_MODELS, REGRESSION_MODELS, get_model
from autoop.core.ml.metric import get_metric
from autoop.core.ml.pipeline import Pipeline
import re

# Page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")


class ModelingPage:
    def __init__(self) -> None:
        """
        Initializes an instance of the ModelingPage class,
        which is responsible for managing the modeling configuration
        in the application, including dataset selection, feature configuration,
        model selection, and task type determination.

        Attributes:
            automl (AutoMLSystem): Singleton instance of the
                AutoMLSystem for managing model and dataset resources.
            task_type (str): The type of modeling task to
                be performed (e.g., "classification" or "regression").
            selected_dataset_id (str): ID of the dataset
                selected by the user, retrieved from session state.
            selected_dataset (Dataset): Dataset object
                retrieved based on the selected ID, if available.
            input_features_box (Any): Placeholder for the Streamlit
                component managing input feature selection.
            selected_model_name_box (Any): Placeholder for the
                Streamlit component managing model selection.
            model_instance (Model): Instance of the
                selected model, if available.
            hyperparameters (dict): Dictionary containing
                hyperparameters of the selected model.
            updated_hyperparameters (dict): Dictionary storing
                updated hyperparameters during user modification.
            train_split (float): Train-test split ratio for the
                dataset, as chosen by the user.
        """
        # Access the singleton instance of AutoMLSystem
        self.automl = AutoMLSystem.get_instance()

        # Initialize task_type outside any block to
        # ensure it's accessible in the training summary
        self.task_type = "Unknown"

        # Retrieve the selected dataset ID from session state
        self.selected_dataset_id = st.session_state.get("selected_dataset")

        # Fetch the full dataset object from the registry using the ID
        self.selected_dataset = (
            self.automl.registry.get(self.selected_dataset_id)
            if self.selected_dataset_id
            else None
        )

        self.input_features_box = None
        self.selected_model_name_box = None
        self.model_instance = None
        self.hyperparameters = None
        self.updated_hyperparameters = None
        self.train_split = None

    def write_helper_text(self, text: str) -> None:
        """Display helper text in gray font on the page."""
        st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)

    def display_header(self) -> None:
        """Display page header and a description for the modeling section."""
        # Display page header and description
        st.write("# ⚙ Modelling")
        self.write_helper_text(
            "In this section, you can design a machine"
            "learning pipeline to train a model on a dataset."
        )
        # Display selected dataset information
        st.write("## Dataset Selected:")

    def initialize_session_state(self) -> None:
        """Initialize or reset session state variables for modeling settings."""
        # Initialize session state variables if not already set
        if "training_mode" not in st.session_state:
            st.session_state["training_mode"] = (
                False  # Tracks if we're in 'training' mode
            )
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

    def display_dataset(self) -> None:
        """Display selected dataset information,
        including a toggle for viewing dataset details."""
        # Set the key for the Show/Hide Details button state
        details_key = "selected_dataset_details"

        # Initialize the session state for the Show/Hide Details toggle if not set
        if details_key not in st.session_state:
            st.session_state[details_key] = False

        if self.selected_dataset:
            # FETCHING THE SELECTED DATASET
            col1, col2 = st.columns([2, 1])

            # Display dataset information in col1
            with col1:
                st.markdown(
                    f"<div style='font-size: 20px; font-style: italic;'>"
                    f"{self.selected_dataset.name} | "
                    f"{self.selected_dataset.version}</div>",
                    unsafe_allow_html=True,
                )

            # Show/Hide Details button in col2
            with col2:
                toggle_text = (
                    "Hide Details" if st.session_state[details_key] else "Show Details"
                )
                if st.button(toggle_text, key="btn_show_selected_dataset"):
                    st.session_state[details_key] = not st.session_state[details_key]
                    st.rerun()  # Refresh page to update the display immediately

            # Display dataset head if "Show Details" is active
            if st.session_state[details_key]:
                dataset_instance = self.automl.registry.get(self.selected_dataset.id)
                if dataset_instance:
                    st.write("Dataset Head:")
                    st.dataframe(dataset_instance.read().head())
        else:
            st.write("None")
            self.write_helper_text("Please select a dataset on the Datasets page.")

    def display_features(self) -> None:
        """Display target and input feature selection
        options and update task type accordingly."""
        # Detect features using the FeatureTypeDetector
        features: List[Feature] = FeatureTypeDetector(
            self.selected_dataset
        ).detect_feature_types()

        # Separate feature names
        feature_names = [feature.name for feature in features]

        st.write("### Feature Selection")

        previous_target = st.session_state.get("target_feature")

        if (
            "target_feature" not in st.session_state
            or st.session_state["target_feature"] not in feature_names
        ):
            st.session_state["target_feature"] = feature_names[0]
            st.session_state["target_feature_object"] = next(
                feature for feature in features if feature.name == feature_names[0]
            )

        def update_target_features():
            # Update session state for the target
            # feature and the corresponding feature object
            st.session_state["target_feature"] = st.session_state[
                "target_feature_widget"
            ]
            st.session_state["target_feature_object"] = next(
                feature
                for feature in features
                if feature.name == st.session_state["target_feature_widget"]
            )
            # If the target feature has changed,
            # reset the input features and other settings
            if previous_target != st.session_state["target_feature"]:
                st.session_state["input_features"] = []
                st.session_state["input_features_objects"] = []
                st.session_state["selected_metrics"] = []
                st.session_state["selected_model_name"] = None
                st.session_state["selected_model_instance"] = None
                st.session_state["selected_model_hyperparameters"] = {}

        # Use st.session_state['target_feature'] to set the default selection
        target_feature_box = st.selectbox(
            "Select Target Feature",
            feature_names,
            index=(
                feature_names.index(st.session_state["target_feature"])
                if st.session_state["target_feature"] in feature_names
                else 0
            ),
            key="target_feature_widget",
            on_change=update_target_features,
        )

        # Filter out the selected target feature from input features list
        input_features_options = [
            feature
            for feature in feature_names
            if feature != st.session_state.get("target_feature")
        ]

        # Filter previously selected input features
        # to ensure they exist in the current options
        filtered_default_input_features = [
            feature
            for feature in st.session_state["input_features"]
            if feature in input_features_options
        ]

        if not filtered_default_input_features:
            print("")

        def update_input_features():
            # Set selected input feature names in session state
            selected_feature_names = st.session_state["input_features_widget"]
            st.session_state["input_features"] = selected_feature_names
            # Map feature names to Feature objects and store them in session state
            st.session_state["input_features_objects"] = [
                feature
                for feature in features
                if feature.name in selected_feature_names
            ]

        # Use a temporary variable to hold selected input features
        self.input_features_box = st.multiselect(
            "Select Input Features",
            options=[
                f for f in feature_names if f != st.session_state["target_feature"]
            ],
            default=st.session_state["input_features"],
            key="input_features_widget",
            on_change=update_input_features,
        )

        # Display task type based on the selected target feature's type
        if target_feature_box:
            # Find the feature type for task detection
            target_feature_type = next(
                (
                    feature.type
                    for feature in features
                    if feature.name == target_feature_box
                ),
                None,
            )

            # Determine task type based on the target feature's type
            if target_feature_type == "numerical":
                self.task_type = "regression"
            elif target_feature_type == "categorical":
                self.task_type = "classification"
            else:
                self.task_type = "Unknown"

            st.write(f"**Detected Task Type:** {self.task_type}")

            st.session_state["task_type"] = self.task_type

    def display_model(self) -> None:
        """Provide model selection dropdown based on task type and clear any
        previous hyperparameters."""
        st.write("## Select Model")

        # Model selection dropdown based on task type
        model_options = (
            CLASSIFICATION_MODELS
            if self.task_type == "classification"
            else REGRESSION_MODELS
        )

        if "selected_model_widget" not in st.session_state:
            # Set it to the first option in model_options by default
            st.session_state["selected_model_widget"] = model_options[0]

        # Callback function to update the selected
        # model and clear previous hyperparameters
        def update_selected_model():
            # Clear any existing hyperparameters in the session state
            if "selected_model_name" in st.session_state:
                prev_model_name = st.session_state["selected_model_name"]
                if f"{prev_model_name}_hyperparameters" in st.session_state:
                    del st.session_state[f"{prev_model_name}_hyperparameters"]

            # Update the selected model name in session state
            st.session_state["selected_model_widget"] = st.session_state[
                "selected_model_box"
            ]

        self.selected_model_name_box = st.selectbox(
            "Choose a model",
            model_options,
            index=(
                model_options.index(st.session_state["selected_model_widget"])
                if "selected_model_widget" in st.session_state
                and st.session_state["selected_model_widget"] in model_options
                else 0
            ),
            key="selected_model_box",
            on_change=update_selected_model,
        )

        # Instantiate the model if selected
        if self.selected_model_name_box:
            self.model_instance = get_model(self.selected_model_name_box)
            st.session_state["selected_model_instance"] = self.model_instance
            st.session_state["selected_model_name"] = self.model_instance.name

            # Ensure hyperparameters are initialized once in session state
            if (
                f"{self.selected_model_name_box}_hyperparameters"
                not in st.session_state
            ):
                st.session_state[f"{self.selected_model_name_box}_hyperparameters"] = (
                    self.model_instance.hyperparameters
                )

            # Reference the hyperparameters directly in session state
            self.hyperparameters = st.session_state[
                f"{self.selected_model_name_box}_hyperparameters"
            ]
            st.session_state["selected_model_hyperparameters"] = self.hyperparameters

    def display_hyperparameters(self) -> None:
        """Display adjustable hyperparameters for
        the selected model with constraints."""
        st.write("#### Set Hyperparameters")

        # Create input fields for each hyperparameter, directly using session state keys
        for param, value in self.hyperparameters.items():
            # Get Pydantic field constraints from metadata
            field = self.model_instance.__fields__[param]

            min_value = None
            max_value = None

            EPSILON = 1e-10
            pattern = r"(gt|ge|lt|le)=([0-9]+)(\.*[0-9]+)*"
            for item in field.metadata:
                matches = re.findall(pattern, str(item))
                # Iterate through found matches
                for constraint, int_value, decimal_value in matches:
                    # Combine integer and decimal parts
                    full_value = float(
                        int_value + (decimal_value if decimal_value else "")
                    )

                    # Set min_value and max_value based on the constraints
                    if constraint == "gt":
                        min_value = full_value + EPSILON
                    elif constraint == "ge":
                        min_value = full_value  # ge means greater than or equal to
                    elif constraint == "lt":
                        max_value = full_value - EPSILON
                    elif constraint == "le":
                        max_value = full_value  # le means less than or equal to

            # Check if the field type is a Literal
            # and extract allowed values for dropdown
            field_type = field.annotation
            literal_values = (
                get_args(field_type)
                if hasattr(field_type, "__origin__")
                and field_type.__origin__ is Literal
                else None
            )

            # If the field is Literal, render a dropdown with allowed values
            if literal_values:
                st.selectbox(
                    f"{param}",
                    options=literal_values,
                    index=literal_values.index(
                        st.session_state.get(
                            f"{self.selected_model_name_box}_{param}", value
                        )
                    ),
                    key=f"{self.selected_model_name_box}_{param}",
                )
            # Determine the widget based on the type and handle None defaults
            elif value is None:
                # Generate the range based on min and
                # max values for fields with None default
                if min_value is not None and max_value is not None:
                    options = [None] + list(range(int(min_value), int(max_value) + 1))
                elif min_value is not None:
                    options = [None] + list(
                        range(int(min_value), int(min_value) + 20)
                    )  # Fallback range if only min_value is defined
                elif max_value is not None:
                    options = [None] + list(
                        range(int(max_value) - 20, int(max_value) + 1)
                    )  # Fallback range if only max_value is defined
                else:
                    options = [None] + list(
                        range(1, 21)
                    )  # General fallback if no constraints are defined

                # Display a selectbox with `None` as an option
                st.selectbox(
                    f"{param} (Optional)",
                    options=options,
                    format_func=lambda x: "None" if x is None else x,
                    index=(
                        0
                        if st.session_state.get(
                            f"{self.selected_model_name_box}_{param}", value
                        )
                        is None
                        else options.index(
                            st.session_state.get(
                                f"{self.selected_model_name_box}_{param}", value
                            )
                        )
                    ),
                )
            elif isinstance(value, bool):
                # Checkbox for boolean values
                st.checkbox(
                    param,
                    value=st.session_state.get(
                        f"{self.selected_model_name_box}_{param}", value
                    ),
                    key=f"{self.selected_model_name_box}_{param}",
                )
            elif isinstance(value, int):
                # Render with constraints; if out of range,
                # adjust to the nearest boundary
                input_value = st.session_state.get(
                    f"{self.selected_model_name_box}_{param}", value
                )
                adjusted_value = (
                    max(
                        (
                            min(input_value, max_value)
                            if max_value is not None
                            else input_value
                        ),
                        min_value,
                    )
                    if min_value is not None
                    else input_value
                )
                st.number_input(
                    param,
                    min_value=int(min_value) if min_value is not None else None,
                    max_value=int(max_value) if max_value is not None else None,
                    value=int(adjusted_value),
                    step=1,
                    key=f"{self.selected_model_name_box}_{param}",
                )
            elif isinstance(value, float):
                # Render with constraints; if out of range,
                # adjust to the nearest boundary
                input_value = st.session_state.get(
                    f"{self.selected_model_name_box}_{param}", value
                )
                adjusted_value = (
                    max(
                        (
                            min(input_value, max_value)
                            if max_value is not None
                            else input_value
                        ),
                        min_value,
                    )
                    if min_value is not None
                    else input_value
                )
                st.number_input(
                    param,
                    min_value=float(min_value) if min_value is not None else None,
                    max_value=float(max_value) if max_value is not None else None,
                    value=float(adjusted_value),
                    step=0.01,
                    key=f"{self.selected_model_name_box}_{param}",
                )
            elif isinstance(value, str):
                st.text_input(
                    param,
                    value=st.session_state.get(
                        f"{self.selected_model_name_box}_{param}", value
                    ),
                    key=f"{self.selected_model_name_box}_{param}",
                )

        # Update the main hyperparameters dictionary with new values from widgetas
        self.updated_hyperparameters = {
            param: st.session_state[f"{self.selected_model_name_box}_{param}"]
            for param in self.hyperparameters.keys()
        }
        # Assign the updated dictionary back to the main
        # hyperparameters in session state
        st.session_state[f"{self.selected_model_name_box}_hyperparameters"] = (
            self.updated_hyperparameters
        )

    def display_split(self) -> None:
        """Provide slider for selecting train-test
        split percentage and display chosen split."""
        # Step: Train-Test Split Selection
        st.write("#### Select Train-Test Split")

        def update_split():
            # Directly set selected_metrics in session state based on widget state
            st.session_state["train_split"] = st.session_state["split_widget"]

        # Use a slider to let the user choose the percentage for the training set
        st.session_state["train_split"] = st.slider(
            "Select Train Split Percentage",
            min_value=1,
            max_value=99,
            value=st.session_state["train_split"],
            key="split_widget",
            step=1,
            help="Percentage of data for training; remainder for testing.",
            on_change=update_split,
        )

        # Calculate the test split automatically
        self.train_split = st.session_state["train_split"]
        test_split = 100 - self.train_split

        # Display the selected split
        st.write(f"Training set: {self.train_split}%, Testing set: {test_split}%")

    def display_metrics(self) -> None:
        """Display options to select evaluation metrics based on task type."""
        # Step: Select Metrics Step
        # Filter metrics based on task type
        if self.task_type == "classification":
            compatible_metrics = ["accuracy", "precision", "recall", "f1_score"]
        elif self.task_type == "regression":
            compatible_metrics = [
                "mean_squared_error",
                "root_mean_squared_error",
                "mean_absolute_error",
            ]

        st.write("### Select Evaluation Metrics")

        # Define a callback function to update session
        # state directly for selected metrics
        def update_selected_metrics():
            # Directly set selected_metrics in session state based on widget state
            st.session_state["selected_metrics"] = st.session_state["metrics_widget"]

        # Multiselect widget with an on_change callback for metrics selection
        selected_metrics_box = st.multiselect(
            "Select Evaluation Metrics",
            options=compatible_metrics,
            default=st.session_state[
                "selected_metrics"
            ],  # Initialize with current session state
            key="metrics_widget",  # Unique key for the widget
            on_change=update_selected_metrics,
        )

        if selected_metrics_box:
            if st.button("Train Model") and selected_metrics_box:
                st.session_state["training_mode"] = True  # Switch to training mode
                st.rerun()

    def fetch_session_variables(self) -> None:
        """Retrieve session variables needed for training summary display."""
        # Ready to Train summary display
        self.selected_dataset_name = (
            self.selected_dataset.name if self.selected_dataset else "None"
        )
        self.selected_target_feature_object = st.session_state.get(
            "target_feature_object", None
        )
        self.selected_target_feature = st.session_state.get("target_feature", None)
        self.selected_input_features_objects = st.session_state.get(
            "input_features_objects", []
        )
        self.selected_input_features = st.session_state.get("input_features", [])
        self.train_split = st.session_state.get("train_split", 70)
        self.test_split = 100 - self.train_split
        self.train_test_split = f"{self.train_split}% - {self.test_split}%"
        self.selected_model_instance = st.session_state["selected_model_instance"]
        self.selected_model_name = st.session_state.get("selected_model_name", "None")
        self.hyperparameters = st.session_state.get(
            "selected_model_hyperparameters", {}
        )
        self.selected_metrics = st.session_state.get("selected_metrics", [])
        self.task_type = st.session_state.get("task_type", "None")

    def display_training_summary(self) -> None:
        """Display an overview of dataset, features, model,
        hyperparameters, and metrics chosen for training."""
        st.write("## Overview:")
        st.markdown(
            f"""
        <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
            <h4 style="color: #4CAF50;">📂 Dataset Configuration</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li><strong>Dataset:</strong> {self.selected_dataset_name}</li>
                <li><strong>Train-Test Split:</strong> {self.train_test_split}</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
            <h4 style="color: #4CAF50;">🔍 Feature Selection</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li><strong>Target Feature:</strong> {self.selected_target_feature}</li>
                <li><strong>Input Features:</strong>
                {", ".join(self.selected_input_features)}</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Generate the hyperparameters list HTML separately
        hyperparameters_list = "".join(
            [f"<li>{key}: {value}</li>" for key, value in self.hyperparameters.items()]
        )

        # Define the main HTML content with the shorter line length
        st.markdown(
            f"""
            <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
                <h4 style="color: #4CAF50;">🤖 Model Configuration</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li><strong>Task Type:</strong> {self.task_type.capitalize()}</li>
                    <li><strong>Model:</strong> {self.selected_model_name}</li>
                    <li><strong>Hyperparameters:</strong></li>
                    <ul>
                        {hyperparameters_list}
                    </ul>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px;">
            <h4 style="color: #4CAF50;">📏 Evaluation Metrics</h4>
            <ul style="list-style-type: none; padding-left: 0;">
                <li><strong>Metrics:</strong> {", ".join(self.selected_metrics)}</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def display_pipeline(self) -> None:
        """Provide options to name and save the pipeline, start
        training, and display training metrics upon completion."""
        pipeline_name = st.text_input(
            "Pipeline Name",
            value="My_Custom_Pipeline",
            help="Enter a custom name for the pipeline",
        )
        pipeline_version = st.text_input(
            "Pipeline Version", value="1.0.0", help="Enter a version for the pipeline"
        )

        metrics = [get_metric(metric_name) for metric_name in self.selected_metrics]
        pipeline = Pipeline(
            metrics=metrics,
            dataset=self.selected_dataset,
            model=self.selected_model_instance,
            input_features=self.selected_input_features_objects,
            target_feature=self.selected_target_feature_object,
            split=self.train_split,
            name=pipeline_name,
            version=pipeline_version,
        )

        # Button to save the pipeline
        if st.button("Save Pipeline", help="Click to save the pipeline configuration."):
            pipeline.save()

        # Add a "Return" button to go back to the setup page
        if st.button("Return"):
            st.session_state["training_mode"] = (
                False  # Reset training mode to go back to setup page
            )
            st.rerun()  # Refresh the page to show setup content

        # Add a green "Train Model" button to initiate the training process
        if st.button("Train Model", help="Click to start training the model."):
            # Update session state to indicate training is in progress
            st.session_state["is_training"] = True

            # Display a message to indicate training has started
            st.write("🚀 Training the model...")

            # Simulate a delay for training time
            # (replace this with actual training code)
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

    def render(self) -> None:
        """Render the modeling page by displaying dataset, features,
        model selection, hyperparameters, and training options."""
        self.display_header()
        self.initialize_session_state()
        if not st.session_state["training_mode"]:
            self.display_dataset()
            if self.selected_dataset:
                self.display_features()
                if self.input_features_box:
                    self.display_model()
                    if self.model_instance:
                        self.display_hyperparameters()
                        if self.updated_hyperparameters:
                            self.display_split()
                            if self.train_split:
                                self.display_metrics()
        else:
            self.fetch_session_variables()
            self.display_training_summary()
            self.display_pipeline()


# Main execution
if __name__ == "__main__":
    page = ModelingPage()
    page.render()
