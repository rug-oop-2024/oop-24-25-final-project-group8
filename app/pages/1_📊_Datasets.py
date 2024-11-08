import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

class DatasetsPage:
    def __init__(self) -> None:
        """Initialize DatasetsPage with the AutoMLSystem instance and dataset artifacts."""
        # Ensure AutoMLSystem is created only once per session using Streamlit's session_state
        if 'automl_system' not in st.session_state:
            st.session_state.automl_system = AutoMLSystem.get_instance()
        
        # initialze session state variables
        if "selected_dataset" not in st.session_state:
            st.session_state["selected_dataset"] = None

        self.automl = st.session_state.automl_system  # Reference the singleton instance

        # Retrieve and filter only dataset artifacts
        self.artifacts = [artifact for artifact in self.automl.registry.list() if artifact.type == "dataset"]

    def reset_training(self) -> None:
        """Reset training-related session state variables."""
        st.session_state["training_mode"] = False
        st.session_state["target_feature"] = None
        st.session_state["input_features"] = []
        st.session_state["target_feature_object"] = None
        st.session_state["input_features_objects"] = []
        st.session_state["selected_model_name"] = None
        st.session_state["selected_model_instance"] = None
        st.session_state["selected_model_hyperparameters"] = {}
        st.session_state["selected_metrics"] = []
        st.session_state["task_type"] = None
        st.session_state["is_training"] = False


    def delete_dataset(self, artifact: Dataset) -> None:
        """
        Deletes the specified dataset's data file and metadata file using the registry's delete method.
        
        Args:
            dataset_id: The ID of the dataset to delete.
        """
        # Use the registry's delete function to remove both the data file and metadata file
        self.automl.registry.delete(artifact)
        self.reset_training()

    def display_datasets(self) -> None:
        """Display available datasets with options to select, show details, and delete."""
        # Display available dataset artifacts
        if self.artifacts:
            for idx, dataset in enumerate(self.artifacts):
        
                dataset_key = f"{dataset.id}_details"

                # Initialize the visibility state for the dataset's details if not set
                if dataset_key not in st.session_state:
                    st.session_state[dataset_key] = False
                
                # Check if this dataset is currently selected
                is_selected = st.session_state.get("selected_dataset") == dataset.id
                row_bg_color = "background-color: lightgreen;" if is_selected else ""
                
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        # Display dataset name and version 
                        st.markdown(
                            f"<div style='{row_bg_color} font-size: 20px; font-style: italic;'> {dataset.name} | {dataset.version}</div>", 
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        # Select Dataset button to highlight and set selected dataset
                        if st.button("Select Dataset", key=f"btn_select_{idx}_{dataset.name}"):
                            # if switching selection clear all session state variables
                            if st.session_state["selected_dataset"] and st.session_state["selected_dataset"] != dataset.id:
                                self.reset_training()
                            st.session_state["selected_dataset"] = dataset.id
                            st.rerun()

                    with col3:
                        # Toggle "Show Details" / "Hide Details" button
                        toggle_text = "Hide Details" if st.session_state[dataset_key] else "Show Details"
                        if st.button(toggle_text, key=f"btn_show_{idx}_{dataset.name}"):
                            st.session_state[dataset_key] = not st.session_state[dataset_key]
                            st.rerun()
                    
                    with col4:
                        # Delete button to remove the dataset
                        if st.button("Remove Dataset", key=f"btn_delete_{idx}_{dataset.name}"):
                            self.delete_dataset(dataset)
                            st.success(f"Dataset '{dataset.name}' has been deleted.")
                            st.rerun()

                    # Display dataset head if "Show Details" is active
                    if st.session_state[dataset_key]:
                        dataset_instance = self.automl.registry.get(dataset.id)
                        if dataset_instance:
                            st.write("Dataset Head:")
                            st.dataframe(dataset_instance.read().head())
        else:
            st.write("No datasets found in the registry.")

    def display_managment(self) -> None:
        """Display dataset management and upload interface."""
        # Display datasets section
        st.header("Dataset Management")
        self.display_datasets()

        # Upload section
        st.header("Upload New Dataset")
        uploaded_file = st.file_uploader("Choose a file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            st.write("Uploaded Dataset Head:")
            st.dataframe(df.head())
            
            dataset_name = st.text_input("Dataset Name")
            dataset_version = st.text_input("Dataset Version", value="1.0.0")
            
            if st.button("Save Dataset"):
                new_dataset = Dataset.from_dataframe(data=df, name=dataset_name, asset_path=f"{dataset_name}.bin", version=dataset_version)
                self.automl.registry.register(new_dataset)
                st.success(f"Dataset '{dataset_name}' has been registered successfully!")
                st.rerun()

# Main execution
if __name__ == "__main__":
    page = DatasetsPage()
    page.display_managment()