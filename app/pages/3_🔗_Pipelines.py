import streamlit as st
import pandas as pd
from typing import List

from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.model import Model
from autoop.core.ml.metric import Metric
import os

# Ensure AutoMLSystem is created only once per session using Streamlit's session_state
if 'automl_system' not in st.session_state:
    st.session_state.automl_system = AutoMLSystem.get_instance()

automl = st.session_state.automl_system  # Reference the singleton instance

# Retrieve and filter only pipeline artifacts
artifacts = [artifact for artifact in automl.registry.list() if artifact.type == "pipeline"]

# Initialize session state variables
if "selected_pipeline" not in st.session_state:
    st.session_state["selected_pipeline"] = None

def delete_pipeline(artifact: Pipeline):
    """
    Deletes the specified pipeline artifact using the registry's delete method.
    
    Args:
        artifact: The pipeline artifact to delete.
    """
    automl.registry.delete(artifact)

def display_pipeliness():
    if artifacts:
        for idx, artifact in enumerate(artifacts):
            pipeline_key = f"{artifact.id}_details"

            if pipeline_key not in st.session_state:
                st.session_state[pipeline_key] = False
            
            # Check if this pipeline is currently selected
            is_selected = st.session_state.get("selected_pipeline") == artifact.id
            row_bg_color = "background-color: lightgreen;" if is_selected else ""

            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(
                        f"<div style='{row_bg_color} font-size: 20px; font-style: italic;'> {artifact.name} | {artifact.version}</div>", 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    if st.button("Select pipeline", key=f"btn_select_{idx}_{artifact.name}"):
                        st.session_state["selected_pipeline"] = artifact.id
                        st.rerun()

                with col3:
                    toggle_text = "Hide Details" if st.session_state[pipeline_key] else "Show Details"
                    if st.button(toggle_text, key=f"btn_show_{idx}_{artifact.name}"):
                        st.session_state[pipeline_key] = not st.session_state[pipeline_key]
                        st.rerun()
                
                with col4:
                    if st.button("Remove pipeline", key=f"btn_delete_{idx}_{artifact.name}"):
                        automl.registry.delete(artifact)
                        st.success(f"Pipeline '{artifact.name}' has been deleted.")
                        st.rerun()

                if st.session_state[pipeline_key]:
                    # Load the pipeline from the pickle file
                    pipeline_path = os.path.join("assets", "objects", f"{artifact.name}v{artifact.id}.pkl")
                    try:
                        pipeline_instance = Pipeline.load(pipeline_path)

                        # Use HTML for a styled, compact layout
                        details_html = f"""
                        <div style="padding: 10px; border: 1px solid #e6e6e6; border-radius: 5px; font-size: 14px;">
                            <h4 style="color: #4CAF50; margin-bottom: 5px;">Pipeline Details</h4>
                            <p style="margin: 5px 0;"><strong>Dataset:</strong> {pipeline_instance._dataset.name if pipeline_instance._dataset else 'N/A'}</p>
                            <p style="margin: 5px 0;"><strong>Split:</strong> {pipeline_instance._split}%</p>
                            <p style="margin: 5px 0;"><strong>Model:</strong> {pipeline_instance._model.name}</p>
                            <p style="margin: 5px 0;"><strong>Input Features:</strong> {', '.join([feat.name for feat in pipeline_instance._input_features])}</p>
                            <p style="margin: 5px 0;"><strong>Target Feature:</strong> {pipeline_instance._target_feature.name if pipeline_instance._target_feature else 'N/A'}</p>
                            <p style="margin: 5px 0;"><strong>Metrics:</strong> {', '.join([type(metric).__name__ for metric in pipeline_instance._metrics])}</p>
                        </div>
                        """
                        st.markdown(details_html, unsafe_allow_html=True)

                    except FileNotFoundError:
                        st.error(f"Could not find pipeline file at {pipeline_path}")
                    except Exception as e:
                        st.error(f"An error occurred while loading the pipeline: {e}")
    else:
        st.write("No pipelines found in the registry.")

# Display datasets section
st.header("Dataset Management")
display_pipeliness()