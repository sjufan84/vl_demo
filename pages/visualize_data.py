""" This will be the page that allows the user to interact with
and visualize the different audio features extracted from the audio files. """

# Import the required libraries
import librosa
import librosa.display
import plotly.graph_objects as go
import streamlit as st

# We will create columns for the different audio features
# For now the only visualization is for the waveform data
# For each value in the waveform_data dictionary, we will
# add the data to a plot for the user to visualize
def display_waveform_data():
    """ Function to display the waveform data.  
    Takes the waveform data and plots it. """
    # Check to see if there is any data in the waveform_data dictionary
    if st.session_state.waveform_data:
        fig = go.Figure()
        # Create a multi-select box for the user to select which
        # waveform they would like to view
        waveform_selection = st.multiselect("Select a waveform to view:", list(st.session_state.waveform_data.keys()))
        # For each waveform selected, display the waveform
        for waveform in waveform_selection:
            # Create a plotly figure
            # Add the waveform data to the figure
            fig.add_trace(go.Scatter(y=st.session_state.waveform_data[waveform][0], mode='lines', name=waveform))
            # Add a title to the figure
            fig.update_layout(title='Waveform Data')
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
    #else:
    #    st.warning("No waveform data found. Please upload some audio files.")

# Create the columns for the page
#col1, col2 = st.columns(2)

display_waveform_data() # Display the waveform data

def display_feature_embeddings
