""" This will be the page that allows the user to interact with
and visualize the different audio features extracted from the audio files. """

import plotly.express as px
import streamlit as st
from utils.visualization import prepare_data_for_plot
from utils.audio_processing import perform_tsne
import pandas as pd



# Prepare data for the plot
st.write(st.session_state.features_data)
df = prepare_data_for_plot(st.session_state.tsne_results)
st.write(df)

calculate_tsne_button = st.button("Calculate t-SNE", type='primary', use_container_width=True)
if calculate_tsne_button:
    # Perform t-SNE
    with st.spinner('Calculating t-SNE...'):
        perform_tsne()
        df = prepare_data_for_plot(st.session_state.tsne_results)
        st.success("t-SNE calculated successfully!")
    st.experimental_rerun()

show_plot = st.button("Show Plot", type='primary', use_container_width=True)
if show_plot:
    # Create the scatter plot
    fig = px.scatter(df, x="Dimension 1", y="Dimension 2", color="File Name",
                     title="t-SNE Visualization of Audio Files",
                     labels={
                         "Dimension 1": "t-SNE Dimension 1",
                         "Dimension 2": "t-SNE Dimension 2",
                         "File Name": "Audio File"
                     })

    # Show the plot
    st.plotly_chart(fig)