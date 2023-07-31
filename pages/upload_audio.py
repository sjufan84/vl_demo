""" This file contains the upload_audio() function.  This function is used to
upload audio from the user's local machine to be processed by the various 
audio extraction functions. """

# Import required libraries
from PIL import Image
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from sklearn.cluster import KMeans
from utils.audio_processing import get_waveform_data, extract_features, perform_kmeans_clustering, perform_tsne

if 'progress' not in st.session_state:
    st.session_state.progress = 0


def upload_audio():
    """ Upload audio from the user's local machine. """
    # Display the upload mechanism
    # Use PIL to read in the image
    upload_image = Image.open('./resources/record_upload1.png')
    st.image(upload_image)

    st.warning('**Click on the button below to upload a file from your\
            local machine.  This is the second step / option to grab a sample of\
            the speakers voice.  This sample should be in .wav format**')
    uploaded_file = st.file_uploader("Upload a file", type="wav")
    file_name  = st.text_input("Enter a name for the recording:")
    if uploaded_file:
        upload_files_button = st.button("Upload File", type='primary', use_container_width=True)
        if upload_files_button:
            with st.spinner('Processing file...'):
                # Save the file to the session state
                with open(uploaded_file.name, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                # Get the file
                file = uploaded_file.name
                # Save the file to the session state
                st.session_state.audio_files[file_name] = file

                # Update the waveform data
                st.session_state.waveform_data[file_name] = get_waveform_data(file)
                # Extract the features from the audio file
                features = extract_features(file)
                st.session_state.features_data[file_name] = features
                # Use Kmeans to cluster the features
                st.session_state.cluster_data[file_name] = perform_kmeans_clustering(features, 2)
                # Use tnse to reduce the dimensionality of the data
                st.session_state.tsne_results[file_name] = perform_tsne(features)

                # Once the file is uploaded, display buttons for the user to
                # select the next step
                st.success("File uploaded and processed successfully.\
                          What would you like to do next?")
    # Display the buttons
    upload_more_button = st.button("Upload more files", type='primary', use_container_width=True)
    if upload_more_button:
        # If the user wants to upload more files, reset the progress bar
        st.experimental_rerun()
    record_audio_button = st.button("Record Audio", type='primary', use_container_width=True)
    if record_audio_button:
        # If the user wants to record audio, switch to the record_audio page
        switch_page("record_audio")
        st.experimental_rerun()
    visualize_data_button1 = st.button("Visualize Data", type='primary', use_container_width=True)
    if visualize_data_button1:
        # If the user wants to visualize data, switch to the visualize_data page
        switch_page("visualize_data")

if st.session_state.upload_audio_page == 'upload_audio':
    upload_audio()