""" This file contains the code for the Visualize Data page of the Streamlit app. """
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
#from utils.audio_processing import get_spectrogram

# Returned features for reference:
#return {"signal": signal, "stft": stft, "fbanks": fbanks, "mfccs": 
# mfccs, "delta1": delta1, "delta2": delta2, "cw": cw, "norm": norm}

#speaker_data = {}
# Create the speaker data from the st.session_state.features dictionary
#for i, key in enumerate(st.session_state.features.keys()):
    # Convert the keys to Speaker {i} format
#    speaker_data[f"Speaker {i}"] = st.session_state.features[key]

# Load the pickle file that contains the speaker data
with open('speaker_data.pkl', 'rb') as f:
    speaker_data = pickle.load(f)

def visualize_waveform_data(wavform_data):
    """ Visualize waveform data """
    # For each audio file, display the waveform
    # Allow the user to choose from a multi-select box of the audio files
    # Display the waveform for each audio file
    wave_form_selection = st.multiselect("Select audio files", list(wavform_data.keys()))
    for audio_file in wave_form_selection:
        st.write(f"Waveform for {audio_file}")
        st.line_chart(wavform_data[audio_file])

def visualize_fbanks_data():
    """ Visualize filter bank data """
    # Introduce the user to filter banks
    st.markdown("#### Visualizing filter banks")
    st.markdown('---')
    st.markdown("**The 3D chart produced by the above code represents the FBanks features of an audio signal,\
                where the x-axis, y-axis, and z-axis represent the frame index, the filter bank index,\
                and the corresponding magnitude respectively.**")
    st.markdown("**Frame Index (x-axis):** This represents the time dimension.\
                Each frame is a short snippet of sound (usually around 20-30 milliseconds long) and the audio signal\
                is divided into these frames for analysis.")
    st.markdown("**Filter Bank Index (y-axis):** This represents different\
                frequency bands. Filter banks are essentially a set of band-pass filters that divide the frequency\
                spectrum into 'bins'. Each index corresponds to a different frequency band that the FBanks feature\
                extraction method focuses on.")
    st.markdown("**Magnitude (z-axis):** The magnitude at a given (frame, filter bank)\
                point indicates the strength of the audio signal in that particular filter bank's frequency band during\
                that particular frame. This captures the frequency characteristics of the sound at that moment in time.")
    st.markdown("**Taken together, this 3D chart visualizes how the frequency content of the audio signal changes over time,\
                capturing the unique 'frequency fingerprints' of the sound or voice in the audio clip.**")
    
    # For each audio file, display the filter bank data
    # Allow the user to choose from a multi-select box speakers
    # Display the filter bank data for each speaker
    # Get the list of speakers
    speakers_list = list(speaker_data.keys())
    # Allow the user to select the speaker
    fbanks_selection = st.multiselect("Select speaker", speakers_list)
    # Display the fbanks chart for the selected speaker
    for speaker in fbanks_selection:
        fbanks = speaker_data[speaker]['fbanks']
        # To visualize the FBanks features, we can average them across the channels.
        fbanks_avg = torch.mean(fbanks, dim=-1).squeeze().numpy()

        # Create x, y coordinates for the plot. Assuming the number of FBanks and frames are in variables num_fbanks and num_frames.
        x = list(range(fbanks_avg.shape[1]))
        y = list(range(fbanks_avg.shape[0]))

        # Create a meshgrid for the coordinates.
        X, Y = np.meshgrid(x, y)

        # Create a 3D surface plot.
        fig = go.Figure(data=[go.Surface(z=fbanks_avg, x=X, y=Y)])

        fig.update_layout(title=f'FBanks Features for {speaker} ', autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90))

        # Display the figure using Streamlit.
        st.plotly_chart(fig)

def visualize_embeddings():
    """ Visualize embeddings """
    st.markdown("""**The audio files we are working with are transformed into what we call 'embeddings'. 
                Think of these embeddings as unique fingerprints for each speaker's voice. They capture
                the unique characteristics of a person's speech such as tone, pitch, accent, and more.
                By analyzing these embeddings, we can differentiate speakers, even when they're saying
                the same words. This unique 'voice print' can be utilized in various applications, including
                speaker recognition, personalized voice assistants, and even in security for voice-based
                authentication systems.**""")
    
    st.markdown('---')

    st.markdown("""**The 3D plot you see below is a visual representation of these voice prints. 
                Each point in the plot is a unique voice print extracted from an audio file. 
                The closer the points, the more similar the voice characteristics they represent. 
                This is achieved using a technique called 'Principal Component Analysis' or PCA, 
                which simplifies the complex data while retaining the essential differences between voices. 
                The colors represent different clusters or groups of similar voices identified
                by our model.**""")
    
    # Read in the pca csv file
    df_pca = pd.read_csv('pca.csv')
    # Create a 3D scatter plot for PCA
    fig_pca = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='label', hover_name='names', 
                            color_continuous_scale="viridis", title="PCA Analysis of Speaker Embeddings")  # display filename on hover
    
    st.plotly_chart(fig_pca)

async def generate_embeddings_df():
    """ Generate embeddings DataFrame """
    # 1. Flatten embeddings and create list of labels
    flat_data = []
    labels = []
    # Get the list of names of the files
    names = list(st.session_state.embeddings.keys())

    for filename, embeddings in st.session_state.embeddings.items():
        flat_data.append(embeddings)
        labels.append(filename)
    stacked_data = np.vstack(flat_data)

    # 2. Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(stacked_data)

    # 3. Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(normalized_data)

    # 4. Perform KMeans clustering on PCA results
    kmeans = KMeans(n_clusters=3)  # adjust as needed
    labels = kmeans.fit_predict(pca_result)

    # 5. Create a DataFrame for PCA visualization
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
    df_pca['label'] = labels  
    df_pca['names'] = names  # use filenames as labels

    # Export the pandas dataframe to a csv file
    df_pca.to_csv('pca.csv', index=False)

def visualize_spectrogram_data():
    """ Visualize the spectrogram data stored in the speaker data dictionary """
    # Brief explanation of spectrograms
    st.markdown(
    """
    ## Spectrogram:

    A **spectrogram** is like a map of sound. It takes a piece of music and transforms it into a colorful image that allows us to "see" the sound. It shows us how the "ingredients" of the sound (the frequencies) change over time. 

    On the x-axis of this image is time, like the timeline you'd see on a music player. On the y-axis, you have different frequencies, think of them like musical notes, from low to high. The color of each point in the image tells us how strong that "note" is at each moment in the song. 

    """
    )
    
    # Allow the user to select the speaker(s) to visualize
    # Get the list of speakers
    speakers_list = list(speaker_data.keys())
    # Allow the user to select the speaker
    spectrogram_selection = st.selectbox("Select speaker", speakers_list)
    # Display the spectrogram chart for the selected speaker
    
    if spectrogram_selection == "Speaker 0":
        image = Image.open("./images/spectrograms/spectrogram0.png")
        st.image(image)
    elif spectrogram_selection == "Speaker 1":
        image = Image.open("./images/spectrograms/spectrogram1.png")
        st.image(image)
    else: 
        image = Image.open("./images/spectrograms/spectrogram2.png")
        st.image(image)

    st.warning("""**Note:** While a spectrogram can help us see the sound,
                it doesn't quite match how we humans "hear" the sound.
                Our ears don't hear all frequencies equally. We are more sensitive
                to certain frequencies, and we perceive changes in frequency on
                a logarithmic scale (meaning, we can tell the difference between low frequencies better than high ones).
                This is where Mel-filterbanks, or simply fbanks, come in. They take the same information as in a spectrogram
                but adjust the frequency scale to better match human hearing. If you would
                like to see a visual representation of these filter banks, select "Filter Banks" from the sidebar menu.""")
    
    #spectrogram = speaker_data[speaker]['spectrogram']
    # Converting tensor to numpy for plotly
    #spectrogram = spectrogram.numpy()
    # Create a heatmap with plotly
    #fig = go.Figure(data=go.Heatmap(z=spectrogram,
    #                                colorscale='Viridis',
    #                                zmin=0, zmax=1))
    #fig.update_layout(title=f'Spectrogram for {speaker} ', autosize=False,
    #                xaxis_title='Time',
    #                yaxis_title='Frequency',
    #                )
    # In streamlit, use the plotly_chart function to display the plotly figure
    #st.plotly_chart(fig)

def visualize_mfccs_data():
    """ Visualize the MFCCs data stored in the speaker data dictionary """
    st.markdown(
    """
    ## MFCCs Plot:

    Think of a piece of music as a delicious cake. The MFCCs plot is like the recipe for the cake. 

    The x-axis represents time (like the steps in the recipe), and the y-axis represents different MFCC coefficients (like the different parts of the cake). The color at each point tells us the value of each MFCC coefficient at each point in time. This allows us to see not just what frequencies are present in the music, but also how these frequencies are combined and change over time.

    """
    )
    # Create the list of speakers
    speakers_list = list(speaker_data.keys())
    # Allow the user to select the speaker
    mfccs_selection = st.selectbox("Select speaker", speakers_list)
    if mfccs_selection == "Speaker 0":
        mfccs = speaker_data['Speaker 0']['mfccs']
    elif mfccs_selection == "Speaker 1":
        mfccs = speaker_data['Speaker 1']['mfccs']
    else:
        mfccs = speaker_data['Speaker 2']['mfccs']
    # Display the MFCCs chart for the selected speaker
    mfccs = torch.sqrt((mfccs**2).sum(-1))  # Compute the magnitude if the last dimension represents a complex number
    mfccs = mfccs.squeeze(0)  # Remove the first dimension of size 1

    # Now, mfccs should have shape [8269, 20], which you can transpose and visualize
    mfccs = mfccs.t()
    # Convert tensor to numpy for plotly
    mfccs = mfccs.numpy()

    # Create a heatmap with plotly
    fig = go.Figure(data=go.Heatmap(z=mfccs,
                                    colorscale='Hot',
                                    zmin=0, zmax=1))
    fig.update_layout(title='MFCCs',
                    xaxis_title='Time',
                    yaxis_title='Frequency')

    # Display the plotly figure in streamlit
    st.plotly_chart(fig)

def app():
    """ Visualize data home """
    # Brief explanation of the different features and embeddings
    st.markdown("""**Voice prints, similar to fingerprints, are unique to each individual
    and can be extracted using various voice feature representations such
    as MFCCs, STFTs, and FBanks. These methods translate the raw complexity
    of speech into more manageable data formats. MFCCs, or Mel Frequency
    Cepstral Coefficients, focus on the human auditory system's perception
    of sound. STFTs, or Short-Time Fourier Transforms, provide a dynamic
    spectrum analysis of the signal, revealing its frequency components over
    time. FBanks, or Filter Banks, offer frequency-based representations, capturing
    crucial elements of the speech pattern. By combining these different analyses,
    our app is able to capture a detailed 'voice print' - a unique auditory signature
    that allows for precise speaker identification and voice recognition.**""")
    st.markdown('---')
    st.markdown("""**We have prepared visualizations of the different features and embeddings
                which are illustrative of the unique characteristics of each speaker's voice.
                Select an option below to get started.**""")

# Create sidebar buttons to allow the user to select which visualization to view
st.sidebar.markdown('**Select a visualization:**')
# Create a sidebar with radio buttons for the user to select which visualization to view
visual_choice = st.sidebar.radio('Visualizations', ['Home', 'Embeddings', 'Filter Banks', 'MFCCs',
                                        'Spectrograms', 'Normalized Features'])
if visual_choice == 'Home':
    st.session_state.visual_page = 'visual_home'
elif visual_choice == 'Embeddings':
    st.session_state.visual_page = 'embeddings'
elif visual_choice == 'Filter Banks':
    st.session_state.visual_page = 'fbanks'
elif visual_choice == 'MFCCs':
    st.session_state.visual_page = 'mfccs'
elif visual_choice == 'Spectrograms':
    st.session_state.visual_page = 'spectrogram'
elif visual_choice == 'Normalized Features':
    st.session_state.visual_page = 'norm_features'

# App flow
if st.session_state.visual_page == 'visual_home':
    app()
elif st.session_state.visual_page == 'embeddings':
    visualize_embeddings()
elif st.session_state.visual_page == 'fbanks':
    visualize_fbanks_data()
elif st.session_state.visual_page == 'mfccs':
    visualize_mfccs_data()
elif st.session_state.visual_page == 'spectrogram':
    visualize_spectrogram_data()
#elif st.session_state.visual_page == 'norm_features':
#    visualize_norm_features_data()'''
