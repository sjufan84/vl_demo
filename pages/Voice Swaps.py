""" Demo page for visualizing audio features """ 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import librosa
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from scipy.spatial import distance
from scipy.spatial.distance import cosine
from utils.audio_processing import extract_features, read_audio

st.set_page_config(page_title="Artist Vault Demo", page_icon=":microphone:",
                initial_sidebar_state="expanded", layout="wide")


def get_3d_chart_mv():
    """ Get the 3D chart for Fast Car """
    # Read and preprocess audio signals
    jenny_signal = read_audio('./audio_samples/originals/combs_fcar2.wav')
    joel_clone_signal = read_audio('./audio_samples/clones/jenny_fcar.wav')
    joel_og_signal = read_audio('./audio_samples/clones/joel_fcar.wav')
    min_length = min(len(jenny_signal), len(joel_clone_signal), len(joel_og_signal))
    jenny_signal = jenny_signal[:min_length]
    joel_clone_signal = joel_clone_signal[:min_length]
    joel_og_signal = joel_og_signal[:min_length]    

    # Extract features
    jenny_features = extract_features(jenny_signal)
    joel_clone_features = extract_features(joel_clone_signal)
    joel_og_features = extract_features(joel_og_signal)

    # Transpose to have features as columns
    jenny_features = jenny_features.T
    joel_clone_features = joel_clone_features.T
    joel_og_features = joel_og_features.T

    # Create a DataFrame by concatenating the two feature sets
    df = pd.concat([pd.DataFrame(jenny_features), pd.DataFrame(joel_clone_features),
                    pd.DataFrame(joel_og_features)], axis=0)
    st.dataframe(df)
    st.stop()
    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=35)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(columns = [f'PC{i+1}' for i in range(35)], data = pca_df)

    # Create segments (you may need to adjust this to match your actual segmentation logic)
    jenny_segments = np.array_split(jenny_signal, len(plot_df)//3)
    joel_clone_segments = np.array_split(joel_clone_signal, len(plot_df)//3)
    joel_og_segments = np.array_split(joel_og_signal, len(plot_df)//3)

   # Create labels for the segments
    jenny_labels = [f"Jenny Clone - Segment {i+1}" for i in range(len(jenny_segments))]
    joel_clone_labels = [f"Joel Clone - Segment {i+1}" for i in range(len(joel_clone_segments))]
    joel_og_labels = [f"Joel OG - Segment {i+1}" for i in range(len(joel_og_segments))]
    segment_labels = jenny_labels + joel_clone_labels + joel_og_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(joel_clone_segments))] * 3

    # Add segment names and numbers to the DataFrame
    plot_df['segment_name'] = segment_labels
    plot_df['segment_number'] = segment_numbers
    
    #col1, col2 = st.columns([1.75, 1], gap='large')


    #with col2:# Add a Streamlit multiselect widget to allow users to select artists
    #    st.text("")
    #    st.text("")
    #    st.text("")
        # Display the original clips
        # Convert the clips to bytes using librosa
       
    st.markdown("**Original Audio Clips:**")
    joel_og_bytes = librosa.to_mono(joel_og_signal)
    joel_clone_bytes = librosa.to_mono(joel_clone_signal)
    jenny_bytes = librosa.to_mono(jenny_signal)
    st.markdown("**Joel Clone**")
    st.audio(joel_clone_bytes, format='audio/wav', start_time=0, sample_rate=16000)
    st.markdown("**Joel OG**")
    st.audio(joel_og_bytes, format='audio/wav', start_time=0, sample_rate=16000)
    st.markdown("**Jenny Clone**")
    st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
    selected_artists = st.multiselect(
    "Select Artists to Display:",
    options=['Joel OG', 'Joel Clone', 'Jenny Clone'],
    default=['Joel OG', 'Joel Clone', 'Jenny Clone'],
    )
    # Filter the DataFrame based on selected artists
    filtered_plot_df = plot_df[plot_df['segment_name'].str.contains('|'.join(selected_artists))]

    # Plot using the filtered DataFrame    
    with col1:
        fig = px.scatter_3d(
        filtered_plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='segment_number',
        color_continuous_scale='rainbow',
        title='3D Representation of Melodic Voiceprints -- Joel Original, Joel Clone, Jenny Clone',
        text='segment_name',
    )
        # Make the "segment_name" column the index
        filtered_plot_df.set_index('segment_name', inplace=True)
        # Drop the segment_number column
        filtered_plot_df.drop('segment_number', axis=1, inplace=True)
        # Sort the index by segment number
        # Use regex to strip out the f"Segment {number}" from each index value,
        # sort the index, and then add the f"{ArtistdSegment {number}" back to each index value
        #filtered_plot_df.to_csv('filtered_plot_df.csv')

    # Split the DataFrame into separate DataFrames for each artist
    jenny_clone_df = filtered_plot_df[filtered_plot_df['segment_name'].
                                    str.contains('Jenny Clone')].reset_index(drop=True)
    joel_clone_df = filtered_plot_df[filtered_plot_df['segment_name'].
                                    str.contains('Joel Clone')].reset_index(drop=True)
    joel_og_df = filtered_plot_df[filtered_plot_df['segment_name'].
                                str.contains('Joel OG')].reset_index(drop=True)

    # Initialize a DataFrame to store similarity scores
    similarity_df_full = pd.DataFrame(columns=['Segment', 'Jenny_vs_Joel_Clone',
                                            'Jenny_vs_Joel_OG', 'Joel_Clone_vs_Joel_OG'])

    # Calculate Euclidean distance between each segment for each pair of artists
    for i in range(len(jenny_clone_df)):
        segment = f'Segment {i+1}'
        jenny_coords = jenny_clone_df.loc[i, ['PC' + str(j) for j in range(1, 36)]].to_numpy()
        joel_clone_coords = joel_clone_df.loc[i, ['PC' + str(j) for j in range(1, 36)]].to_numpy()
        joel_og_coords = joel_og_df.loc[i, ['PC' + str(j) for j in range(1, 36)]].to_numpy()
        
        jenny_vs_joel_clone = distance.euclidean(jenny_coords, joel_clone_coords)
        jenny_vs_joel_og = distance.euclidean(jenny_coords, joel_og_coords)
        joel_clone_vs_joel_og = distance.euclidean(joel_clone_coords, joel_og_coords)
        
        similarity_df_full.loc[i] = [segment, jenny_vs_joel_clone,
                                    jenny_vs_joel_og, joel_clone_vs_joel_og]

    # Calculate the maximum distance for each segment
    similarity_df_full['Max_Distance'] = similarity_df_full[['Jenny_vs_Joel_Clone',
                            'Jenny_vs_Joel_OG', 'Joel_Clone_vs_Joel_OG']].max(axis=1)

    # Normalize the similarity scores and convert them into percentages
    for col in ['Jenny_vs_Joel_Clone', 'Jenny_vs_Joel_OG', 'Joel_Clone_vs_Joel_OG']:
        similarity_df_full[f'{col}_Percentage'] = (1 - (similarity_df_full[col] / 
                                            similarity_df_full['Max_Distance'])) * 100

    # Drop the Max_Distance column as it's no longer needed
    similarity_df_full.drop('Max_Distance', axis=1, inplace=True)

    # Initialize a DataFrame to store similarity scores
    similarity_df_cosine = pd.DataFrame(columns=['Segment',
    'Jenny_vs_Joel_Clone', 'Jenny_vs_Joel_OG', 'Joel_Clone_vs_Joel_OG'])
    for i in range(len(jenny_clone_df)):
        try:
            segment = f'Segment {i+1}'
            
            jenny_coords = jenny_clone_df.loc[i,
            ['PC' + str(j) for j in range(1, 36)]].to_numpy()
            joel_clone_coords = joel_clone_df.loc[i,
            ['PC' + str(j) for j in range(1, 36)]].to_numpy()
            joel_og_coords = joel_og_df.loc[i,
            ['PC' + str(j) for j in range(1, 36)]].to_numpy()

            # Subtract from 1 as scipy's function calculates distance
            jenny_vs_joel_clone = 1 - cosine(jenny_coords, joel_clone_coords)  
            jenny_vs_joel_og = 1 - cosine(jenny_coords, joel_og_coords)
            joel_clone_vs_joel_og = 1 - cosine(joel_clone_coords, joel_og_coords)
            
            similarity_df_cosine.loc[i] = [segment, jenny_vs_joel_clone,
                                        jenny_vs_joel_og, joel_clone_vs_joel_og]
        
        except TypeError as e:
            print(f"Error on segment {i+1}: {e}")
            continue

    st.dataframe(similarity_df_full)
    # Make the segment names the index
    similarity_df_cosine.set_index('Segment', inplace=True)
    similarity_df_full.set_index('Segment', inplace=True)
    # Calculate the mean of each column and add it as a final row
    mean_series1 = pd.Series(similarity_df_cosine.mean())
    st.write(mean_series1)
    mean_series2 = pd.Series(similarity_df_full.mean())
    st.write(mean_series2)
    st.dataframe(similarity_df_cosine)  

    # Add a final row to the similarity_df_cosine that displays the mean of each column
    similarity_df_cosine.to_csv('similarity_df_cosine.csv')
    # Create a plot for the cosine similarity scores
    similarity_df_cosine_melted = similarity_df_cosine.melt(id_vars='Segment',
                                var_name='Comparison', value_name='Cosine_Similarity')
    fig = px.bar(
        similarity_df_cosine_melted,
        x='Segment',
        y='Cosine_Similarity',
        color='Comparison',
        color_discrete_sequence=['green', 'red', 'blue'],
        title='Cosine Similarity Scores',
        barmode='group',
        text='Cosine_Similarity',
        height=500,
        width=1000,
    )
    fig.update_layout(
        xaxis=dict(title='Segment'),
        yaxis=dict(title='Cosine Similarity'),
        showlegend=True,
    )
    fig.update_traces(
        textposition='outside',
        texttemplate='%{text:.2f}',
    )
    st.plotly_chart(fig, use_container_width=True)

def get_3d_chart_fcar():
    """ Get the 3D chart for Fast Car """
    # Read and preprocess audio signals
    jenny_signal = read_audio('./audio_samples/clones/jenny_fcar.wav')
    lc_signal = read_audio('./audio_samples/originals/combs_fcar2.wav')
    joel_signal = read_audio('./audio_samples/clones/joel_fcar.wav')
    min_length = min(len(jenny_signal), len(lc_signal), len(joel_signal))
    jenny_signal = jenny_signal[:min_length]
    lc_signal = lc_signal[:min_length]
    joel_signal = joel_signal[:min_length]
    
    # Extract features
    if len(jenny_signal) < 2048 or len(lc_signal) < 2048 or len(joel_signal) < 2048:
        st.warning("One of the audio signals is too short for FFT. Skipping...")
        return

    jenny_features = extract_features(jenny_signal)
    lc_features = extract_features(lc_signal)
    joel_features = extract_features(joel_signal)

    # Transpose to have features as columns
    jenny_features = jenny_features.T
    lc_features = lc_features.T
    joel_features = joel_features.T

    # Create a DataFrame by concatenating the two feature sets
    df = pd.concat([pd.DataFrame(jenny_features), 
    pd.DataFrame(lc_features), pd.DataFrame(joel_features)], axis=0)
    # Standardize the data before applying PCA and KMeans
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # Perform PCA to capture 95% of the variance
    pca = PCA(n_components=3)
    pca_df = pca.fit_transform(scaled_df)

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(pca_df, columns=['PC1', 'PC2', 'PC3'])

    # Create segments (you may need to adjust this to match your actual segmentation logic)
    jenny_segments = np.array_split(jenny_signal, len(plot_df)//3)
    lc_segments = np.array_split(lc_signal, len(plot_df)//3)
    joel_segments = np.array_split(joel_signal, len(plot_df)//3)

   # Create labels for the segments
    jenny_labels = [f"Jenny - Segment {i+1}" for i in range(len(jenny_segments))]
    lc_labels = [f"LC - Segment {i+1}" for i in range(len(lc_segments))]
    joel_labels = [f"Joel - Segment {i+1}" for i in range(len(joel_segments))]
    segment_labels = jenny_labels + lc_labels + joel_labels

    # Create segment numbers (e.g., Segment 1, Segment 2, ...)
    segment_numbers = [f"Segment {i+1}" for i in range(len(lc_segments))] * 3

    # Add segment names and numbers to the DataFrame
    plot_df['segment_name'] = segment_labels
    plot_df['segment_number'] = segment_numbers

    # Create two columns, one for the chart and one for the audio
    col1, col2 = st.columns([1.75, 1], gap='large') 
    with col2:  # Add a Streamlit multiselect widget to allow users to select artists
        st.text("")
        st.text("")
        st.text("")
        # Display the original clips
        # Convert the clips to bytes using librosa
       
        st.markdown("**Original Audio Clips:**")
        joel_bytes = librosa.to_mono(joel_signal)
        lc_bytes = librosa.to_mono(lc_signal)
        jenny_bytes = librosa.to_mono(jenny_signal)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif;
                    color: #3D82FF; font-size: 15px; font-weight: 550;">
                    Jenny Fast Car</p>
                    """, unsafe_allow_html=True)
        st.audio(jenny_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
                    font-size: 15px; font-weight: 550;">
                    Joel Fast Car</p>
                    """, unsafe_allow_html=True)
        st.audio(joel_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        st.markdown("""
        <p style="font-family: 'Montserrat', sans-serif;
                    color: #3D82FF; font-size: 15px; font-weight: 550;">
                    LC Fast Car</p>
                    """, unsafe_allow_html=True)
        st.audio(lc_bytes, format='audio/wav', start_time=0, sample_rate=16000)
        selected_artists = st.multiselect(
        "Select Artists to Display:",
        options=['Jenny', 'LC', 'Joel'],
        default=['Jenny', 'LC', 'Joel'],
        )
        # Filter the DataFrame based on selected artists
        filtered_plot_df = plot_df[plot_df['segment_name'].
                        str.contains('|'.join(selected_artists))]
        st.text("")
        # Create a button to continue to the next page
        continue_secure_button = st.button("Continue to next page",
                        type='secondary', use_container_width=True)
        if continue_secure_button:
            switch_page("Secure")
        st.markdown("---")
        st.markdown("Curious about the cloning process? [Try it out for yourself](https://huggingface.co/spaces/dthomas84/RVC_RULE1)\
                with Jenny and Joel's voices in our First Rule AI playground!")

    with col1:  # Plot using the filtered DataFrame
        fig = px.scatter_3d(
        filtered_plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='segment_number',
        color_continuous_scale='rainbow',
        title='3D Representation of Vocal Features -- LC, Joel, Jenny',
        text='segment_name',
    )

        fig.update_layout(
            width=750,  # Width of the plot in pixels
            height=750,  # Height of the plot in pixels
            scene=dict(
                xaxis=dict(title='PC1'),
                yaxis=dict(title='PC2'),
                zaxis=dict(title='PC3')
            ),
            showlegend=False,
        )
        fig.update_traces(
        textposition='top center',  # Position of the text labels
        textfont_size=10,
        marker_size=8            # Font size of the text labels
    )
        st.plotly_chart(fig, use_container_width=True)

def home():
    """ Home page for the application. Display the mechanism """
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <h4 id='headline' style="font-family: 'Montserrat', sans-serif; color: #3D82FF;
        font-size: 26px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">So what's the problem?</h4>
        <br>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 5s;
        -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s; -o-animation:
        fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">Never before has someone
        been able to clone an artist's voice and then be able to <i>generate new content with it.</i>
        This is not about simply copying and misusing actual recordings.  This is about
        the ability to create vocals virtually <i>indistinguishable</i> from the artist's, and have them
        "sing" literally anything, with no quick way to verify it's authenticity.  The stakes
        in the age of social media, where disinformation spreads like wildfire and very few people
        would even think to check whether or not a voice is authentic, could not be higher.</h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">In order to prove the point, we trained
        models using Joel and Jenny's voices, with less than 10 minutes of data each.  We can then separate
        vocals from Luke Comb's "Fast Car", feed it to our model, and within approximately 30 seconds
        generate a deepfake.  While it isn't perfect, it's important to remember the little amount
        of data we trained Joel and Jenny's models with, and with some fine-tuning and a little more
        training time we could produce whole songs or even albums that could pass for an actual artist's.
        </h3>
        <h3 id='body'style="font-family: 'Montserrat', sans-serif; color: #ecebe4;
        font-size: 17px; font-weight: 550; margin-bottom: -10px; animation: fadeIn ease 3s;
        -webkit-animation: fadeIn ease 5s; -moz-animation: fadeIn ease 8s; -o-animation:
        fadeIn ease 8s; -ms-animation: fadeIn ease 8s;">In the 3d chart below, we have mapped out the high level
            features of each artist's vocals.  You will notice that for each point in the song, the artists' voices
            cluster together, but are not identical.  It is this difference that distinguishes each voiceprint
            and allows for security measures to be taken to protect the artist and their autheticity.</h3>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                -o-animation: fadeIn ease 3s; -ms-animation: fadeIn ease 3s;">
                </div>""", unsafe_allow_html=True)
            
    get_3d_chart_fcar()

if __name__ == "__main__":
    home()
      
    # Display the "Business Chat" sidebar
    from utils.bplan_utils import chat_state_sidebar
    chat_state_sidebar()