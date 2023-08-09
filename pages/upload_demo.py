""" Demo upload page for walk-through """
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from PIL import Image

st.warning('**The first step in the process is capturing an artist\'s voice and then passing\
                it through multiple alogorithms to generate the MV.  Ideally this would be high\
                quality recordings of their vocals that capture a full range of pitches\
                rhythms, tempos, etc.  However even with very crude samples it is possible to create\
                embeddings models that can accurately identify potential copyright issues and generate\
                basic voice clones as we will see shortly.**')
# Display the studio mic icon and record audio buttons
col1, col2 = st.columns(2)
with col1:
    upload_image = Image.open('./resources/studio_mic1_small.png')
    st.image(upload_image)
with col2:
    record_image = Image.open('./resources/record_upload1_small.png')
    st.image(record_image)

# Create the "visualize data" button
visualize_data_button = st.button("Visualize Data", type='primary', use_container_width=True)
if visualize_data_button:
    st.session_state.visual_page='visual_home'
    # If the user wants to visualize data, switch to the visualize_data page
    switch_page("visualize_data")
