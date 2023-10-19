""" This file contains the upload_audio() function.  This function is used to
upload audio from the user's local machine to be processed by the various 
audio extraction functions. """

# Import required libraries
import streamlit as st
from streamlit_extras.switch_page_button import switch_page


st.text("")
    st.text("")
    # Create two columns, one for uploading, and one for recording
    col1, col2 = st.columns(2, gap="large")
    with col1:
        mic_image = Image.open('./resources/studio_mic1_small.png')
        st.image(mic_image, use_column_width=True)
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        
        col3, col4, col5 = st.columns([0.5, 1.5, 1])
        with col3:
            st.text("")
        with col4:
            st.markdown("Click to record: :green[Stopped] / \
                        :red[Recording]")
        with col5:
            recorded_audio_bytes = audio_recorder(text="", icon_name="record-vinyl",
            sample_rate=16000, neutral_color = "green", icon_size="3x")
        st.text("")
        st.text("")
        st.text("")
        if recorded_audio_bytes:
            st.markdown("""
                        <p style="color:#EDC480; font-size: 23px; text-align: center;">
                        Recorded audio clip:
                        </p>
                        """, unsafe_allow_html=True)
            st.audio(recorded_audio_bytes)
            recorded_audio = librosa.load(io.BytesIO(recorded_audio_bytes))
            # Using soundfile to read the audio file into a NumPy array
            st.session_state.audio_bytes_list.append(recorded_audio)

    with col2:
        upload_image = Image.open('./resources/record_upload1_small.png')
        st.image(upload_image, use_column_width=True)
        st.text("")
        uploaded_file = st.file_uploader("Upload an Audio File", type=['wav'],
                                        key='upload_audio')
        if uploaded_file:
            with io.BytesIO(uploaded_file.getbuffer()) as f:
                # Using soundfile to read the audio file into a NumPy array
                audio = librosa.load(f)
                if audio:
                    st.audio(audio[0], sample_rate=audio[1])
                st.session_state.audio_bytes_list.append(audio)

    st.markdown("""
                **Once you have uploaded and / or recorded your audio,\
                click below to generate your Melodic Voiceprint.**
                """)
    
    generate_mv_button = st.button("Generate Melodic Voiceprint",
    type='primary', use_container_width=True)
    if generate_mv_button:
        if len(st.session_state.audio_bytes_list) == 0:
            st.error("Please upload or record an audio clip.")
        else:# Generate the 3D plot
            with st.spinner("Generating your Melodic Voiceprint..."):
                st.session_state.fig = generate_3d_plot()
                # Switch to the plot page
                st.session_state.secure_page = "secure_plot"
                st.experimental_rerun()

def secure_plot():
    """ Display the user generated Melodic Voiceprint """
    st.markdown("""
    <div style="font-size: 15px">
    <h5> Congratulations!  You have successfully\
    generated your Melodic Voiceprint and it is ready\
    to be secured.  Check out the 3d representation of your\
    Melodic Voiceprint below.  Once we have your MV, we can\
    store the trained model in a secure location so that only you\
    can access it and license it out.</h5>
    <h5>
    When you are done reviewing the plot, secure your MV and proceed\
    to the next step! 
    </h5>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Display the 3D plot
    st.plotly_chart(st.session_state.fig, use_container_width=True)

    secure_button = st.button("Secure Your Voiceprint", type='primary',
    use_container_width=True)
    if secure_button:
        # Count down from 3 to 1 and then display the secure message
        for i in range(3, 0, -1):
            time.sleep(1.5)
            if i == 3:
                st.markdown(f"""
                <div style="font-size: 30px;">
                <h5 style="text-align: center">
                Locking down in {i}
                </h5>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="font-size: 30px;">
                <h5 style="text-align: center">
                {i}
                </h5>
                </div>
                """, unsafe_allow_html=True)
        st.balloons()
        switch_page("Voiceprint Demo")
    create_new_mv_button = st.button("Create a new Melodic Voiceprint",
    type='primary', use_container_width=True)
    if create_new_mv_button:
        st.session_state.audio_bytes_list = []
        st.session_state.secure_page = "secure_home"
        st.experimental_rerun()

            

            
if st.session_state.secure_page == "secure_home":
    secure_home()
elif st.session_state.secure_page == "secure_plot":
    secure_plot()
