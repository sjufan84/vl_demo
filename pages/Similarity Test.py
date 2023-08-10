import streamlit as st
from speechbrain.pretrained import SpeakerRecognition

def main():
    st.title('WAV File Comparison')
    st.markdown('---')
    st.markdown('#### Upload two wav files to compare their similarity.\
                The model will respond with a verdict and a probability or\
                confidence score.  Smaller clips are more efficient and\
                should still provide accurate results.')
    st.text("")
    # Create a file uploader and a button to run the model
    uploaded_files = st.file_uploader("Upload two audio files to compare", accept_multiple_files=True, type=['wav'])
    upload_files_button = st.button("Compare")

    if len(uploaded_files) < 2:
        st.warning("Please upload two audio files to compare.")
    elif upload_files_button:
        # Load the model
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

        # Save the files to temporary storage and get their paths
        file1_path = f"temp_file1.wav"
        file2_path = f"temp_file2.wav"
        with open(file1_path, 'wb') as f1:
            f1.write(uploaded_files[0].getvalue())

        with open(file2_path, 'wb') as f2:
            f2.write(uploaded_files[1].getvalue())

        # Compute the similarity using the 'verify_files' method
        score, prediction = verification.verify_files(file1_path, file2_path)

        # Display the results
        st.warning('**Note:** A score closer to 1 indicates that the two audio files are from the same speaker,\
                    while a score closer to 0 indicates that the two audio files are from different speakers.\
                    Thus the closer to one or closer to 0 a score is, the higher the probability that the\
                    result is correct.')
        if prediction[0]:
            st.markdown(f"The two audio files are from the **same artist**.")
            st.markdown(f"The confidence score is: **{score[0]:.2f}**")
        else:
            st.markdown(f"The two audio files are from **different artists**.")
            st.markdown(f"The confidence score is: **{score[0]:.2f}**")
        

if __name__ == "__main__":
    main()



    