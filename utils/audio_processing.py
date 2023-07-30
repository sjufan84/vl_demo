""" Audio processing utilities live here. """

import librosa
import librosa.display

def get_waveform_data(wav_file):    
    """ Loads the audio data from librosa
    and displays the waveform. """
    audio, sr = librosa.load(wav_file, sr=None)

    # Return a tuple of the audio data and the sample rate
    return audio, sr

