import numpy as np
import sounddevice as sd
from pydub import AudioSegment

def save_audio_to_file(audio, samplerate, file_path):
    """
    Simpan audio ke file WAV.
    """
    from scipy.io.wavfile import write
    write(file_path, samplerate, (audio * 32767).astype(np.int16))

def play_audio(file_path):
    """
    Memutar file audio WAV.
    """
    # Baca file WAV
    audio = AudioSegment.from_file(file_path, format="wav")
    # Konversi ke array NumPy
    samples = np.array(audio.get_array_of_samples())
    # Jika stereo, ubah menjadi format (samples, channels)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    # Mainkan audio
    print(f"Memainkan file audio: {file_path}")
    sd.play(samples, samplerate=audio.frame_rate)
    sd.wait()
