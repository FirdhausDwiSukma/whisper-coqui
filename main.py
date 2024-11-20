import whisper
import sounddevice as sd
import numpy as np
import wave
from TTS.api import TTS
from pydub import AudioSegment

# Inisialisasi Whisper model
model = whisper.load_model("small")

# Inisialisasi Coqui TTS model untuk Indonesia dan Inggris
tts_id = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
tts_en = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")

# Fungsi untuk menyimpan audio ke file WAV
def save_audio_to_file(audio, samplerate, file_path):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Audio mono
        wf.setsampwidth(2)  # 16-bit (2 bytes per sample)
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())

# Fungsi untuk merekam suara
def record_audio(duration=5, samplerate=16000):
    print("Mulai merekam... Silakan berbicara.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Tunggu sampai rekaman selesai
    print("Rekaman selesai.")
    return audio.flatten()

# Fungsi untuk memproses audio dan menghasilkan output
def save_as_mp3(input_wav, output_mp3):
    """
    Mengkonversi file WAV ke MP3 menggunakan pydub.
    """
    # Buka file WAV
    sound = AudioSegment.from_wav(input_wav)
    # Simpan sebagai MP3
    sound.export(output_mp3, format="mp3")
    print(f"File MP3 disimpan: {output_mp3}")

def play_audio(file_path):
    """
    Memutar file audio WAV menggunakan sounddevice.
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

def process_audio(audio):
    # Simpan sementara audio ke file untuk diproses Whisper
    audio_file = "temp_audio.wav"
    save_audio_to_file(audio, 16000, audio_file)
    
    # Transkripsi menggunakan Whisper
    print("Memproses audio dengan Whisper...")
    result = model.transcribe(audio_file)
    text = result["text"]
    detected_lang = result["language"]
    
    print(f"Teks transkripsi: {text}")
    print(f"Bahasa terdeteksi: {detected_lang}")
    
    # Pilih model TTS berdasarkan bahasa
    if detected_lang == "id":
        print("Menggunakan TTS untuk Bahasa Indonesia.")
        # Bersihkan daftar speaker
        cleaned_speakers = [s.strip() for s in tts_id.speakers]
        speaker_id = cleaned_speakers[0]  # Pilih speaker pertama
        language_id = tts_id.languages[0]  # Pilih bahasa pertama (Indonesia)
        tts_id.tts_to_file(text, speaker=speaker_id, language=language_id, file_path="output_id.wav")
        save_as_mp3("output_id.wav", "output_id.mp3")
        play_audio("output_id.wav")
    elif detected_lang == "en":
        print("Menggunakan TTS untuk Bahasa Inggris.")
        language_id = tts_en.languages[0]  # Pilih bahasa pertama (Inggris)
        tts_en.tts_to_file(text, language=language_id, file_path="output_en.wav")
        save_as_mp3("output_en.wav", "output_en.mp3")
        play_audio("output_en.wav")
    else:
        print("Bahasa tidak didukung. Tidak ada output TTS.")
        return
    
# Main loop program
if __name__ == "__main__":
    while True:
        print("\nTekan Enter untuk merekam (atau ketik 'exit' untuk keluar):")
        command = input().strip()
        if command.lower() == "exit":
            break
        audio_data = record_audio(duration=5)
        process_audio(audio_data)