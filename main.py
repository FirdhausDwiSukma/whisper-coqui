import sounddevice as sd
from audio.audio_handler import save_audio_to_file, play_audio
from whisperopenai.whisper_handler import transcribe_audio
from coquitts.tts_handler import tts_to_audio

def main():
    print("Silakan berbicara...")
    duration = 5  # Durasi rekaman dalam detik
    samplerate = 16000  # Frekuensi sampel
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()

    # Simpan audio sementara ke file
    audio_file = "temp_audio.wav"
    save_audio_to_file(audio, samplerate, audio_file)

    # Transkripsi menggunakan Whisper
    print("Memproses audio dengan Whisper...")
    text, detected_lang = transcribe_audio(audio_file)

    print(f"Teks transkripsi: {text}")
    print(f"Bahasa terdeteksi: {detected_lang}")

    # Proses TTS
    output_wav = "output_audio.wav"
    tts_to_audio(text, detected_lang, output_wav)

    # Mainkan hasil audio
    play_audio(output_wav)

if __name__ == "__main__":
    main()
