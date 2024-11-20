from TTS.api import TTS
import os

# Inisialisasi model TTS
# tts_id = TTS("tts_models/multilingual/multi-dataset/your_indonesian_model")
# tts_en = TTS("tts_models/multilingual/multi-dataset/your_english_model")

tts_id = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
tts_en = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")

def tts_to_audio(text, detected_lang, output_wav):
    """
    Mengubah teks menjadi audio menggunakan model TTS yang sesuai.
    """
    if detected_lang == "id":
        speaker_id = tts_id.speakers[0]
        language_id = tts_id.languages[0]
        tts_id.tts_to_file(text, speaker=speaker_id, language=language_id, file_path=output_wav)
    elif detected_lang == "en":
        language_id = tts_en.languages[0]
        tts_en.tts_to_file(text, language=language_id, file_path=output_wav)
    else:
        raise ValueError("Bahasa tidak didukung.")
