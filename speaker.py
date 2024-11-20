from TTS.api import TTS

# Inisialisasi model TTS
tts_id = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")  # Model untuk Bahasa Indonesia
tts_en = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA")  # Model untuk Bahasa Inggris

# Cetak daftar speaker
print("Speakers untuk model Indonesia (tts_id):", tts_id.speakers)
print("Speakers untuk model Inggris (tts_en):", tts_en.speakers)
