import whisper

# Inisialisasi model Whisper
model = whisper.load_model("small")

def transcribe_audio(file_path):
    """
    Transkripsi audio menggunakan Whisper.
    """
    result = model.transcribe(file_path, task="transcribe")
    text = result["text"]
    detected_lang = result["language"]
    return text, detected_lang
