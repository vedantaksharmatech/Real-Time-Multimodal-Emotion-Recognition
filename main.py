import whisper
from src.realtime.speech_to_text import record_audio  # your current working import

# Load Whisper model
model = whisper.load_model("small")

def transcribe_audio(file_path, lang="en"):
    """
    Transcribe audio to text using Whisper.
    lang = "en" for English
    """
    result = model.transcribe(file_path, language=lang)
    return result["text"]

if __name__ == "__main__":
    print("\n🎤 Starting audio recording...")
    # Record audio (5 seconds)
    audio_file = record_audio(duration=5)
    
    print("\n📝 Transcribing audio to English...\n")
    
    # English transcription
    english_text = transcribe_audio(audio_file, lang="en")
    
    # Display nicely
    print("="*50)
    print("📌 Transcription Results")
    print("="*50)
    print("\n🟢 English:\n")
    print(english_text)
    print("="*50)
    print("\n✅ Done!\n")

    # # Hindi transcription (commented out for now)
    # hindi_text = transcribe_audio(audio_file, lang="hi")
    # print("\n🔴 Hindi:\n")
    # print(hindi_text)