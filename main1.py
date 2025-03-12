import librosa
import soundfile as sf
from faster_whisper import WhisperModel

class AudioToText:
    """Converts an MP3 file to text using Faster Whisper."""

    def __init__(self, model_size="medium.en", device="cpu", compute_type="int8", language="en"):
        """Initialize the Whisper model for transcription."""
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        print("✅ Whisper model loaded successfully!")

    def transcribe_audio(self, file_path):
        """Transcribe an MP3 (or other format) audio file to text."""
        print(f"🎙️ Loading audio file: {file_path}")

        # Load and convert the audio file
        audio, sr = librosa.load(file_path, sr=16000)  # Converts audio to 16kHz sample rate

        print("📝 Transcribing audio file...")
        segments, _ = self.model.transcribe(audio, beam_size=5, language=self.language, vad_filter=True)

        # Combine transcribed segments
        transcription = "\n".join([segment.text.strip() for segment in segments])

        print("\n✅ Full Transcription:\n", transcription)
        return transcription

# 🎯 Usage Example
if __name__ == "__main__":
    transcriber = AudioToText()

    # Ask user for MP3 file path
    file_path = input("📂 Enter the path to your MP3 file: ").strip().strip('"')

    transcriber.transcribe_audio(file_path)
