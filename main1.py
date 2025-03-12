import librosa
import gradio as gr
from faster_whisper import WhisperModel

class AudioToText:
    """Converts an MP3 file to text using Faster Whisper."""

    def __init__(self, model_size="medium.en", device="cpu", compute_type="int8", language="en"):
        """Initialize the Whisper model for transcription."""
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe_audio(self, file_path):
        """Transcribe an MP3 audio file to text."""
        audio, sr = librosa.load(file_path, sr=16000)
        segments, _ = self.model.transcribe(audio, beam_size=5, language=self.language, vad_filter=True)
        transcription = "\n".join([segment.text.strip() for segment in segments])
        return transcription

# Create an instance of the transcription model
transcriber = AudioToText()

# Define the Gradio function
def transcribe_gradio(file_path):
    return transcriber.transcribe_audio(file_path)

# Create Gradio interface
demo = gr.Interface(
    fn=transcribe_gradio,
    inputs=gr.Audio(type="filepath"),  # ✅ Fixed: Changed "file" to "filepath"
    outputs="text",
    title="Speech-to-Text Whisper Demo",
    description="Upload an MP3 file to transcribe it to text using Faster Whisper.",
)

# Launch the demo
demo.launch(share=True)
