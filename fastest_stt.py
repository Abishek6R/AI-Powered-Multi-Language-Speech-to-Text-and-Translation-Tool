import os
import time
import warnings
import torch
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(self, model_size="large"):
        warnings.filterwarnings("ignore")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Measure the time taken to load the model
        start_time = time.time()
        print("Loading Faster Whisper model...")
        
        # Use mixed precision (float16) by setting compute_type to 'float16'
        self.model = WhisperModel(model_size, device=self.device, compute_type="float16")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds.")

    def transcribe_and_translate(self, audio_file):
        start_time = time.time()
        print("Transcribing and translating audio...")

        # Perform transcription and translation
        segments, info = self.model.transcribe(audio_file, beam_size=2, task="translate")
        transcription = "".join([segment.text for segment in segments])
        
        # Measure the time taken
        process_time = time.time() - start_time
        print(f"Process completed in {process_time:.2f} seconds.")
        
        return {"transcription": transcription, "language": info.language}

if __name__ == "__main__":
    audio_file = "tamiltest.mp3"
    transcriber = WhisperTranscriber(model_size="large")
    
    # Measure total processing time
    total_start_time = time.time()
    result = transcriber.transcribe_and_translate(audio_file)
    total_time = time.time() - total_start_time
    
    print("\n--- Speech-to-Text Results ---")
    print(f"Transcription: {result['transcription']}")
    print(f"Detected Language: {result['language']}")
    print(f"Total time for processing: {total_time:.2f} seconds.")