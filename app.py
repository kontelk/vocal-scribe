import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip

def video_to_text(video_path, output_file="transcript.txt"):
    
    try:
        # Step 1: Extract audio from video
        video = VideoFileClip(video_path)
        audio_path = "audio.aac" # audio path
        video.audio.write_audiofile(audio_path, codec="aac")
        
        # Step 2: Load Whisper model
        model = whisper.load_model("small")  # Choose from: tiny, base, small, medium, large
        
        # Step 3: Transcribe audio
        result = model.transcribe(audio_path,
            fp16=False,           # Use fp16 or fp32
            task="transcribe",   # Choose between "transcribe" and "translate"
            language="en",       # Optional language hint
            # temperature=0.2,     # Balance speed/accuracy
            # beam_size=3          # Moderate beam search
            # best_of=5,           # Best of N sampling
            # word_timestamps=True, # Get word timestamps
            # initial_prompt="Hello", # Initial prompt
            # suppress_tokens="-1", # Suppress tokens
            # condition_on_previous_text=True, # Condition on previous text
            # without_timestamps=False, # Without timestamps
            # word_timestamps=True, # Word timestamps
            # no_speech_threshold=0.6, # No speech threshold
            # language_detection=True, # Language detection
        )
        
        # Step 4: Save transcript
        with open(output_file, "w") as f:
            f.write(result["text"])
        
        print(f"Transcript saved to {output_file}")

    except Exception as e:
        print(f"Exception - An error occurred: {e}")


if __name__ == "__main__":
    video_path = "2025-03-27 10-55-31.mp4"  
    video_to_text(video_path)