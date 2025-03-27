import whisper
from moviepy.editor import VideoFileClip

def video_to_text(video_path, output_file="whisper_transcript.txt"):
    # Step 1: Extract audio from video
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")  # WAV format
    
    # Step 2: Load Whisper model
    model = whisper.load_model("base")  # Choose from: tiny, base, small, medium, large
    
    # Step 3: Transcribe audio
    result = model.transcribe(audio_path)
    
    # Step 4: Save transcript
    with open(output_file, "w") as f:
        f.write(result["text"])
    
    print(f"Transcript saved to {output_file}")

# Example usage
if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video file
    video_to_text(video_path)