import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import whisper
import os

# --- Step 1: Record audio from microphone ---
fs = 16000  # Sample rate
seconds = 5 # Recording duration

print("Recording... Speak now!")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
print("Recording complete!")

# Save as temporary WAV
wav_file = "temp.wav"
write(wav_file, fs, recording)

# --- Step 2: Convert WAV → MP3 ---
mp3_file = "temp.mp3"
AudioSegment.from_wav(wav_file).export(mp3_file, format="mp3")
print(f"Saved recording as {mp3_file}")

# --- Step 3: Load Whisper turbo model and transcribe ---
model = whisper.load_model("turbo")  # or 'base', 'small', etc.
result = model.transcribe(mp3_file)

print("\nTranscription:")
print(result["text"])

if os.path.exists(mp3_file) and os.path.exists(wav_file):
    os.remove(mp3_file)
    os.remove(wav_file)
    print(f"Deleted {mp3_file} and {wav_file}")
