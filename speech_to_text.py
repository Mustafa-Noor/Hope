import os
import json
import time
import wave
import pyaudio
import keyboard  # pip install keyboard
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Audio recording settings
TEMP_WAV_FILE = "temp_audio.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def record_until_key_release(filename=TEMP_WAV_FILE):
    print("🎙️ Press and hold Enter to record...")
    while not keyboard.is_pressed('enter'):
        time.sleep(0.1)

    print("⏺️ Recording... Release Enter to stop.")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    while keyboard.is_pressed('enter'):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ Recording stopped.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_with_groq(filename):
    with open(filename, "rb") as file:
        response = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="en",
            temperature=0.0
        )
        return response.text

if __name__ == "__main__":
    print("🎤 Real-Time Speech-to-Text (Hold Enter to Speak)\nPress Ctrl+C to exit.")
    try:
        while True:
            record_until_key_release()
            text = transcribe_with_groq(TEMP_WAV_FILE)
            print(f"📝 You said: {text}\n")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
