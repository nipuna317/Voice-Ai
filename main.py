import asyncio
import os
import sys
import threading

import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY is not set.")
    print("Create a .env file and add:")
    print("GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)


MODEL = "gemini-3.1-flash-live-preview"

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK_SIZE = 2048

CHANNELS = 2

# Smaller = lower latency, but more callbacks
BLOCK_SIZE = 1024


# ============================================================
# Gemini client
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# Audio queues
# ============================================================

audio_input_queue = asyncio.Queue()
audio_output_queue = asyncio.Queue()


# ============================================================
# Microphone
# ============================================================

mic_stream = None
output_stream = None


def microphone_callback(indata, frames, time_info, status):
    """
    Called by PortAudio from another thread.
    """

    if status:
        print(f"\n[Microphone] {status}", file=sys.stderr)

    audio_bytes = bytes(indata)

    loop = microphone_callback.loop

    loop.call_soon_threadsafe(
        audio_input_queue.put_nowait,
        audio_bytes
    )


# Will be assigned when asyncio starts.
microphone_callback.loop = None


# ============================================================
# Speaker
# ============================================================
def start_audio_output():
    global output_stream

    output_stream = sd.RawOutputStream(
        samplerate=24000,
        blocksize=2048,
        channels=1,
        dtype="int16",
        latency="low",
    )

    output_stream.start()
    
def play_audio(audio_bytes):
    """
    Play 24 kHz, mono, signed 16-bit PCM.
    """

    if output_stream is not None:
        output_stream.write(audio_bytes)


# ============================================================
# Send microphone -> Gemini
# ============================================================

async def send_microphone_audio(session):
    print("🎤 Microphone started.")
    print("Speak Sinhala normally...\n")

    while True:
        audio_chunk = await audio_input_queue.get()

        await session.send_realtime_input(
            audio=types.Blob(
                data=audio_chunk,
                mime_type="audio/pcm;rate=16000",
            )
        )


# ============================================================
# Receive Gemini -> Speaker
# ============================================================

async def receive_gemini_audio(session):
    while True:

        async for response in session.receive():

            if not response.server_content:
                continue

            content = response.server_content

            # ------------------------------------------------
            # User speech transcription
            # ------------------------------------------------

            if content.input_transcription:

                text = content.input_transcription.text

                if text:
                    print(f"\n🧑 You: {text}")

            # ------------------------------------------------
            # Gemini speech transcription
            # ------------------------------------------------

            if content.output_transcription:

                text = content.output_transcription.text

                if text:
                    print(f"🤖 Gemini: {text}")

            # ------------------------------------------------
            # Gemini native audio
            # ------------------------------------------------

            if content.model_turn:

                for part in content.model_turn.parts:

                    if part.inline_data:

                        audio_data = part.inline_data.data

                        if audio_data:
                            play_audio(audio_data)


# ============================================================
# Main
# ============================================================

async def main():

    global microphone_callback

    loop = asyncio.get_running_loop()

    microphone_callback.loop = loop

    # --------------------------------------------------------
    # Live API configuration
    # --------------------------------------------------------

    
    config = {
    "response_modalities": ["AUDIO"],

    "input_audio_transcription": {},
    "output_audio_transcription": {},

    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "leda"
            }
        }
    },

    "system_instruction": """
You are a friendly female AI assistant speaking with a Sri Lankan user.

Speak primarily in Sinhala.

Voice style:
- Use a natural, warm female voice.
- Speak clearly and smoothly.
- Use natural Sri Lankan Sinhala pronunciation.
- Speak at a moderate, comfortable speed.
- Use natural pauses between sentences.
- Sound friendly, calm and human-like.
- Avoid sounding robotic or overly dramatic.
- Do not speak too fast.
- Avoid unnecessary English words.
- When English words are necessary, pronounce them naturally.

Conversation style:
- Be friendly and casual.
- Understand Sinhala-English mixed speech.
- Reply naturally as if having a real conversation.
"""
}

    print("=" * 60)
    print("       🇱🇰 Sinhala Gemini Live Assistant")
    print("=" * 60)
    print()
    print("Connecting to Gemini Live API...")

    try:

        async with client.aio.live.connect(
            model=MODEL,
            config=config,
        ) as session:

            print("✅ Connected!")
            print()
            print("🎧 Gemini is listening.")
            print("🗣️ Speak Sinhala.")
            print("❌ Press Ctrl+C to stop.")
            print()

            # ------------------------------------------------
            # Start microphone
            # ------------------------------------------------

            mic_stream = sd.RawInputStream(
                samplerate=INPUT_SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=CHANNELS,
                dtype="int16",
                callback=microphone_callback,
            )

            mic_stream.start()

            # ------------------------------------------------
            # Start speaker
            # ------------------------------------------------

            start_audio_output()

            # ------------------------------------------------
            # Run microphone sender and Gemini receiver
            # simultaneously
            # ------------------------------------------------

            sender_task = asyncio.create_task(
                send_microphone_audio(session)
            )

            receiver_task = asyncio.create_task(
                receive_gemini_audio(session)
            )

            try:

                await asyncio.gather(
                    sender_task,
                    receiver_task,
                )

            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        print("\n\nStopping...")

    except Exception as e:
        print("\n❌ ERROR:")
        print(type(e).__name__, str(e))

    finally:

        if mic_stream is not None:
            try:
                mic_stream.stop()
                mic_stream.close()
            except Exception:
                pass

        if output_stream is not None:
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass

        print("\n👋 Assistant stopped.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n👋 Bye!")