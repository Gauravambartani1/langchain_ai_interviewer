# tts_module.py (rewritten for streaming)

import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# --- Configuration ---
# Choose a larger model for better accuracy. 'small' or 'medium' are good
# trade-offs between speed and quality. 'large-v3' is the most accurate.
# If you have a GPU, change device="cpu" to device="cuda" for much faster performance.
model = WhisperModel("base", device="cpu", compute_type="int8")


samplerate = 16000
block_duration = 0.5  # seconds
chunk_duration = 2    # seconds
channels = 1
frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)
audio_queue = queue.Queue()
audio_buffer = []
stream_active = threading.Event()
stream_active.set() # Set to active initially

def _audio_callback(indata, frames, time, status):
    """This function is called for each audio block."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def start_recording_stream():
    """Starts the audio recording thread."""
    if not stream_active.is_set():
        stream_active.set()
    
    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        callback=_audio_callback,
        blocksize=frames_per_block
    ) as stream:
        while stream_active.is_set():
            sd.sleep(100)

def stop_recording_stream():
    """Stops the audio recording thread."""
    stream_active.clear()

def transcribe_stream(duration_s: int = 10):
    """
    Transcribes a live audio stream for a specified duration.
    This function should be called after `start_recording_stream`.
    """
    audio_buffer = []
    frames_to_record = int(samplerate * duration_s)

    while len(audio_buffer) < frames_to_record:
        try:
            # Wait for audio blocks
            block = audio_queue.get(timeout=0.1)
            audio_buffer.extend(block.flatten())
        except queue.Empty:
            continue
    
    audio_data = np.array(audio_buffer[:frames_to_record], dtype=np.float32)

    # Transcription using a larger beam size for better accuracy
    segments, _ = model.transcribe(
        audio_data,
        language="en",
        beam_size=5,
        vad_filter=True  # Use voice activity detection to filter out silence
    )

    return " ".join(seg.text for seg in segments)

# --- Entry point for the Streamlit app ---
# This helper function simplifies the interaction for the app.py file
def transcribe_from_input(file_obj) -> str:
    """
    Transcribes a complete audio file-like object (e.g., from st.audio_input)
    using improved model settings.
    """
    import soundfile as sf
    from io import BytesIO

    # Check if the input is a file object or BytesIO
    file_obj.seek(0)
    audio_data, _ = sf.read(BytesIO(file_obj.read()), dtype="float32")
    
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    segments, _ = model.transcribe(
        audio_data,
        language="en",
        beam_size=5,  # Increased beam size for better accuracy
        vad_filter=True # Filters out silent parts for cleaner transcription
    )
    
    return " ".join(seg.text for seg in segments)