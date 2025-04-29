# tcprealtimewhisper.py
import socket
import threading
import time
import struct
import numpy as np
from queue import Queue

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

class AudioProcessor:
    def __init__(self, vad_threshold=0.5):
        # Audio and VAD Configuration
        self.RATE = 16000
        self.SAMPLE_WIDTH = 2
        self.FRAMES_PER_BUFFER = 24
        self.FINISHED_MARGIN = 0.5

        self.VAD_THRESHOLD = vad_threshold              
        self.VAD_NEG_THRESHOLD = None         
        self.VAD_MIN_SPEECH_DURATION_MS = 0   
        self.VAD_MAX_SPEECH_DURATION_S = float("inf")  
        self.VAD_MIN_SILENCE_DURATION_MS = 500  
        self.VAD_SPEECH_PAD_MS = 40

        # Rolling buffer and thread lock
        self.rolling_buffer = bytearray()
        self.buffer_lock = threading.Lock()

        # Output queue for transcription text
        self.transcribed_text_queue = Queue()

        # TCP server configuration for audio
        self.TCP_IP = "0.0.0.0"
        self.TCP_PORT = 5006

        # Initialize models
        self.vad_model = load_silero_vad()
        self.whisper_model = WhisperModel("turbo", device="cuda", compute_type="float16")

    def tcp_audio_receiver(self):
        """Receives audio packets and appends them to the rolling buffer."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow immediate reuse


        server_sock.bind((self.TCP_IP, self.TCP_PORT))
        server_sock.listen(1)
        print(f"[Audio Receiver] Listening on {self.TCP_IP}:{self.TCP_PORT}...")

        while True:
            conn, addr = server_sock.accept()
            print(f"[Audio Receiver] Connection from {addr}")
            try:
                while True:
                    header = b""
                    while len(header) < 4:
                        chunk = conn.recv(4 - len(header))
                        if not chunk:
                            raise ConnectionError("Connection closed.")
                        header += chunk
                    packet_size = int.from_bytes(header, "big")

                    audio_data = b""
                    while len(audio_data) < packet_size:
                        chunk = conn.recv(packet_size - len(audio_data))
                        if not chunk:
                            raise ConnectionError("Connection closed.")
                        audio_data += chunk

                    with self.buffer_lock:
                        self.rolling_buffer.extend(audio_data)
            except Exception as e:
                print("[Audio Receiver] Error:", e)
            finally:
                conn.close()
                print("[Audio Receiver] Client disconnected.")

    def process_audio_buffer(self):
        """Processes audio in the rolling buffer, applies VAD, and transcribes segments."""
        while True:
            time.sleep(0.1)
            with self.buffer_lock:
                if len(self.rolling_buffer) < self.FRAMES_PER_BUFFER * self.SAMPLE_WIDTH:
                    continue
                audio_bytes = bytes(self.rolling_buffer)

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(audio_np) / self.RATE

            try:
                speech_segments = get_speech_timestamps(
                    audio_np,
                    self.vad_model,
                    return_seconds=True,
                    threshold=self.VAD_THRESHOLD,
                    neg_threshold=self.VAD_NEG_THRESHOLD,
                    min_speech_duration_ms=self.VAD_MIN_SPEECH_DURATION_MS,
                    max_speech_duration_s=self.VAD_MAX_SPEECH_DURATION_S,
                    min_silence_duration_ms=self.VAD_MIN_SILENCE_DURATION_MS,
                    speech_pad_ms=self.VAD_SPEECH_PAD_MS,
                )
            except Exception as e:
                print("[Transcription] VAD error:", e)
                continue

            finished_segments = [seg for seg in speech_segments if seg["end"] < (duration - self.FINISHED_MARGIN)]
            if not finished_segments:
                continue

            last_processed_time = 0.0
            for seg in finished_segments:
                seg_start, seg_end = seg["start"], seg["end"]
                if seg_end <= last_processed_time:
                    continue

                start_sample = int(seg_start * self.RATE)
                end_sample = int(seg_end * self.RATE)
                segment_audio = audio_np[start_sample:end_sample]

                #print(f"[Transcription] Transcribing {seg_start:.2f}-{seg_end:.2f}s...")
                segments_gen, info = self.whisper_model.transcribe(segment_audio, beam_size=5, language='en')
                transcription_text = ""
                for s in segments_gen:
                    segment_str = f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}"
                    #print(segment_str)
                    transcription_text += segment_str + "\n"
                self.transcribed_text_queue.put(transcription_text)
                last_processed_time = seg_end

            remove_seconds = finished_segments[-1]["end"]
            remove_bytes = int(remove_seconds * self.RATE * self.SAMPLE_WIDTH)
            with self.buffer_lock:
                self.rolling_buffer = self.rolling_buffer[remove_bytes:]

    def start_audio_processing(self, vad_threshold=None):
        """
        Starts the TCP audio receiver and transcription threads.
        Optionally override the VAD threshold.
        """
        if vad_threshold is not None:
            self.VAD_THRESHOLD = vad_threshold

        threading.Thread(target=self.tcp_audio_receiver, daemon=True).start()
        threading.Thread(target=self.process_audio_buffer, daemon=True).start()
        print("[Audio Module] Audio processing started.")

# Allow module testing independently
if __name__ == "__main__":
    processor = AudioProcessor()
    # Optionally, you can pass a different VAD threshold:
    # processor.start_audio_processing(vad_threshold=0.6)
    processor.start_audio_processing(vad_threshold=0.8)
    print("[Audio Module] Running standalone test. Awaiting audio input...")
    try:
        while True:
            if not processor.transcribed_text_queue.empty():
                transcription = processor.transcribed_text_queue.get()
                print("[Standalone Test] Transcription:\n", transcription)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Audio Module] Test terminated.")
