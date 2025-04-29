import socket
import pyaudio
import time

class AudioStreamer:
    def __init__(self, dest_ip, dest_port, chunk=5, format=pyaudio.paInt16, channels=1, rate=16000,input_device_index =-1):
        # Audio Configuration
        self.CHUNK = chunk
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        
        # TCP Configuration
        self.DEST_IP = dest_ip
        self.DEST_PORT = dest_port
        self.input_device_index =input_device_index
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK, input_device_index=self.input_device_index)
    
    def send_audio(self):
        """Continuously sends audio chunks while handling connection losses."""
        reconnect_delay = 1  # Initial delay before reconnecting

        while True:
            try:
                print(f"Connecting to {self.DEST_IP}:{self.DEST_PORT}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.connect((self.DEST_IP, self.DEST_PORT))
                print("Connected. Streaming real-time audio...")

                reconnect_delay = 1  # Reset backoff delay after a successful connection

                while True:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    
                    # Send chunk size first (for receiver to handle packet properly)
                    chunk_size = len(data)
                    sock.sendall(chunk_size.to_bytes(4, 'big'))  # Send chunk size as 4-byte int
                    
                    # Send actual audio data
                    sock.sendall(data)

            except (ConnectionResetError, BrokenPipeError, socket.timeout, socket.error) as e:
                print(f"Connection lost ({e}). Reconnecting in {reconnect_delay} seconds...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 10)  # Exponential backoff (max 10s)
            except KeyboardInterrupt:
                print("Stopping sender.")
                break
            finally:
                sock.close()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    #put your receiver ip here
    streamer = AudioStreamer(dest_ip="0.0.0.0", dest_port=5006)
    try:
        streamer.send_audio()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        streamer.stop()
