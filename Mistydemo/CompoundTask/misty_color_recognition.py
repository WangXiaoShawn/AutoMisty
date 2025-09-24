# filename: misty_color_recognition.py

from CUBS_Misty import Robot
import cv2
import queue
import threading
import time  # Importing the time module

class MistyColorRecognition(Robot):
    def __init__(self, ip: str):
        super().__init__(ip)
        self.recognized_color = None

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                while not self.transcript_queue.empty():
                    _, self.last_transcript = self.transcript_queue.get()
                    self.recognized_color = self.extract_color_from_transcript(self.last_transcript)

                if self.recognized_color:
                    # Change the LED to the recognized color
                    self.misty_recognize_and_display_color_with_delay(self.ip, self.recognized_color)
                    self.recognized_color = None

                if self.last_transcript:
                    cv2.putText(frame, self.last_transcript, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow("MistyVideo", frame)

            except queue.Empty:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def extract_color_from_transcript(self, transcript: str) -> str:
        # Extract the color from the transcription
        colors = ["red", "green", "blue", "yellow", "purple", "white"]
        for color in colors:
            if color in transcript.lower():
                return color
        return None

    def misty_recognize_and_display_color_with_delay(self, robot_ip, recognized_color):
        # This function uses the ActionAgent's code directly
        # Color mapping for LED
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "white": (255, 255, 255)
        }

        # Initialize Misty
        misty = Robot(robot_ip)

        # Check if the recognized color is supported and change LED
        if recognized_color in color_map:
            red, green, blue = color_map[recognized_color]
            misty.change_led(red, green, blue)
            time.sleep(2) # Keep the color change for 2 seconds
          
        else:
            print(f"Color '{recognized_color}' is not recognized. Available colors: {list(color_map.keys())}")

        # Reset Misty to her normal state
        misty.return_to_normal()

    def start_perception_loop(self) -> None:  ### -1 effecienccy
        '''
        Starts the main perception loop of the robot. This includes:
        1) Listening for Ctrl+X to exit.
        2) Stopping any existing audio/video streaming.
        3) Starting new AV streams for video and audio.
        4) Loading the Whisper model for speech recognition.
        5) Spinning up background threads to handle audio, video, and event registration.
        6) Running the main multimodal processing loop.
        7) Handling system shutdown.
        '''
        # 1. Start a listener for detecting the Ctrl+X key combination in a non-blocking manner.
        self._listen_for_ctrl_x()

        # 2. Stop any existing AV streams to ensure a fresh start.
        print("[INFO] Stopping existing streams...")
        self.stop_av_streaming()

        # 3. Start a new AV stream for both video and audio.
        print("[INFO] Starting AV stream...")
        self.start_av_stream()

        # 4. Load the Whisper model for real-time or near-real-time speech transcription.
        print("[INFO] Loading Whisper speech recognition model...")
        self.load_whisper_model()

        # 5. Create and launch background threads for different tasks:
        #    - Reading the audio stream
        #    - Processing audio data
        #    - Reading video frames
        threads = [
            threading.Thread(target=self._read_audio_stream, daemon=True),
            threading.Thread(target=self._process_audio, daemon=True),
            threading.Thread(target=self._video_reader_thread, daemon=True)
        ]

        # Start each of the background threads.
        for t in threads:
            t.start()

        print("[INFO] System is running. Press Ctrl+X to exit.")

        # 6. Run the main loop that handles video display (OpenCV) and transcription events.
        self.process_multimodal_task()

        # 7. On exit, set a stop event flag for all threads and perform system cleanup.
        self._stop_event.set()
        print("[MAIN] System shutdown.")

if __name__ == "__main__":
    misty = MistyColorRecognition("67.20.199.168")
    misty.start_perception_loop()