# filename: misty_hello_perception.py

from CUBS_Misty import Robot
import cv2
import queue
import threading
import time

# Define the smile and wave action for Misty
def misty_hello_smile_wave(robot_ip):
    '''
    Execute actions for Misty to smile and wave when greeted with "Hello".
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display a joyful expression and play a welcoming sound
    misty.emotion_Joy()
    misty.sound_PhraseHello()
    
    # Step 2: Perform a waving action with arm
    for _ in range(2):  # Repeat twice for a waving gesture
        misty.move_arms(leftArmPosition=30, rightArmPosition=-29, duration=0.5)
        time.sleep(0.5)
        misty.move_arms(leftArmPosition=-29, rightArmPosition=30, duration=0.5)
        time.sleep(0.5)
    
    # Step 3: Keep a cheerful facial expression after waving
    misty.emotion_JoyGoofy()
    
    # Step 4: Reset Misty back to her neutral state
    misty.return_to_normal()

class MistyHelloDetection(Robot):
    def __init__(self, ip: str) -> None:
        super().__init__(ip)
        self.excited_action_queued = False

    def process_multimodal_task(self) -> None:
        '''
        Continuously captures video frames and listens for the verbal cue 'Hello'.
        On detecting 'Hello', triggers Misty's smiling and waving actions.
        '''
        
        # Create a resizable window to display the video feed
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Ensure window dynamically fits video feed
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
                cv2.resizeWindow("MistyVideo", frame_width, frame_height)
                
                # Process speech transcription queue
                while not self.transcript_queue.empty():
                    _, self.last_transcript = self.transcript_queue.get()

                if self.last_transcript:
                    # Overlay the spoken text onto the video
                    cv2.putText(
                        frame, self.last_transcript, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA
                    )

                # Check for 'hello' and trigger smiling and waving actions
                if "hello" in self.last_transcript.lower() and not self.excited_action_queued:
                    self.excited_action_queued = True
                    misty_hello_smile_wave(self.ip)
                    # Clear the transcript to prevent repeated triggers
                    self.last_transcript = ""
                    self.excited_action_queued = False

                # Display the video feed
                cv2.imshow("MistyVideo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        # Properly close the video window
        cv2.destroyAllWindows()

    def start_perception_loop(self) -> None:
        '''
        Starts the main perception loop of the robot. This includes real-time video 
        and audio processing, and event handling.
        '''
        self._listen_for_ctrl_x()
        self.stop_av_streaming()
        print("[INFO] Starting AV stream...")
        self.start_av_stream()
        print("[INFO] Loading Whisper speech recognition model...")
        self.load_whisper_model()
        
        threads = [
            threading.Thread(target=self._read_audio_stream, daemon=True),
            threading.Thread(target=self._process_audio, daemon=True),
            threading.Thread(target=self._video_reader_thread, daemon=True)
        ]

        for t in threads:
            t.start()

        print("[INFO] System is running. Press Ctrl+X to exit.")
        self.process_multimodal_task()
        self._stop_event.set()
        print("[MAIN] System shutdown.")

if __name__ == "__main__":
    misty = MistyHelloDetection("67.20.199.168")
    misty.start_perception_loop()