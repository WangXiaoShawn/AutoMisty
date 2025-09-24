# filename: MistyHeadTouchCountdownTakePhoto.py

from CUBS_Misty import Robot
import cv2
import time
import base64
from typing import Any, Dict
import threading

def event_filter(name: str, comparison_operator: str, comparison_value: Any) -> Dict[str, Any]:
    '''
    Creates a dictionary for filtering event properties based on a condition.
    
    :param name: The property name to filter.
    :param comparison_operator: The operator for comparison (e.g., '=', '!=').
    :param comparison_value: The value to compare against.
    :return: A dictionary representing the filter condition.
    '''
    return {
        "Property": name,
        "Inequality": comparison_operator,
        "Value": comparison_value
    }

class CustomRobot(Robot):
    def __init__(self, ip: str):
        '''
        Initializes the CustomRobot with the given IP address.
        
        :param ip: The IP address of the Misty robot.
        '''
        super().__init__(ip)

    def misty_countdown_and_cute_action(self) -> None:
        '''
        Executes a countdown followed by a cute action by Misty upon triggering an event.
        '''
        # Step 1: Count down from 3 to 1
        self.speak("Three")
        time.sleep(1)
        self.speak("Two")
        time.sleep(1)
        self.speak("One")
        time.sleep(1)

        # Step 2: Capture a photo
        self.take_photo()

        # Step 3: Perform a cute action
        self.perform_cute_action()

    def take_photo(self) -> None:
        '''
        Captures a photo using the video feed.
        '''
        try:
            # Retrieve the latest frame from the video feed
            frame = self.frame_queue.get(timeout=0.1)
            
            # Generate a unique filename using the current timestamp
            photo_filename = f"photo_{int(time.time())}.jpg"
            
            # Save the frame as an image file
            cv2.imwrite(photo_filename, frame)
            
            print(f"[INFO] {photo_filename} has been captured.")
        except Exception as e:
            print(f"[ERROR] Failed to capture photo: {e}")

    def perform_cute_action(self) -> None:
        '''
        Executes a predefined cute action involving arm movements and LED changes.
        '''
        # Display a joyful expression
        self.emotion_Joy()
        
        # Change LED to a bright pink color
        self.change_led(255, 182, 193)  # Pink LED
        
        # Play a joyful sound
        self.sound_Joy()
        
        # Perform waving arms in a cute manner
        for _ in range(2):
            self.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=0.5)
            time.sleep(0.5)
            self.move_arms(leftArmPosition=90, rightArmPosition=90, duration=0.5)
            time.sleep(0.5)

        # Return Misty to normal state
        self.return_to_normal()

    def register_head_touch_event(self) -> None:
        '''
        Registers an event for the head touch, triggering the countdown and action sequence.
        '''
        def head_touch_callback(data):
            '''
            Callback function for head touch event, invoking countdown and action.
            '''
            print("[INFO] Head touch detected.")
            self.misty_countdown_and_cute_action()

        self.register_event(
            event_type="TouchSensor",
            event_name="HeadTouchEvent",
            condition=[event_filter("sensorPosition", "=", "HeadFront")],
            debounce=500,
            keep_alive=True,
            callback_function=head_touch_callback
        )

    def register_and_run_events(self):
        '''
        Registers all relevant events and begins the robot's event handling loop.
        '''
        self.register_head_touch_event()
        self.start()

    def start_perception_loop(self) -> None:
        '''
        Starts the main perception loop, initializing AV streams and threads.
        '''
        self._listen_for_ctrl_x()
        print("[INFO] Stopping existing AV streams...")
        self.stop_av_streaming()

        print("[INFO] Initiating AV stream...")
        self.start_av_stream()

        print("[INFO] Loading Whisper model for speech recognition...")
        self.load_whisper_model()

        # Launching background threads for stream processing
        threads = [
            threading.Thread(target=self._read_audio_stream, daemon=True),
            threading.Thread(target=self._process_audio, daemon=True),
            threading.Thread(target=self._video_reader_thread, daemon=True),
            threading.Thread(target=self.register_and_run_events, daemon=True)
        ]

        for t in threads:
            t.start()

        print("[INFO] System is now active. Use Ctrl+X to terminate.")

        self.process_multimodal_task()

        self._stop_event.set()
        print("[INFO] Shutting down the system.")

def main():
    misty_ip = "67.20.201.57"
    
    # Initialize the custom robot instance
    misty = CustomRobot(misty_ip)
    
    # Start the main perception loop
    misty.start_perception_loop()

if __name__ == "__main__":
    main()