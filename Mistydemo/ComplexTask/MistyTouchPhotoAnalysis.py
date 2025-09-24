# filename: MistyTouchPhotoAnalysis.py

from CUBS_Misty import Robot
from typing import Any, Dict
import cv2
import time
import base64
import threading
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

def event_filter(name: str, comparison_operator: str, comparison_value: Any) -> Dict[str, Any]:
    '''
    Creates a dictionary for filtering event properties based on a condition.
    
    :param name: The name of the property to filter on.
    :param comparison_operator: A string representing the comparison operator (e.g. "=", ">", "<").
    :param comparison_value: The value against which the property is compared.
    :return: A dictionary containing the filter condition details.
    '''
    return {
        "Property": name,
        "Inequality": comparison_operator,
        "Value": comparison_value
    }

class CustomRobot(Robot):
    def __init__(self, ip: str):
        '''
        Initializes the CustomRobot with the given IP address and API key.
        
        :param ip: The IP address of the robot.
        '''
        super().__init__(ip)
        
        # Instantiate the ChatOpenAI class with your chosen model, API key, and temperature settings.
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,  # 使用传入的API key参数
            temperature=0.7,
        )
        
        # A predefined task or prompt for the GPT model to perform when analyzing images.
        self.vision_analysis_task = '''
            Analyze the photo carefully and describe the main elements and any notable details you observe.
            Provide insights into what is happening or what the image might represent.
        '''

    def capture_and_analyze_photo(self) -> None:
        '''
        Captures a photo from the video feed and analyzes it using GPT.
        '''
        try:
            # Attempt to retrieve the most recent frame from the frame queue with a small timeout.
            frame = self.frame_queue.get(timeout=0.1)
            
            # Create a filename for the photo using the current time to ensure uniqueness.
            photo_filename = f"photo_{int(time.time())}.jpg"
            
            # Save the retrieved frame as an image file.
            cv2.imwrite(photo_filename, frame)
            print(f"[PHOTO] {photo_filename} saved.")
            
            # Call the analyze_image_with_gpt method to describe the image content.
            result = self.analyze_image_with_gpt(photo_filename)
            
            # Print GPT’s result in the console for debugging or confirmation.
            print("[GPT RESULT]", result)
            
            # Have the robot speak out the result (assuming a speak method is available).
            self.speak(result)
        except Exception as e:
            # If the frame capture or image analysis fails, print out the error.
            print(f"Failed to capture or analyze photo: {e}")

    def analyze_image_with_gpt(self, image_path: str) -> str: #-1
        '''
        Analyzes an image using GPT to describe its content.

        :param image_path: The path to the image file that needs to be analyzed.
        :return: A string containing GPT’s analysis of the image.
        '''
        try:
            # 1) Read the image file in binary mode and encode it into Base64 string.
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')

            # 2) Construct messages for GPT that include both a text instruction and the Base64-encoded image.
            messages = [
                # AIMessage acts as a system or "assistant" context to guide GPT’s behavior.
                AIMessage(content="You are an expert in analyzing image content."),
                
                # HumanMessage includes the prompt and the image data. 
                HumanMessage(content=[
                    {"type": "text", "text": self.vision_analysis_task},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ])
            ]

            # 3) Invoke the GPT model (ChatOpenAI) with the constructed messages.
            response = self.llm.invoke(messages)

            # 4) Extract and return GPT's response by stripping extra whitespace.
            return response.content.strip()

        except Exception as e:
            # If there's an issue with invoking GPT, print an error message and return a default string.
            print(f"[GPT] Error calling invoke: {e}")
            return "GPT request failed."

    def register_head_touch(self) -> None:
        '''
        Registers the head touch event with a callback function that captures and analyzes a photo 
        when the robot's head is touched.
        '''
        
        # Define a callback function that will be triggered upon a head touch event.
        def head_touch_callback(data):
            '''
            The callback function triggered by a head touch event.
            
            :param data: Event data passed from the robot’s event system.
            '''
            print("[INFO] Head touched. Event successfully triggered.")
            print("[INFO] Capturing and analyzing photo...")
            
            # Upon detecting head touch, call the method to capture and analyze a photo.
            self.capture_and_analyze_photo()

        # Register the head touch event, specifying the event type, conditions, debounce time, 
        # keep_alive property, and the callback function.
        self.register_event(
            event_type="TouchSensor",
            event_name="HeadTouch",
            condition=[
                event_filter("sensorPosition", "=", "HeadFront")
            ],
            debounce=500,
            keep_alive=True,
            callback_function=head_touch_callback
        )

    def regist_and_run_event(self):
        '''
        Registers the head touch event and starts the robot's event loop.
        This method is often run in a separate thread to keep the main thread available.
        '''
        self.register_head_touch()
        self.start()  # Start the event handling loop in the parent class (Robot).

    def start_perception_loop(self) -> None:
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
        #    - Registering events and running the event loop
        threads = [
            threading.Thread(target=self._read_audio_stream, daemon=True),
            threading.Thread(target=self._process_audio, daemon=True),
            threading.Thread(target=self._video_reader_thread, daemon=True),
            threading.Thread(target=self.regist_and_run_event, daemon=True)  # Event registration and loop
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

def main():
    '''
    The main entry point of the script. Initializes and starts the CustomRobot's functionalities.
    '''
    # Robot IP (replace with your own valid credentials).
    misty_ip = '67.20.199.168'
    
    # Create an instance of CustomRobot with the provided IP.
    misty = CustomRobot(misty_ip)
    
    try:
        # Register the robot's head touch event.
        misty.register_head_touch()
        
        # Start the main perception loop, which includes AV streaming and event handling.
        misty.start_perception_loop()
    except Exception as e:
        # If there's an error during setup or the main loop, print it out.
        print(f"An error occurred: {e}")

# If this file is run directly, call the main function.
if __name__ == "__main__":
    main()