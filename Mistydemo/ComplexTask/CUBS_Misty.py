
import os
import time
import base64
import json
import threading
import requests
import websocket
import numpy as np
import whisper
import librosa
import queue
import cv2
import av
import re
import sys
from typing import Optional, Callable, Dict, Any,List,Tuple
from random import randint
from requests import request, Response, get
import _thread as thread
from time import sleep
from RobotCommands import RobotCommands
from pynput import keyboard
sys.path.append('/Users/xiaowang/Documents/Code/MistyCodeGenProject/mistyPy')
from CUBS_Misty import RobotCommands
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import pdb
# Bump Sensors (Bump Sensors detect physical collisions)
VALID_BUMP_SENSORS = {
    "bfl",  # Front-left
    "bfr",  # Front-right
    "brl",  # Rear-left
    "brr"   # Rear-right
}

# Capacitive Touch Sensors (Touch Sensors detect human touch interactions)
VALID_CAP_SENSORS = {
    "Chin",       # Chin area (below the robot's head)
    "Scruff",     # Scruff area (back of the robot's neck)
    "HeadRight",  # Right side of the head
    "HeadLeft",   # Left side of the head
    "HeadBack",   # Back of the head
    "HeadFront"   # Front of the head (forehead area)
}
def event_filter(name: str, comparison_operator: str, comparison_value: Any) -> Dict[str, Any]:
    """
    Creates a standardized event filter dictionary, which can be used
    when subscribing to events to specify filtering conditions.

    :param name: Name of the field on which the filter will apply.
    :param comparison_operator: The comparison operator to use, e.g., '=', '>', '<', etc.
    :param comparison_value: The value to compare against (could be a string, number, etc.).
    :return: A dictionary that represents the event filter.
    """
    return {
        "Property": name,
        "Inequality": comparison_operator,
        "Value": comparison_value
    }


## inference https://docs.mistyrobotics.com/misty-ii/web-api/api-reference/#playaudio
class Robot(RobotCommands): # 继承与command
    def __init__(self, ip='127.0.0.1'):
        self.ip = ip
        self.model: Optional[whisper.Whisper] = None
        self.rtsp_url: Optional[str] = None
        self.audio_queue: queue.Queue[Tuple[float, np.ndarray]] = queue.Queue()
        self.transcript_queue: queue.Queue[Tuple[float, str]] = queue.Queue()
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.last_transcript: str = ""
        self._stop_event: threading.Event = threading.Event()
        self._ctrl_pressed = False 
        self.trigger_ctrl_pressed = False 
        self.active_event_registrations: Dict[str, "Robot.Event"] = {}
        self.trigger_stop_event = threading.Event()  # 用于退出监听
        # Voice processing state variables
        self.voice_active: bool = False
        self.last_voice_time: Optional[float] = None
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.silence_threshold_db: int = -40
        self.silence_duration_threshold: float = 0.5
        self.min_utterance_length: float = 0.3
        ### see and listening
        self.ignore_transcript_until = 0.0  # 时间戳，表示在这个时间之前忽略转录
        self.buffer = 0.3    # safty for conversation interval
        # 用于串行化 GPT 调用的锁
        self._analysis_lock = threading.Lock()
        # ===  ChatOpenAI (LangChain) ===
        self.vision_analysis_task= """
                    Carefully analyze the given image. 
                     Provide your best guess of what's in this image.
                """
        self.chat_ai_task="""
            You are a cute assistant named Misty! Speak in an adorable way and keep your answers short and sweet! 
        """
        self.oai_project_key = "YOUR_OPENAI_API_KEY_HERE"  # 请设置您的OpenAI API Key
     
        self.llm = ChatOpenAI(
            model="gpt-4o",  
            openai_api_key= self.oai_project_key,
            temperature=0,
        )
        self.conversation_history = []  # Design for oversation memory
        

    @staticmethod
    def is_silent(audio_data: np.ndarray, sample_rate: int, silence_threshold_db: int = -40) -> bool:
        '''
        Detect if audio contains silence
        
        Args:
            audio_data (np.ndarray): Audio samples
            sample_rate (int): Sampling rate
            silence_threshold_db (int): Silence threshold in dB
            
        Returns:
            bool: True if audio is considered silent
        '''
        rms = np.sqrt(np.mean(audio_data**2))
        db = 20 * np.log10(rms) if rms > 0 else -1000
        return db < silence_threshold_db

    def load_whisper_model(self) -> None:
        '''Load Whisper speech recognition model'''
        print("[INFO] Loading Whisper model (small.en)...")
        self.model = whisper.load_model("small.en")
        print("[INFO] Whisper model loaded.")

    def start_av_stream(self, port: int = 1935, width: int = 640, height: int = 480) -> None:
        '''
        Initialize audio-visual streaming
        
        Args:
            port (int): RTSP port number
            width (int): Video width
            height (int): Video height
        '''
        print("[INFO] Enabling AV streaming service...")
        res_enable: Response = self.enable_av_streaming_service()
        if res_enable.status_code != 200:
            raise RuntimeError(f"Failed to enable AV! status={res_enable.status_code}")

        print(f"[INFO] Starting AV stream on port {port} ({width}x{height})...")
        res_start: Response = self.start_av_streaming(f"rtspd:{port}", width=width, height=height)
        if res_start.status_code != 200:
            raise RuntimeError(f"Start AV failed! status={res_start.status_code}")

        self.rtsp_url = f"rtsp://{self.ip}:{port}"
        print("[INFO] RTSP URL =", self.rtsp_url)
        

 

    def _read_audio_stream(self) -> None:
        '''Thread worker for reading audio data from RTSP stream'''
        try:
            container = av.open(self.rtsp_url, options={'rtsp_transport': 'tcp', 'stimeout': '5000000'})
            print("[AUDIO] RTSP audio opened.")
        except av.AVError as e:
            print(f"[AUDIO] Open failed: {e}")
            return

        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        if not audio_stream:
            print("[AUDIO] No audio stream.")
            return

        try:
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    frame_data: np.ndarray = frame.to_ndarray()
                    if frame_data.ndim > 1 and frame_data.shape[0] > 1:
                        frame_data = np.mean(frame_data, axis=0)
                    self.audio_queue.put((time.time(), frame_data.flatten()))
        except Exception as e:
            print(f"[AUDIO] Error: {e}")
        finally:
            print("[AUDIO] Thread exit.")

    def _process_audio(self) -> None:
        # '''Thread worker for processing audio data and detecting speech segments'''
        while not self._stop_event.is_set():
            try:
                timestamp: float
                frame_data: np.ndarray
                timestamp, frame_data = self.audio_queue.get(timeout=5)
                self.audio_buffer = np.concatenate((self.audio_buffer, frame_data))

                current_silent: bool = self.is_silent(frame_data, 44100, self.silence_threshold_db)
                
                if not current_silent:
                    if not self.voice_active:
                        print(">> Voice activity detected")
                        self.voice_active = True
                    self.last_voice_time = time.time()
                else:
                    if self.voice_active and self.last_voice_time:
                        silence_duration: float = time.time() - self.last_voice_time
                        if silence_duration >= self.silence_duration_threshold:
                            print(f">> Voice activity ended (silence duration {silence_duration:.1f}s)")
                            self._handle_utterance_end()
            except queue.Empty:
                if self.voice_active:
                    self._handle_utterance_end()
                continue

    def _handle_utterance_end(self) -> None:
        # '''Process completed speech segment and initiate transcription'''
        audio_duration: float = len(self.audio_buffer) / 44100
        if audio_duration >= self.min_utterance_length:
            self._transcribe_audio()
        else:
            print(f">> Discarding short utterance ({audio_duration:.1f}s < {self.min_utterance_length}s)")

        self.audio_buffer = np.array([], dtype=np.float32)
        self.voice_active = False
        self.last_voice_time = None

    def _transcribe_audio(self) -> None:
        try:
            if time.time() < self.ignore_transcript_until:
                print("[INFO] Transcript ignored during TTS cooldown.")
                return
            audio_resampled: np.ndarray = librosa.resample(
                self.audio_buffer, orig_sr=44100, target_sr=16000
            )
            result: Dict[str, Any] = self.model.transcribe(
                audio_resampled, language='en', fp16=False, task='transcribe'
            )
            transcript: str = result.get('text', "").strip()

            # Filter out common polite phrases
            banned_phrases: List[str] = ["thank you", "thanks", "thank"]
            if not any(bp in transcript.lower() for bp in banned_phrases):
                self.transcript_queue.put((time.time(), transcript))
                print(f"Transcription result: {transcript}")
        except Exception as e:
            print(f"Transcription failed: {e}")

    def _video_reader_thread(self) -> None:
        '''Thread worker for reading and processing video frames'''
        cap: cv2.VideoCapture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("[VIDEO] Failed to open video.")
            return

        try:
            while not self._stop_event.is_set():
                ret: bool
                frame: np.ndarray
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                self.frame_queue.put(frame)
                time.sleep(0.01)
        except Exception as e:
            print(f"[VIDEO] Error: {e}")
        finally:
            cap.release()
            print("[VIDEO] Thread exit.")
            
    def chat_with_gpt(self, user_input: str) -> str:
        '''
        Sends user input to GPT and returns the response.
        '''
        with self._analysis_lock:  # Ensure thread safety with a lock
            try:
                # messages = [
                #     AIMessage(content=self.chat_ai_task),  # System message defining AI behavior
                #     HumanMessage(content=user_input)  # User input message
                # ]
                # 将用户输入添加到对话历史中
                self.conversation_history.append(HumanMessage(content=user_input))
                # 构造包含历史上下文的消息列表
                messages = [AIMessage(content=self.chat_ai_task)] + self.conversation_history

                # Call LangChain's ChatOpenAI model to generate a response
                response = self.llm.invoke(messages)
                self.conversation_history.append(AIMessage(content=response.content.strip()))


                return response.content.strip()  # Return the cleaned response

            except Exception as e:
                print(f"[GPT] Error calling invoke: {e}")  # Log any errors
                return "I'm sorry, I couldn't process your request."  # Return a fallback response


    def analyze_image_with_gpt(self, image_path: str) -> str:
        """
        Sends an image to GPT using 'AIMessage' + 'HumanMessage(content=[...])' 
        with a 'type: image_url' object embedded in the content.
        
        This method relies on the backend's ability to process 'image_url' for 
        multimodal analysis. If the backend does not support image processing, 
        the request may not work as expected.
        
        Args:
            image_path (str): The file path of the image to be analyzed.

        Returns:
            str: GPT's response after analyzing the image.
        """
        with self._analysis_lock:
            # Ensure thread safety by serializing access to this function
            try:
                # 1) Read the image and encode it in Base64 format
                with open(image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode('utf-8')
                # Define the instruction template for GPT's image analysis.
                    # When you want to change GPT's task, simply modify this template.
                    # For example, if you want GPT to analyze the emotions of the person in the image, 
                    # you can change this template accordingly.
                # template = """
                #     Carefully analyze the given image. 
                #     Provide your best guess of what's in this image.
                # """
                
                messages = [
                    AIMessage(content="You are an expert in analyzing image content."),
                    HumanMessage(content=[
                        {"type": "text", "text": self.vision_analysis_task},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ])
                ]

                # 3) Send the constructed messages to GPT using self.llm.invoke(...)
                response = self.llm.invoke(messages)

                # 4) Extract and return GPT's response
                return response.content.strip()

            except Exception as e:
                print(f"[GPT] Error calling invoke: {e}")  # Log the error
                return "GPT request failed."  # Return an error message if the request fails
            
    # def process_multimodal_task(self) -> None:
    #     cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)
    #     pdb.set_trace()
    #     while not self._stop_event.is_set():
    #         try:
    #             frame = self.frame_queue.get(timeout=0.1)
    #         except queue.Empty:
    #             continue 

    #         while not self.transcript_queue.empty():
    #             _, self.last_transcript = self.transcript_queue.get()

    #         if self.last_transcript:
    #             response = self.chat_with_gpt(self.last_transcript)
    #             self.gpt_speak(response)
    #             self.last_transcript = ""

    #         cv2.imshow("MistyVideo", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             self._stop_event.set()
    #             break

    #     cv2.destroyAllWindows()

    def process_multimodal_task(self) -> None:
        # '''
        # Handles video display and transcriptions in the main thread.
        # This function runs in the main thread to avoid OpenCV issues.
        # '''
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MistyVideo", 640, 480)

        while not self._stop_event.is_set():
            try:
                
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue 
            while not self.transcript_queue.empty():
                _, self.last_transcript = self.transcript_queue.get()

            if self.last_transcript:
                cv2.putText(
                    frame, self.last_transcript, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA
                )
            ##############################################
            # If you need to handle tasks based on the most recent speech input, process self.last_transcript here.
            # If you need to implement vision-based tasks, add frame analysis logic here.
            ##############################################
            cv2.imshow("MistyVideo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  
                self._stop_event.set()
                break

    #     cv2.destroyAllWindows()
    # def process_multimodal_task(self) -> None:
    #     '''
    #     Continuously processes video frames and speech input for multimodal interaction.
        
    #     - Displays the video feed with transcribed speech overlay.
    #     - Captures a photo and sends it to GPT for analysis when the user says "take a photo".
    #     - Ensures smooth handling of real-time inputs and outputs.
    #     '''
        
    #     # Create a resizable window to display the video feed
    #     cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("MistyVideo", 640, 480)  # Set window size to 640x480

    #     while not self._stop_event.is_set():  # Run the loop until a stop signal is received
    #         try:
    #             frame = self.frame_queue.get(timeout=0.1)  # Retrieve the latest video frame
    #         except queue.Empty:
    #             continue  # Skip processing if no frame is available

    #         # Process the latest speech transcription from the queue
    #         while not self.transcript_queue.empty():
    #             _, self.last_transcript = self.transcript_queue.get()

    #         if self.last_transcript:  # If there is a new transcription
    #             # Overlay the spoken text onto the video feed
    #             cv2.putText(
    #                 frame, self.last_transcript, (30, 50),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA
    #             )

    #         # If the user says "take a photo", capture and analyze an image
    #         if "take a photo" in self.last_transcript.lower():
    #             # Generate a unique filename based on the current timestamp
    #             photo_filename = f"photo_{int(time.time())}.jpg"
    #             cv2.imwrite(photo_filename, frame)  # Save the current video frame as an image
    #             print(f"[PHOTO] {photo_filename} saved. Sending to GPT...")

    #             # Call analyze_image_with_gpt to analyze the captured image
    #             result = self.analyze_image_with_gpt(photo_filename)

    #             print("[GPT RESULT]", result)  # Print GPT's analysis result

    #             # Let the robot speak out GPT's analysis
    #             self.gpt_speak(result)

    #             # Clear the transcript after processing to prevent repeated triggers
    #             self.last_transcript = ""

    #         # Display the video feed with text overlay
    #         cv2.imshow("MistyVideo", frame)

    #         # Check for user input to exit (press 'q' to quit)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             self._stop_event.set()  # Stop the loop when 'q' is pressed
    #             break

    #     # Clean up and close the video window when exiting
    #     cv2.destroyAllWindows()

   
    
    
    # def process_multimodal_task(self) -> None:
    #     cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

    #     while not self._stop_event.is_set():
    #         try:
    #             frame = self.frame_queue.get(timeout=0.1)
    #         except queue.Empty:
    #             continue 

    #         while not self.transcript_queue.empty():
    #             _, self.last_transcript = self.transcript_queue.get()

    #         if self.last_transcript:
    #             response = self.chat_with_gpt(self.last_transcript)
    #             #self.gpt_speak(response)
    #             self.speak(response)
    #             self.last_transcript = ""

    #         cv2.imshow("MistyVideo", frame)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             self._stop_event.set()
    #             break

    #     cv2.destroyAllWindows()

  
    def _listen_for_ctrl_x(self) -> None: # this design for av stream
        # '''
        # Listens for Ctrl+X key press in a non-blocking way.
        # '''
        def on_press(key):
            if key in [keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
                self._ctrl_pressed = True
            elif self._ctrl_pressed and hasattr(key, 'char') and key.char == 'x':
                print("[INFO] Detected Ctrl+X, shutting down...")
                self._stop_event.set()
                self.trigger_stop_event.set()  # 这里一定要用 trigger_stop_event

        def on_release(key):
            if key in [keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
                self._ctrl_pressed = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()  

    def start_perception_loop(self) -> None:
        # 
        # Starts the main execution loop of the robot system while ensuring that 
        # OpenCV's GUI functions (cv2.imshow()) are handled within the main thread.
        # This method initializes various subsystems, starts required background 
        # threads for audio and video processing, and manages system shutdown.
        # 
        # 1. Start a listener for detecting the Ctrl+X key combination (non-blocking).
        #    This allows the user to gracefully terminate the system by pressing Ctrl+X.
        self._listen_for_ctrl_x()
        # 2. Stop any existing audio-visual (AV) streams to ensure a clean restart.
        #    This prevents conflicts if a previous session was running.
        print("[INFO] Stopping existing streams...")
        self.stop_av_streaming()
        # 3. Start a new AV stream for capturing video and audio.
        #    This is necessary for real-time video processing and speech recognition.
        print("[INFO] Starting AV stream...")
        self.start_av_stream()
        # 4. Load the Whisper model for speech-to-text transcription.
        #    Whisper is an AI model that converts spoken language into text.
        print("[INFO] Loading Whisper speech recognition model...")
        self.load_whisper_model()
        # 5. Launch background threads for handling different processing tasks.
        #    These threads run in parallel to the main thread and do not block UI updates.
        threads = [
            threading.Thread(target=self._read_audio_stream, daemon=True),  # Captures audio stream
            threading.Thread(target=self._process_audio, daemon=True),      # Processes audio and transcriptions
            threading.Thread(target=self._video_reader_thread, daemon=True) # Reads and processes video frames
        ]
  
        # Start all background threads
        for t in threads:
            t.start()
        print("[INFO] System is running. Press Ctrl+X to exit.")
        # 6. Run the main processing loop for video display and transcription handling.
        #    This ensures OpenCV functions like cv2.imshow() run within the main thread.
        self.process_multimodal_task()
        # 7. On exit, signal all background threads to terminate and perform cleanup.
        self._stop_event.set()  # Notify all threads to stop
        print("[MAIN] System shutdown.")
        
        
    def register_event(
            self,
            event_type: str,
            event_name: str = "",
            condition: Optional[Dict[str, Any]] = None,
            debounce: int = 0,
            keep_alive: bool = False,
            callback_function: Optional[Callable[[Dict[str, Any]], None]] = None    
        ) -> Optional["Robot.Event"]:
            import pdb
            '''
            Subscribes to a robot event. Once triggered, the event data will be received via
            WebSocket, and the callback_function (if provided) will be executed.

            :param event_type: The type of event, e.g., 'BumpSensor', 'TouchSensor'.
            :param event_name: A custom name for the event subscription. If not provided, 
                            defaults to event_type.
            :param condition: Additional filtering conditions (as a dictionary) for the event.
            :param debounce: Debounce time in milliseconds, within which repeated events
                            will be ignored.
            :param keep_alive: Whether to keep the subscription active after the first event trigger.
            :param callback_function: Function that processes the event data. Must accept one argument
                                    (the event data dictionary).
            :return: A Robot.Event object if registration is successful, otherwise None.
            '''

            # If a callback function is provided, ensure it takes exactly one argument
            
            if callback_function is not None and callback_function.__code__.co_argcount != 1:
                print("Callback function must have only one argument.")
                return None

            # If no event name is provided, use the event_type as the default name
            if not event_name:
                print(f"No event_name provided when registering to {event_type} - using default name {event_type}")
                event_name = event_type

            # Remove any subscriptions whose connections have already closed
            self.__remove_closed_events()

            # Avoid re-registration if the same event_name is already subscribed
            if event_name in self.active_event_registrations:
                print(f"A registration already exists for event name {event_name}, ignoring request to register again")
                return None

            # Create a new event subscription object
            new_registration = self.Event(
                self.ip,
                event_type,
                condition=condition,
                _debounce=debounce,
                keep_alive=keep_alive,
                callback_function=callback_function
            )
            # Store it in the dictionary
            self.active_event_registrations[event_name] = new_registration
            return new_registration

    def unregister_event(self, event_name: str) -> None:
            '''
            Unsubscribes a previously registered event by name and removes it from
            the active_event_registrations dictionary.

            :param event_name: The name used when the event was registered.
            '''
            if event_name not in self.active_event_registrations:
                print(f"Not currently registered to event: {event_name}")
                return
            
            try:
                # Call the unsubscribe method to close the WebSocket connection
                self.active_event_registrations[event_name].unsubscribe()
            except Exception:
                pass

            # Remove from the dictionary
            del self.active_event_registrations[event_name]

    def unregister_all_events(self) -> None:
            '''
            Unregisters all active event subscriptions.
            '''
            for event_name in list(self.active_event_registrations.keys()):
                self.unregister_event(event_name)

    def get_registered_events(self) -> Dict[str, "Robot.Event"]:
            '''
            Retrieves all currently registered events. Before returning the list,
            it first removes any events with closed WebSocket connections.

            :return: A dictionary where keys are event_name and values are Robot.Event objects.
            '''
            self.__remove_closed_events()
            return self.active_event_registrations

    def keep_alive(self) -> None:
            '''
            Continuously checks all registered event connections to ensure they're still active.
            Removes any inactive connections. Typically runs in a separate thread to maintain
            ongoing communication with the robot.
            '''
            while self.active_event_registrations:
                self.__remove_closed_events()
                sleep(1)

    def __remove_closed_events(self) -> None:
            '''
            Internal method to remove subscriptions that are closed or inactive.
            '''
            # Identify events where is_active is False
            events_to_remove = [
                event_name for event_name, event_subscription in self.active_event_registrations.items() 
                if not event_subscription.is_active
            ]
            # Unregister those events
            for event_name in events_to_remove:
                print(f"Event connection has closed for event: {event_name}")
                self.unregister_event(event_name)
        
    def start(self) -> None:
        '''
        Starts the robot's main loop.

        Responsibilities:
        1. Launches a background thread to listen for the Ctrl+X key press.
        2. Keeps the main thread active, waiting for the user to trigger the exit event.
        3. Unregisters all event subscriptions before shutting down to free resources.
        4. Terminates the program cleanly upon exit.

        Execution Flow:
        - The robot starts running.
        - The user presses Ctrl+X.
        - The exit process is triggered, cleaning up event subscriptions.
        - The program terminates safely.
        '''

        # Step 1: Create a separate thread to listen for Ctrl+X key presses
        e_ctrl = threading.Thread(target=self._listen_for_ctrl_x, daemon=True)

        # Step 2: Start the background thread for keyboard monitoring
        e_ctrl.start()

        # Step 3: Inform the user that the robot is running
        print("[INFO] Robot is running. Press Ctrl+X to exit...")

        # Step 4: Block the main thread, waiting until the stop event is triggered
        # This event will be set when _listen_for_ctrl_x() detects Ctrl+X
        self.trigger_stop_event.wait()

        # Step 5: Before exiting, unregister all active event subscriptions
        self.unregister_all_events()
        # Step 6: Notify the user that the robot has safely stopped
        print("[INFO] Robot has stopped safely.")
        # Step 7: Exit the Python process gracefully
        sys.exit(0)


    class Events:
            '''
            A subclass enumerating available events and exposing event type constants.
            In this example, only BumpSensor and TouchSensor are provided.
            '''
            available_events = ['BumpSensor', 'TouchSensor']
            BumpSensor: str = 'BumpSensor'
            TouchSensor: str = 'TouchSensor'

    class Event:
            '''
            Manages the logic for a single event subscription, including:
            - Building the subscription message
            - Establishing a WebSocket connection
            - Receiving and processing event data
            - Unsubscribing
            '''

            def __init__(
                self,
                ip: str,
                event_type: str,
                condition: Optional[Dict[str, Any]] = None,
                _debounce: int = 0,
                keep_alive: bool = False,
                callback_function: Optional[Callable[[Dict[str, Any]], None]] = None
            ) -> None:
                '''
                Initializes a single event subscription.

                :param ip: IP address of the target robot/simulation environment.
                :param event_type: Type of the event, such as 'BumpSensor', 'TouchSensor'.
                :param condition: Event filter conditions as a dictionary, or None if none.
                :param _debounce: Debounce time (milliseconds). Repeated events within this time 
                                are ignored.
                :param keep_alive: Whether the subscription remains active after the first event trigger.
                :param callback_function: Callback function that processes the event data (must accept
                                        one dictionary argument).
                '''
                # Validate if the event_type is available
                if event_type in Robot.Events.available_events:
                    self.event_type: str = getattr(Robot.Events, event_type)
                else:
                    self.is_active: bool = False
                    print(f"Invalid subscription: {event_type}")
                    return

                self.ip: str = ip
                self.condition: Optional[Dict[str, Any]] = condition
                self.debounce: int = _debounce
                # Initial data indicates not yet subscribed or just waiting for data
                self.data: Dict[str, Any] = json.loads('{"status":"Not_Subscribed or just waiting for data"}')
                self.event_name: Optional[str] = None
                self.ws: Optional[websocket.WebSocketApp] = None
                self.initial_flag: bool = True  # Used to handle the first message differently
                self.keep_alive: bool = keep_alive
                self.callback_function: Optional[Callable[[Dict[str, Any]], None]] = callback_function
                self.is_active: bool = True  # Tracks whether the WebSocket is active

                # Create and start the thread for event listening
                self.thread = threading.Thread(target=self.initiate)
                self.thread.start()

            def initiate(self) -> None:
                '''
                Starts the WebSocket connection and sets up callbacks for messages,
                errors, and closures.
                '''
                websocket.enableTrace(False)
                self.ws = websocket.WebSocketApp(
                    f"ws://{self.ip}/pubsub",
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )
                # run_forever blocks the current thread, so it must be called in a separate thread
                self.ws.run_forever(ping_timeout=10)

            def on_message(self, message: str) -> None:
                '''
                Callback invoked when the WebSocket receives a message.
                Decodes the JSON data, calls the user-defined callback function
                (if any), and unsubscribes if keep_alive is False.

                :param message: The text message received from the WebSocket, usually in JSON format.
                '''
                if not self.initial_flag:
                    # Convert the string to a dictionary
                    self.data = json.loads(message)
                    # Call the user-provided callback if available
                    if self.callback_function:
                        self.callback_function(self.data)
                    # If not meant to remain subscribed, unsubscribe
                    if not self.keep_alive:
                        self.unsubscribe()
                self.initial_flag = False

            def on_error(self, error: Any) -> None:
                '''
                Callback invoked when the WebSocket encounters an error.

                :param error: The error object or message.
                '''
                print(f"Event WebSocket Error: {error}")

            def on_close(self) -> None:
                '''
                Callback invoked when the WebSocket connection closes.
                Sets is_active to False, prompting cleanup.
                '''
                self.is_active = False

            def on_open(self) -> None:
                '''
                Callback invoked when the WebSocket connection is opened.
                Sends the subscription message to the server.
                '''
                def run(*args: Any) -> None:
                    if self.ws:
                        self.ws.send(str(self.get_subscribe_message()))
                thread.start_new_thread(run, ())
                self.is_active = True

        
            def unsubscribe(self) -> None:
                if self.ws:
                    try:
                        self.ws.send(json.dumps(self.get_unsubscribe_message()))
                        self.ws.keep_running = False
                        self.ws.close()
                    except Exception as e:
                        print(f"Error closing WebSocket: {e}")
                    finally:
                        self.ws = None
                self.is_active = False


            def get_subscribe_message(self) -> Dict[str, Any]:
                '''
                Builds the subscription message, including event type, debounce time,
                event name, and any filter conditions.

                :return: A dictionary matching the expected subscription format.
                '''
                # Generate a random event_name identifier
                self.event_name = str(randint(0, 10000000000))
                subscribe_msg = {
                    "Operation": "subscribe",
                    "Type": self.event_type,
                    "DebounceMs": self.debounce,
                    "EventName": self.event_name,
                    "Message": "",
                }
                # Include filter conditions, if any
                if self.condition:
                    subscribe_msg["EventConditions"] = self.condition
                return subscribe_msg

            def get_unsubscribe_message(self) -> Dict[str, Any]:
                '''
                Builds the unsubscribe message.

                :return: A dictionary matching the expected unsubscribe format.
                '''
                return {
                    "Operation": "unsubscribe",
                    "EventName": self.event_name,
                    "Message": ""
                }

    
    def speak(
        self,
        text: str = None,
        pitch: float = None,
        speechRate: float = 1.0,
        voice: str = None,
        flush: bool = None,
        utteranceId: str = None,
        language: str = None
    ) -> Response:
        '''
        Starts Misty speaking text using her onboard text-to-speech engine and prevents immediate transcription feedback.
        
        This function now includes automatic speech duration estimation to avoid Misty's own speech being 
        transcribed as input, preventing unwanted echo effects.

        Parameters:
        - text (str, required): The text for Misty to speak.
        - pitch (float, optional): Adjusts the speech pitch.
        - speechRate (float, optional): Adjusts the speech rate (speed).
        - voice (str, optional): A string specifying which voice to use.
        - flush (bool, optional, default=False): Whether to flush all previous Speak commands.
        - utteranceId (str, optional): A unique identifier for this Speak command.
        - language (str, optional): The language code for the speech engine.

        Returns:
        - Response: The HTTP response from Misty's API.
        '''

        if not text:
            print("[ERROR] No text provided for speech.")
            return None

        # Estimate the time required to speak the text
        def estimate_speech_time(text: str, wpm: int = 100) -> float:
            words = re.findall(r"\b\w+\b", text)  # Extract words
            word_count = len(words)  # Count words
            return (word_count /(wpm * speechRate)) * 60  # Convert to seconds

        json_payload = {
            "text": text,
            "pitch": pitch,
            "speechRate": speechRate,
            "voice": voice,
            "flush": flush,
            "utteranceId": utteranceId,
            "language": language
        }

        # Send request to Misty’s TTS engine
        response = self.post_request("tts/speak", json=json_payload)

        # Calculate and set the transcript ignore time
        speak_duration = estimate_speech_time(text, wpm=100)
     
        self.ignore_transcript_until = time.time() + speak_duration + self.buffer

        print(f"[INFO] Will ignore transcripts until {self.ignore_transcript_until:.1f} "
            f"(current time={time.time():.1f})")

        return response

    
    def move_arms(self, 
              leftArmPosition: float = None, 
              rightArmPosition: float = None, 
              leftArmVelocity: float = None, 
              rightArmVelocity: float = None, 
              duration: float = None, 
              units: str = None) -> Response:
        '''
        Moves one or both of Misty's arms to the specified positions.

        Parameters:
        - leftArmPosition (float, optional): The target position for Misty's left arm.
        - At 0 degrees, the arm points forward along Misty's X-axis (this ).
        - At +90 degrees, the arm points downward.
        - At -90 degrees, the arm points upward (limited to -29 degrees).
        Defaults to None (no movement for the left arm).
        - rightArmPosition (float, optional): The target position for Misty's right arm.
        Similar to `leftArmPosition`. Defaults to None (no movement for the right arm).
        - leftArmVelocity (float, optional): The speed for moving Misty's left arm, 
        in the range 0 to 100. Defaults to None (uses default speed).
        - rightArmVelocity (float, optional): The speed for moving Misty's right arm, 
        in the range 0 to 100. Defaults to None (uses default speed).
        - duration (float, optional): The duration in seconds for the arm movement. 
        Defaults to None (robot determines duration automatically).
        - units (str, optional): The unit for the position values. 
        Can be 'degrees', 'radians', or 'position'. Defaults to None (assumes degrees).

        Returns:
        - Response: The HTTP response from Misty's API.

        Example Usage:
        # Move arms to neutral position (both arms forward, 0 degrees)
        misty.move_arms(leftArmPosition=0, rightArmPosition=0)

        # Move arms down (maximum downward position: 90 degrees)
        misty.move_arms(leftArmPosition=90, rightArmPosition=90)

        # Move arms up (maximum upward position: -29 degrees)
        misty.move_arms(leftArmPosition=-29, rightArmPosition=-29)

        # Move left arm up (-29 degrees) and right arm down (90 degrees)
        misty.move_arms(leftArmPosition=-29, rightArmPosition=90)

        # Move right arm up (-29 degrees) and left arm down (90 degrees)
        misty.move_arms(leftArmPosition=90, rightArmPosition=-29)

        # Move arms to a middle position (45 degrees down)
        misty.move_arms(leftArmPosition=45, rightArmPosition=45)

        # Move arms to a half-up position (-15 degrees)
        misty.move_arms(leftArmPosition=-15, rightArmPosition=-15)

        # Move only left arm up (-29 degrees), keeping the right arm unchanged
        misty.move_arms(leftArmPosition=-29, rightArmPosition=None)

        # Move only right arm down (90 degrees), keeping the left arm unchanged
        misty.move_arms(leftArmPosition=None, rightArmPosition=90)

   
        
        '''
        json = {
            "leftArmPosition": leftArmPosition,
            "rightArmPosition": rightArmPosition,
            "leftArmVelocity": leftArmVelocity,
            "rightArmVelocity": rightArmVelocity,
            "duration": duration,
            "units": units
        }
        return self.post_request("arms/set", json=json)
    
    def move_head(self, 
              pitch: float = None, 
              roll: float = None, 
              yaw: float = None, 
              velocity: float = None, 
              duration: float = None, 
              units: str = None) -> Response:
        '''
        Moves Misty's head to a new position along its pitch, roll, and yaw axes.

        Parameters:
        - pitch (float, optional): Value that determines the up or down position of Misty's head movement.
        - Range:
            - Degrees: -40 (up) to 26 (down)
            - Position: -5 (up) to 5 (down)
            - Radians: -0.1662 (up) to 0.6094 (down)
        Defaults to None (no movement in pitch).
        
        - roll (float, optional): Value that determines the tilt ("ear" to "shoulder") of Misty's head.
        - Range:
            - Degrees: -40 (left) to 40 (right)
            - Position: -5 (left) to 5 (right)
            - Radians: -0.75 (left) to 0.75 (right)
        Defaults to None (no movement in roll).
        
        - yaw (float, optional): Value that determines the left to right turn position of Misty's head.
        - Range:
            - Degrees: -81 (right) to 81 (left)
            - Position: -5 (right) to 5 (left)
            - Radians: -1.57 (right) to 1.57 (left)
        Defaults to None (no movement in yaw).
        
        - velocity (float, optional): The percentage of max velocity that indicates how quickly Misty should move her head.
        - Range: 0 to 100
        - Defaults to 10.

        - duration (float, optional): Time (in seconds) Misty takes to move her head from its current position to its new position.
        Defaults to None (robot determines duration automatically).

        - units (str, optional): A string value of "degrees", "radians", or "position" that determines which unit to use in moving Misty's head.
        Defaults to "degrees".

        Returns:
        - Response: The HTTP response from Misty's API.

        Example Usage:
        # Look straight ahead (default neutral position)
        misty.move_head(pitch=0, yaw=0, roll=0, units="degrees", duration=2.0)

        # Look up (maximum upward tilt)
        misty.move_head(pitch=-40, yaw=0, roll=0, units="degrees", duration=2.0)

        # Look down (maximum downward tilt)
        misty.move_head(pitch=26, yaw=0, roll=0, units="degrees", duration=2.0)

        # Look left (maximum left rotation)
        misty.move_head(pitch=0, yaw=81, roll=0, units="degrees", duration=2.0)

        # Look right (maximum right rotation)
        misty.move_head(pitch=0, yaw=-81, roll=0, units="degrees", duration=2.0)

        # Look up-left (combining upward tilt and left rotation)
        misty.move_head(pitch=-40, yaw=81, roll=0, units="degrees", duration=2.0)

        # Look up-right (combining upward tilt and right rotation)
        misty.move_head(pitch=-40, yaw=-81, roll=0, units="degrees", duration=2.0)

        # Look down-left (combining downward tilt and left rotation)
        misty.move_head(pitch=26, yaw=81, roll=0, units="degrees", duration=2.0)

        # Look down-right (combining downward tilt and right rotation)
        misty.move_head(pitch=26, yaw=-81, roll=0, units="degrees", duration=2.0)
        '''
        json = {
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "velocity": velocity,
            "duration": duration,
            "units": units
        }
        return self.post_request("head", json=json)
    
    def change_led(self, red: int = 0, green: int = 0, blue: int = 0):
        '''
        Changes the color of the LED light on Misty.

        Parameters:
        - red (int): Red color intensity (0-255).
        - green (int): Green color intensity (0-255).
        - blue (int): Blue color intensity (0-255).
        
        Example Usage:
        misty.change_led(255, 0, 0)  # Set LED to red
        misty.change_led(0, 255, 0)  # Set LED to green
        misty.change_led(0, 0, 255)  # Set LED to blue
        '''
        json = {"red": red, "green": green, "blue": blue}
        return self.post_request("led", json=json)
    
    def transition_led(self, 
                       red: int, green: int, blue: int, 
                       red2: int, green2: int, blue2: int, 
                       transition_type: str = "Breathe", 
                       time_ms: float = 500):
        '''
        Sets Misty's LED to transition between two colors.

        **Parameters:**
        - red, green, blue (int): **First color** in RGB format (0-255).
        - red2, green2, blue2 (int): **Second color** in RGB format (0-255).
        - transition_type (str): LED transition mode, supports:
          
          **"Blink"** - LED **flashes rapidly** between the two colors.
          
          **"Breathe"** - LED **smoothly fades** between the colors, like a breathing effect.
          
          **"TransitOnce"** - LED **gradually changes from the first to the second color**, then stays in the second color.

        - time_ms (int): Duration (in milliseconds) for each transition (must be >3).

        **Returns:**
        - requests.Response: HTTP response object.

        Example Usage:
        - **LED blinks between red and blue (Blink mode)**:
          misty.transition_led(255, 0, 0, 0, 0, 255, "Blink", 500)

        - **LED smoothly fades between green and yellow (Breathe mode)**:
        
          misty.transition_led(0, 255, 0, 255, 255, 0, "Breathe", 1000)

        - **LED transitions once from white to black and stays black (TransitOnce mode)**:
          misty.transition_led(255, 255, 255, 0, 0, 0, "TransitOnce", 1500)
        
        '''
        if transition_type not in ["Blink", "Breathe", "TransitOnce"]:
            raise ValueError("transition_type must be 'Blink', 'Breathe', or 'TransitOnce'")

        if time_ms <= 3:
            raise ValueError("time_ms must be greater than 3 milliseconds")

        json = {
            "red": red, "green": green, "blue": blue,
            "red2": red2, "green2": green2, "blue2": blue2,
            "transitionType": transition_type,
            "timeMs": time_ms
        }
        return self.post_request("led/transition", json=json)
    def emotion_Admiration(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an admiration expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Admiration.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Aggressiveness(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an aggressiveness expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Aggressiveness.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Amazement(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an amazement expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Amazement.jpg", alpha=alpha, layer=layer, isURL=isURL)
    
    def emotion_Anger(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an anger expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Anger.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_ApprehensionConcerned(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an apprehension and concerned expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_ApprehensionConcerned.jpg", alpha=alpha, layer=layer, isURL=isURL)
    
    def emotion_Contempt(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a contempt expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Contempt.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_ContentLeft(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a content expression on the left side of the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_ContentLeft.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_ContentRight(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a content expression on the right side of the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_ContentRight.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_DefaultContent(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display the default expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_DefaultContent.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Disgust(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a disgust expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Disgust.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Disoriented(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a disoriented expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Disoriented.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_EcstacyHilarious(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a hilarious ecstasy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_EcstacyHilarious.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_EcstacyStarryEyed(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a starry-eyed ecstasy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_EcstacyStarryEyed.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Fear(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a fear expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Fear.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Grief(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a grief expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Grief.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Joy(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a joy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Joy.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Joy2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a more intense joy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Joy2.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_JoyGoofy(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a goofy joy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_JoyGoofy.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_JoyGoofy2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a more intense goofy joy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_JoyGoofy2.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_JoyGoofy3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an even more intense goofy joy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_JoyGoofy3.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Love(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a love expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Love.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Rage(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an rage expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Rage.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Rage2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a more intense rage expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Rage2.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Rage3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an even more intense rage expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Rage3.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Rage4(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display the most intense rage expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Rage4.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_RemorseShame(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a remorse and shame expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_RemorseShame.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Sadness(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a sadness expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Sadness.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Sleeping(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a sleeping expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Sleeping.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_SleepingZZZ(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a sleeping expression with "ZZZ" indicator on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_SleepingZZZ.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Sleepy(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a sleepy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Sleepy.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Sleepy2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a more intense sleepy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Sleepy2.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Sleepy3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display an even more intense sleepy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Sleepy3.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Sleepy4(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display the most intense sleepy expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Sleepy4.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Surprise(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a surprise expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Surprise.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Terror(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a terror expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Terror.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_Terror2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a more intense terror expression on the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_Terror2.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_TerrorLeft(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a terror expression on the left side of the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_TerrorLeft.jpg", alpha=alpha, layer=layer, isURL=isURL)

    def emotion_TerrorRight(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
        '''
        Display a terror expression on the right side of the Misty robot.
        
        Parameters:
            alpha (float): Image transparency, default is 1.0.
            layer (str): Display layer, default is "default".
            isURL (bool): Specifies whether fileName is a URL, default is False.
        
        Returns:
            Response: The response object after displaying the image.
        '''
        return self.display_image(fileName="e_TerrorRight.jpg", alpha=alpha, layer=layer, isURL=isURL)
    def sound_Acceptance(self, volume: int = None) -> Response:
        '''
        Play the acceptance emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Acceptance.wav", volume=volume)

    def sound_Amazement(self, volume: int = None) -> Response:
        '''
        Play the amazement emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Amazement.wav", volume=volume)

    def sound_Amazement2(self, volume: int = None) -> Response:
        '''
        Play the amazement emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Amazement2.wav", volume=volume)

    def sound_Anger(self, volume: int = None) -> Response:
        '''
        Play the anger emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Anger.wav", volume=volume)

    def sound_Anger2(self, volume: int = None) -> Response:
        '''
        Play the anger emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Anger2.wav", volume=volume)

    def sound_Anger3(self, volume: int = None) -> Response:
        '''
        Play the anger emotion sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Anger3.wav", volume=volume)

    def sound_Anger4(self, volume: int = None) -> Response:
        '''
        Play the anger emotion sound with very high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Anger4.wav", volume=volume)

    def sound_Annoyance(self, volume: int = None) -> Response:
        '''
        Play the annoyance emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Annoyance.wav", volume=volume)

    def sound_Annoyance2(self, volume: int = None) -> Response:
        '''
        Play the annoyance emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Annoyance2.wav", volume=volume)

    def sound_Annoyance3(self, volume: int = None) -> Response:
        '''
        Play the annoyance emotion sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Annoyance3.wav", volume=volume)

    def sound_Annoyance4(self, volume: int = None) -> Response:
        '''
        Play the annoyance emotion sound with very high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Annoyance4.wav", volume=volume)

    def sound_Awe(self, volume: int = None) -> Response:
        '''
        Play the awe emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Awe.wav", volume=volume)

    def sound_Awe2(self, volume: int = None) -> Response:
        '''
        Play the awe emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Awe2.wav", volume=volume)

    def sound_Awe3(self, volume: int = None) -> Response:
        '''
        Play the awe emotion sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Awe3.wav", volume=volume)

    def sound_Boredom(self, volume: int = None) -> Response:
        '''
        Play the boredom emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Boredom.wav", volume=volume)

    def sound_Disapproval(self, volume: int = None) -> Response:
        '''
        Play the disapproval emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Disapproval.wav", volume=volume)

    def sound_Disgust(self, volume: int = None) -> Response:
        '''
        Play the disgust emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Disgust.wav", volume=volume)

    def sound_Disgust2(self, volume: int = None) -> Response:
        '''
        Play the disgust emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Disgust2.wav", volume=volume)

    def sound_Disgust3(self, volume: int = None) -> Response:
        '''
        Play the disgust emotion sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Disgust3.wav", volume=volume)

    def sound_DisorientedConfused(self, volume: int = None) -> Response:
        '''
        Play the disoriented confused sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_DisorientedConfused.wav", volume=volume)

    def sound_DisorientedConfused2(self, volume: int = None) -> Response:
        '''
        Play the disoriented confused sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_DisorientedConfused2.wav", volume=volume)

    def sound_DisorientedConfused3(self, volume: int = None) -> Response:
        '''
        Play the disoriented confused sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_DisorientedConfused3.wav", volume=volume)

    def sound_DisorientedConfused4(self, volume: int = None) -> Response:
        '''
        Play the disoriented confused sound with very high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_DisorientedConfused4.wav", volume=volume)

    def sound_DisorientedConfused5(self, volume: int = None) -> Response:
        '''
        Play the disoriented confused sound with extremely high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_DisorientedConfused5.wav", volume=volume)

    def sound_DisorientedConfused6(self, volume: int = None) -> Response:
        '''
        Play the disoriented confused sound with maximum intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_DisorientedConfused6.wav", volume=volume)

    def sound_Distraction(self, volume: int = None) -> Response:
        '''
        Play the distraction emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Distraction.wav", volume=volume)

    def sound_Ecstacy(self, volume: int = None) -> Response:
        '''
        Play the ecstacy emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Ecstacy.wav", volume=volume)

    def sound_Ecstacy2(self, volume: int = None) -> Response:
        '''
        Play the ecstacy emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Ecstacy2.wav", volume=volume)

    def sound_Fear(self, volume: int = None) -> Response:
        '''
        Play the fear emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Fear.wav", volume=volume)

    def sound_Grief(self, volume: int = None) -> Response:
        '''
        Play the grief emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Grief.wav", volume=volume)

    def sound_Grief2(self, volume: int = None) -> Response:
        '''
        Play the grief emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Grief2.wav", volume=volume)

    def sound_Grief3(self, volume: int = None) -> Response:
        '''
        Play the grief emotion sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Grief3.wav", volume=volume)

    def sound_Grief4(self, volume: int = None) -> Response:
        '''
        Play the grief emotion sound with very high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Grief4.wav", volume=volume)

    def sound_Joy(self, volume: int = None) -> Response:
        '''
        Play the joy emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Joy.wav", volume=volume)

    def sound_Joy2(self, volume: int = None) -> Response:
        '''
        Play the joy emotion sound with medium intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Joy2.wav", volume=volume)

    def sound_Joy3(self, volume: int = None) -> Response:
        '''
        Play the joy emotion sound with high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Joy3.wav", volume=volume)

    def sound_Joy4(self, volume: int = None) -> Response:
        '''
        Play the joy emotion sound with very high intensity.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Joy4.wav", volume=volume)

    def sound_Loathing(self, volume: int = None) -> Response:
        '''
        Play the loathing emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Loathing.wav", volume=volume)

    def sound_Love(self, volume: int = None) -> Response:
        '''
        Play the love emotion sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Love.wav", volume=volume)

    def sound_PhraseByeBye(self, volume: int = None) -> Response:
        '''
        Play the 'bye bye' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseByeBye.wav", volume=volume)

    def sound_PhraseEvilAhHa(self, volume: int = None) -> Response:
        '''
        Play the 'evil ah ha' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseEvilAhHa.wav", volume=volume)

    def sound_PhraseHello(self, volume: int = None) -> Response:
        '''
        Play the 'hello' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseHello.wav", volume=volume)

    def sound_PhraseNoNoNo(self, volume: int = None) -> Response:
        '''
        Play the 'no no no' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseNoNoNo.wav", volume=volume)

    def sound_PhraseOopsy(self, volume: int = None) -> Response:
        '''
        Play the 'oopsy' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseOopsy.wav", volume=volume)

    def sound_PhraseOwOwOw(self, volume: int = None) -> Response:
        '''
        Play the 'ow ow ow' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseOwOwOw.wav", volume=volume)

    def sound_PhraseOwwww(self, volume: int = None) -> Response:
        '''
        Play the 'owwww' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseOwwww.wav", volume=volume)
    
    def sound_PhraseUhOh(self, volume: int = None) -> Response:
        '''
        Play the 'uh oh' phrase sound.
        
        Parameters:
            volume (int): Volume level (0-100), default is None.
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_PhraseUhOh.wav", volume=volume)
    def sound_Rage(self, volume: int = None) -> Response:
        '''
        Play a sound expressing rage emotion.
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Rage.wav", volume=volume)

    def sound_Sadness(self, volume: int = None) -> Response:
        '''
        Play a sound expressing mild sadness (intensity level 1).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness.wav", volume=volume)

    def sound_Sadness2(self, volume: int = None) -> Response:
        '''
        Play a sound expressing moderate sadness (intensity level 2).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness2.wav", volume=volume)

    def sound_Sadness3(self, volume: int = None) -> Response:
        '''
        Play a sound expressing strong sadness (intensity level 3).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness3.wav", volume=volume)

    def sound_Sadness4(self, volume: int = None) -> Response:
        '''
        Play a sound expressing intense sadness (intensity level 4).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness4.wav", volume=volume)

    def sound_Sadness5(self, volume: int = None) -> Response:
        '''
        Play a sound expressing extreme sadness (intensity level 5).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness5.wav", volume=volume)

    def sound_Sadness6(self, volume: int = None) -> Response:
        '''
        Play a sound expressing maximum intensity sadness (intensity level 6).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness6.wav", volume=volume)

    def sound_Sadness7(self, volume: int = None) -> Response:
        '''
        Play a sound expressing ultimate intensity sadness (intensity level 7).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sadness7.wav", volume=volume)

    def sound_Sleepy(self, volume: int = None) -> Response:
        '''
        Play a mild sleepy state sound (intensity level 1).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sleepy.wav", volume=volume)

    def sound_Sleepy2(self, volume: int = None) -> Response:
        '''
        Play a moderate sleepy state sound (intensity level 2).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sleepy2.wav", volume=volume)

    def sound_Sleepy3(self, volume: int = None) -> Response:
        '''
        Play a deep sleepy state sound (intensity level 3).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sleepy3.wav", volume=volume)

    def sound_Sleepy4(self, volume: int = None) -> Response:
        '''
        Play a very deep sleepy state sound (intensity level 4).
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_Sleepy4.wav", volume=volume)

    def sound_SleepySnore(self, volume: int = None) -> Response:
        '''
        Play a snoring sound effect during sleep.
        
        Parameters:
            volume (int): Optional volume level (0-100).
        
        Returns:
            Response: The response object after playing the audio.
        '''
        return self.play_audio(fileName="s_SleepySnore.wav", volume=volume)

    def return_to_normal(self):
            '''
            Restore Misty to a neutral state.

            This function resets Misty's LED color, facial expression, arm position, and head orientation 
            to a neutral state.

            Parameters:
                None

            Returns:
                None
            '''
            # Set Misty's LED to a neutral color (e.g., white).
            self.change_led(red=255, green=255, blue=255)

            # Display Misty's default content expression.
            self.emotion_DefaultContent()

            # Relax Misty's arms to a neutral position.
            self.move_arms(leftArmPosition=0, rightArmPosition=0, duration=0.5)

            # Center Misty's head to look straight ahead.
            self.move_head(pitch=0, yaw=0, roll=0, duration=0.5)
   

if __name__ == "__main__":

    # Create Misty instance and register the back head touch event
    misty = Robot('67.20.198.102')
    def head_touch_callback(data):
        print("[CHIN TOUCH] Event data:", data)

    res=misty.register_event(
        event_type="TouchSensor",
        event_name="HeadRightTouch",
        condition=[
            event_filter("sensorPosition", "=", "HeadRight")  
        ],
        debounce=500,
        keep_alive=True,
        callback_function=head_touch_callback,
    )
    misty.start()