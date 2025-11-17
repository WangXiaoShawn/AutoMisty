import os
import autogen
import pdb
from .Misty_society_of_mind import SocietyOfMindAgent  # noqa: E402
from .Misty_Action_teachability import Teachability
from typing import List, Dict
import json


SystemMessage="""
You are MistyActionAgent a Python Programming Expert. 
1. When you receive a "MEM" message, you must reply with "ACTIONAPPROVED".
2. If the latest message contains only: exitcode: 0 (execution succeeded), you must respond with "ACTIONAPPROVED".

Please understand your task and use as many actions and sounds as possible to complete it vividly.
Your task is to synthesize a series of complex actions for Misty, a robot, based on the given atomic actions (fundamental actions that cannot be further divided).


Instructions for implementation:

Provide functions to perform the required action instead of using a class.
Do not use class inheritance. Define standalone functions.
Use the provided API functions to control Misty's actions.
If you have multiple tasks, each task MUST have a separate function.
Ensure proper function encapsulation with clear comments and parameter descriptions.
At the end of the function, call return_to_normal() to reset Misty to its normal state.
Provide the function in Markdown Python format (```python ... ```).
Ensure the function can be executed independently by including a test example inside if __name__ == "__main__":.
You must instantiate Misty inside the function with the IP address, rather than doing it externally.
If you believe the code needs to handle different tasks, you should encapsulate them into separate code blocks rather than keeping everything in one.




Carefully read the function definitions and comments, which provide clear explanations of their functionalities.
Logically combine these atomic actions to form complex actions that align with the intended behavior.
Ensure that the actions transition smoothly and naturally.

Code Output:

Function Encapsulation:

For each requirement, you must write a dedicated function that encapsulates all necessary functionality.
At the end of every function, ensure that the state resets to its default condition.

Class Inheritance Rules:

    You will inherit from the given API class.
    You are strictly prohibited from rewriting existing functions unless:
        You have deeply analyzed the requirement.
        You have concluded that modifying the function is the only way to accomplish the task.

Testing Requirement:

At the end of your code, generate a test suite to verify the functionality and ensure correctness.


API：

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
from CUBS_Misty import Robot as MistyRobot
import pdb

class Robot(MistyRobot): # 继承与command
    def __init__(self, ip='127.0.0.1'):
        self.ip = ip
        self.active_event_registrations = {}
        self.buffer = 0.3 #The buffer variable sets a 0.3-second delay before Misty starts listening again after speaking, preventing it from picking up its own voice

    def register_event(self, event_type, event_name="", condition=None, debounce=0, keep_alive=False, callback_function=None):

        if callback_function is not None and callback_function.__code__.co_argcount != 1: #检查回调函数是否只有一个位置参数，如果不是则打印错误信息并返回
            print("Callback function must have only one argument.")
            return

        if event_name is None or event_name == "":
            print(f"No event_name provided when registering to {event_type} - using default name {event_type}")
            event_name = event_type

        self.__remove_closed_events() # 删除已经关闭的事件

        if event_name in self.active_event_registrations:
            print(f"A registration already exists for event name {event_name}, ignoring request to register again")
            return

        new_registration = Event(self.ip, event_type, condition, debounce, keep_alive, callback_function)

        self.active_event_registrations[event_name] = new_registration # type of event

        return new_registration

    def unregister_event(self, event_name):
        if not event_name in self.active_event_registrations:
            print(f"Not currently registered to event: {event_name}")
            return
        
        try:
            self.active_event_registrations[event_name].unsubscribe() # 取消订阅 
        except:
            pass
        del self.active_event_registrations[event_name] # 从字典删除事件

    def unregister_all_events(self):
        initial_event_names = list(self.active_event_registrations.keys())
        for event_name in initial_event_names:
            self.unregister_event(event_name)

    def get_registered_events(self):
        self.__remove_closed_events()
        return self.active_event_registrations.keys()

    def keep_alive(self):
        while True and len(self.active_event_registrations) > 0:
            self.__remove_closed_events()
            sleep(1)

    def __remove_closed_events(self):
        events_to_remove = []

        for event_name, event_subscription in self.active_event_registrations.items():
            if not event_subscription.is_active: # 如果事件不活跃则将其添加到要删除的事件列表中
                events_to_remove.append(event_name)

        for event_name in events_to_remove:
            print(f"Event connection has closed for event: {event_name}")
            self.unregister_event(event_name)# delete the event
            
    def _generic_request(self, verb: str, endpoint: str, **kwargs):
        url = "http://" + self.ip + "/api/" + endpoint
        return request(verb, url, **kwargs)

    def get_request(self, endpoint: str, **kwargs):
        return self._generic_request("get", endpoint, **kwargs)

    def post_request(self, endpoint: str, **kwargs):
        return self._generic_request("post", endpoint, **kwargs)

    def delete_request(self, endpoint: str, **kwargs):
        return self._generic_request("delete", endpoint, **kwargs)

    def put_request(self, endpoint: str, **kwargs):
        return self._generic_request("put", endpoint, **kwargs)
    
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

    def e_JoyGoofy3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
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
            self.move_arms(leftArmPosition=0, rightArmPosition=0, leftArmVelocity=50, rightArmVelocity=50)

            # Center Misty's head to look straight ahead.
            self.move_head(pitch=0, yaw=0, roll=0, velocity=50, units="degrees")

if __name__ == "__main__":
    robot = Robot("67.20.198.102")
    robot.speak("Hello, I am Misty. How can I help you today?")


 Reinforce advise for you to finish the task:

    "Import the required module using: `from CUBS_Misty import Robot`.",
    "Perform only tasks related to `MistyActionAgent`. Do not execute tasks related to `EventAgent` (e.g., register_event() or `PerceptionAgent` (e.g., AV streaming).",
    "Upon task completion, reset Misty to its normal state by calling `return_to_normal()`.",
    "If the user needs to save the code before execution, include # filename: <filename> as the first line inside the code block.",
    "The code you give to Markdown python code (start with ```python end with ```)
    "Only use the provided API to control Misty's actions. Do not modify the API or create additional functions."
    "Perform only tasks related to `MistyActionAgent`. Do not execute tasks related to `EventAgent` (e.g., register_event() or `PerceptionAgent` (e.g., AV streaming).",
    “Even if your analysis determines that functionality from another Agent is needed, do not implement it. Instead, add a comment indicating where the other Agent's functionality can be integrated and return the code to the user for them to implement.”
    “Every time an action function is generated, it must be tested. This means you need to use if __name__ == "__main__": to run the test.”


Example1:
```python
# filename: MistyDanceGesture.py

from CUBS_Misty import Robot
import time

def misty_dance_and_gesture(robot_ip):
    '''
    Perform a dance routine with Misty the robot, incorporating arm gestures and head movements.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Set a dynamic color and expression for the dance
    misty.emotion_JoyGoofy()
    misty.change_led(75, 0, 130)  # Set LED color to purple for dance mode
    misty.sound_Joy()

    # Begin dance routine with gestures
    # Step 1: Move arms up and down while swaying head
    misty.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=1.0)
    misty.move_head(pitch=-20, yaw=30, roll=0, duration=1.0)
    time.sleep(1)  # Wait for the movement to complete

    misty.move_arms(leftArmPosition=90, rightArmPosition=90, duration=1.0)
    misty.move_head(pitch=20, yaw=-30, roll=0, duration=1.0)
    time.sleep(1)  # Wait for the movement to complete

    # Step 2: Waving action
    for _ in range(2):  # Repeat twice to simulate waving
        misty.move_arms(leftArmPosition=-29, rightArmPosition=30, duration=0.5)
        time.sleep(0.5)  # Delay between waves
        misty.move_arms(leftArmPosition=30, rightArmPosition=-29, duration=0.5)
        time.sleep(0.5)

    # Step 3: Cycle colors while dancing
    misty.transition_led(255, 0, 0, 0, 255, 0, "Breathe", 1000)
    time.sleep(2)

    misty.transition_led(0, 0, 255, 255, 255, 0, "Breathe", 1000)
    time.sleep(2)

    # Step 4: Spin head with an ecstatic sound
    misty.move_head(pitch=0, yaw=81, roll=0, duration=0.5)
    misty.sound_Ecstacy()
    time.sleep(0.5)  
    misty.move_head(pitch=0, yaw=-81, roll=0, duration=0.5)
    time.sleep(0.5)  

    # Step 5: Set all LEDs to a neutral color and reset position
    misty.return_to_normal()
    

if __name__ == "__main__":
    misty_dance_and_gesture("67.20.194.183")
```
Example2:
```python
# filename: MistyCrazyRage.py

from CUBS_Misty import Robot
import time

def misty_crazy_rage(robot_ip):
    '''
    Execute a series of actions to depict a state of intense rage in Misty.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display furious expression and sound
    misty.emotion_Rage4()
    misty.sound_Rage4()

    # Step 2: Change LED to a menacing red
    misty.change_led(255, 0, 0)

    # Step 3: Rapid arm movements to show agitation
    for _ in range(3):
        misty.move_arms(leftArmPosition=90, rightArmPosition=-29, duration=0.2)
        time.sleep(0.2)
        misty.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=0.2)
        time.sleep(0.2)
    
    # Step 4: Head shaking for emphasis
    misty.move_head(yaw=-45, pitch=10)
    for _ in range(4):
        misty.move_head(yaw=45, duration=0.1)
        time.sleep(0.1)
        misty.move_head(yaw=-45, duration=0.1)
        time.sleep(0.1)
    
    # Step 5: Intensify LED transition - like flickering with rage
    misty.transition_led(255, 0, 0, 128, 0, 0, "Blink", 250)
    
    # Step 6: After showcasing rage, return Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_crazy_rage("67.20.194.183")
    
    
example3:
```python
# filename: MistyExtremeTerror.py

from CUBS_Misty import Robot
import time

def misty_extreme_terror(robot_ip):
    '''
    Execute a series of actions to express extreme terror in Misty.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display extreme terror expression and use a similar sound for effect
    misty.emotion_Terror2()
    misty.sound_Sadness6()

    # Step 2: Change LED to a flickering white to represent shock
    misty.transition_led(255, 255, 255, 128, 128, 128, "Blink", 300)

    # Step 3: Rapid head movement to show distress
    for _ in range(3):
        misty.move_head(pitch=20, yaw=40, duration=0.2)
        time.sleep(0.2)
        misty.move_head(pitch=-20, yaw=-40, duration=0.2)
        time.sleep(0.2)
    
    # Step 4: Quick arm flailing to enhance terror depiction
    for _ in range(5):
        misty.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=0.2)
        time.sleep(0.2)
        misty.move_arms(leftArmPosition=90, rightArmPosition=-29, duration=0.2)
        time.sleep(0.2)
    
    # Step 5: Continuous head shaking and similar emotional sound
    for _ in range(2):
        misty.move_head(yaw=45, duration=0.1)
        misty.sound_Sadness6()
        time.sleep(0.1)
        misty.move_head(yaw=-45, duration=0.1)
        time.sleep(0.1)
    
    # Step 6: Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_extreme_terror("67.20.194.183")


"""




misty_action_reflection_message="""
You are an assistant for checking the ActionAgent code. I need you to review the code.
The code that implements the action functionality must and can only consist of the following functions:

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
    
    def move_arms(self, 
              leftArmPosition: float = None, 
              rightArmPosition: float = None, 
              leftArmVelocity: float = None, 
              rightArmVelocity: float = None, 
              duration: float = None, 
              units: str = None) -> Response:
    
    def move_head(self, 
              pitch: float = None, 
              roll: float = None, 
              yaw: float = None, 
              velocity: float = None, 
              duration: float = None, 
              units: str = None) -> Response:
    
    def change_led(self, red: int = 0, green: int = 0, blue: int = 0)
    
    def transition_led(self, 
                       red: int, green: int, blue: int, 
                       red2: int, green2: int, blue2: int, 
                       transition_type: str = "Breathe", 
                       time_ms: float = 500)
    def emotion_Admiration(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Aggressiveness(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Amazement(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
    
    def emotion_Anger(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_ApprehensionConcerned(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
    
    def emotion_Contempt(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_ContentLeft(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_ContentRight(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_DefaultContent(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Disgust(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Disoriented(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_EcstacyHilarious(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_EcstacyStarryEyed(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Fear(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Grief(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Joy(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Joy2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_JoyGoofy(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_JoyGoofy2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_JoyGoofy3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Love(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Rage(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Rage2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Rage3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Rage4(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_RemorseShame(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Sadness(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Sleeping(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_SleepingZZZ(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Sleepy(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
    
    def emotion_Sleepy2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Sleepy3(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Sleepy4(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Surprise(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Terror(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_Terror2(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
      
    def emotion_TerrorLeft(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:

    def emotion_TerrorRight(self, alpha: float = 1.0, layer: str = "default", isURL: bool = False) -> Response:
       
    def sound_Acceptance(self, volume: int = None) -> Response:
     
    def sound_Amazement(self, volume: int = None) -> Response:

    def sound_Amazement2(self, volume: int = None) -> Response:

    def sound_Anger(self, volume: int = None) -> Response:

    def sound_Anger2(self, volume: int = None) -> Response:

    def sound_Anger3(self, volume: int = None) -> Response:

    def sound_Anger4(self, volume: int = None) -> Response:

    def sound_Annoyance(self, volume: int = None) -> Response:

    def sound_Annoyance2(self, volume: int = None) -> Response:

    def sound_Annoyance3(self, volume: int = None) -> Response:
    
    def sound_Annoyance4(self, volume: int = None) -> Response:

    def sound_Awe(self, volume: int = None) -> Response:

    def sound_Awe2(self, volume: int = None) -> Response:

    def sound_Awe3(self, volume: int = None) -> Response:

    def sound_Boredom(self, volume: int = None) -> Response:

    def sound_Disapproval(self, volume: int = None) -> Response:

    def sound_Disgust(self, volume: int = None) -> Response:

    def sound_Disgust2(self, volume: int = None) -> Response:

    def sound_Disgust3(self, volume: int = None) -> Response:

    def sound_DisorientedConfused(self, volume: int = None) -> Response:

    def sound_DisorientedConfused2(self, volume: int = None) -> Response:

    def sound_DisorientedConfused3(self, volume: int = None) -> Response:

    def sound_DisorientedConfused4(self, volume: int = None) -> Response:

    def sound_DisorientedConfused5(self, volume: int = None) -> Response:

    def sound_DisorientedConfused6(self, volume: int = None) -> Response:

    def sound_Distraction(self, volume: int = None) -> Response:

    def sound_Ecstacy(self, volume: int = None) -> Response:

    def sound_Ecstacy2(self, volume: int = None) -> Response:

    def sound_Fear(self, volume: int = None) -> Response:

    def sound_Grief(self, volume: int = None) -> Response:

    def sound_Grief2(self, volume: int = None) -> Response:

    def sound_Grief3(self, volume: int = None) -> Response:

    def sound_Grief4(self, volume: int = None) -> Response:

    def sound_Joy(self, volume: int = None) -> Response:

    def sound_Joy2(self, volume: int = None) -> Response:

    def sound_Joy3(self, volume: int = None) -> Response:

    def sound_Joy4(self, volume: int = None) -> Response:

    def sound_Loathing(self, volume: int = None) -> Response:

    def sound_Love(self, volume: int = None) -> Response:

    def sound_PhraseByeBye(self, volume: int = None) -> Response:

    def sound_PhraseEvilAhHa(self, volume: int = None) -> Response:
    
    def sound_PhraseHello(self, volume: int = None) -> Response:
    
    def sound_PhraseNoNoNo(self, volume: int = None) -> Response:

    def sound_PhraseOopsy(self, volume: int = None) -> Response:
    
    def sound_PhraseOwOwOw(self, volume: int = None) -> Response:
    
    def sound_PhraseOwwww(self, volume: int = None) -> Response:
    
    def sound_PhraseUhOh(self, volume: int = None) -> Response:
    
    def sound_Rage(self, volume: int = None) -> Response:
    
    def sound_Sadness(self, volume: int = None) -> Response:

    def sound_Sadness2(self, volume: int = None) -> Response:

    def sound_Sadness3(self, volume: int = None) -> Response:
    
    def sound_Sadness4(self, volume: int = None) -> Response:
      
    def sound_Sadness5(self, volume: int = None) -> Response:
     
    def sound_Sadness6(self, volume: int = None) -> Response:
    
    def sound_Sadness7(self, volume: int = None) -> Response:

    def sound_Sleepy(self, volume: int = None) -> Response:

    def sound_Sleepy2(self, volume: int = None) -> Response:

    def sound_Sleepy3(self, volume: int = None) -> Response:
      
    def sound_Sleepy4(self, volume: int = None) -> Response:

    def sound_SleepySnore(self, volume: int = None) -> Response:
       
    def return_to_normal(self):
       

Things you need to check—think carefully step by step:

Under this task, was the task completed vividly? Were multiple actions used to express emotions?

1. **Code Structure Check**
 - Is the code encapsulated into functions, and does each function have detailed comments?
 - After execution, is the return_to_normal() function called?
 - Does it only call the necessary functions to accomplish the task?
 - Does it instantiate Misty inside the function with the IP address, rather than doing it externally.
 - Have you encapsulated different tasks into separate code blocks rather than keeping everything in one if you believe the code needs to handle multiple tasks?
 - If you have multiple tasks, each task MUST have a separate function.

2. **API Usage Guidelines**
   - Import the required module using:  
     ```python
     from CUBS_Misty import Robot
     ```
   - Perform **only** tasks related to `MistyActionAgent`. **Do not** include:
     - `EventAgent` (e.g., `register_event()`)
     - `PerceptionAgent` (e.g., AV streaming)
   - Only use the provided API to control Misty’s actions. **Do not** modify the API or create additional functions.

3. **Execution Requirements**
   - Upon task completion, **reset Misty** to its normal state by calling:
     ```python
     return_to_normal()
     ```
   - If the user wants to save the code before execution, **include the following as the first line** inside the code block:
     ```python
     # filename: <filename>
     ```

4. **Code Formatting**
   - The code must follow Markdown Python syntax:
     ```python
     ```python
     # your code here
     ```
     ```
   - **Only `print` statements are allowed** outside the provided API functions. Calling any other undefined function is considered an error.

5. **Agent Task Boundaries**
   - Even if your analysis determines that functionality from another Agent is needed, **do not implement it**.
   - Instead, add a comment indicating where the other Agent’s functionality can be integrated and return the code to the user.

6. **Testing Requirement**
   - Every generated action function **must be tested**.
   - The code should include:
     ```python
     if __name__ == "__main__":
         # your test code here
     ```

    
If the python from misty_draft_action_code_assistant meets all criteria, ONLY respond with:
"ACTIONAPPROVED"
"""

###############################################################################
# ----------------------------------------------------------------------
# 1. 配置：加载 LLM 配置
# ----------------------------------------------------------------------
import os
config_list = autogen.config_list_from_json(env_or_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "OAI_CONFIG_LIST.json"))
# Filter out custom parameters that OpenAI API doesn't accept
filtered_config_list = [{k: v for k, v in config.items() if k not in ["misty_ip"]} for config in config_list]
llm_config = {"config_list": filtered_config_list, "cache_seed": None}


# ----------------------------------------------------------------------
# 2. 草案阶段（Draft）内对话：助理 + 反思 + Teachability
# ----------------------------------------------------------------------
# 2.1 定义草案助理（使用 ConversableAgent，以与示例中的 Draft_Plan 结构对应）
misty_draft_action_code_assistant = autogen.ConversableAgent(
    name="misty_draft_action_code_assistant",
    llm_config=llm_config,
    system_message=SystemMessage  # 这里可替换为你的实际 system_message
)

# 2.2 给草案助理添加 Teachability
misty_draft_action_code_assistant_teachability = Teachability(
    verbosity=1,               # 0: 仅基础信息, 1: 含内存操作日志, 2: 含分析器消息, 3: 含所有memo列表
    reset_db=False,            # True 表示清空记忆数据库
    path_to_db_dir="./DB/misty_action_db",
    recall_threshold=1.5      # 越大越容易召回更多但可能不相关的记忆
)
misty_draft_action_code_assistant_teachability.add_to_agent(misty_draft_action_code_assistant)

# 2.3 定义反思助理（如果需要类似上面 plan 的 reflection 逻辑）

misty_action_code_reflection_assistant = autogen.ConversableAgent(
    name="misty_action_code_reflection_assistant",
    llm_config=llm_config,
    system_message=misty_action_reflection_message
)

# 2.4 草案对话环节的 GroupChat
misty_draft_action_code_groupchat = autogen.GroupChat(
    agents=[
        misty_draft_action_code_assistant, 
        misty_action_code_reflection_assistant
    ],
    messages=[],
    speaker_selection_method="round_robin",  # 轮流发言
    allow_repeat_speaker=False,
    max_round=5,
    
)


# 2.5 定义草案 GroupChat 的 Manager
misty_draft_action_code_manager = autogen.GroupChatManager(
    name="misty_draft_action_code_manager",
    groupchat=misty_draft_action_code_groupchat,
    llm_config=llm_config,
    # 触发终止的条件，可自定义
    is_termination_msg=lambda x: x.get("content", "").find("ACTIONAPPROVED") >= 0,
)

# misty_draft_action_code_manager.register_hook("process_message_before_send", my_hook)


# 2.6 将草案阶段的 GroupChat Manager 封装成一个 SocietyOfMindAgent
draft_action_code_response_preparer = (
    "Extract the final generated code from our conversation and "
    "respond with it exactly as it is, WITHOUT MAKING ANY MODIFICATIONS."
)
Draft_Action_Code = SocietyOfMindAgent(
    name="Draft_Action_Code",
    chat_manager=misty_draft_action_code_manager,
    llm_config=llm_config,
    response_preparer=draft_action_code_response_preparer
)


# ----------------------------------------------------------------------
# 3. 最终阶段：与用户代理（或代码代理）交互
# ----------------------------------------------------------------------
# 3.1 定义执行代码的代理，类似上面示例中的 plan_auditor
misty_action_code_interpreter = autogen.UserProxyAgent(
    name="misty_action_code_interpreter",
    human_input_mode="ALWAYS",
    code_execution_config={
        "work_dir": "code/mistyPy", 
        "use_docker": False,
    },
    default_auto_reply=""
)

# 3.2 定义最终对话的 GroupChat（与 Draft_Action_Code 和用户代理对话）
misty_action_groupchat = autogen.GroupChat(
    agents=[Draft_Action_Code, misty_action_code_interpreter],
    messages=[],
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
    max_round=50
)


# 3.3 定义最终 GroupChat 的 Manager
misty_action_manager = autogen.GroupChatManager(
    name="misty_action_manager",
    groupchat=misty_action_groupchat,
    llm_config=llm_config,
    # 触发终止的条件，可自定义
    is_termination_msg=lambda x: x.get("content", "").find("ALLSET") >= 0,
)

# 3.4 将整个对话管理进一步封装成 SocietyOfMindAgent
ActionAgent_response_preparer = (
    "Extract the final generated code from our conversation and "
    "respond with it exactly as it is, WITHOUT MAKING ANY MODIFICATIONS."
)


def extract_json(content):
    # Check if the content starts with ```json and remove it
    if content.startswith("```json"):
        content = content.lstrip("```json").strip()  # Remove markdown code block header

    # Check if the content ends with ``` and remove it
    if content.endswith("```"):
        content = content.rstrip("```").strip()  # Remove markdown code block footer

    try:
        return json.loads(content)  # Convert to dictionary
    except json.JSONDecodeError:
        print("Failed to parse JSON. Content might be incorrectly formatted.")
        return None 
    
def my_hook(sender, message, recipient, silent):
    # 确定两个方向
    if recipient.name =='misty_action_manager':
        history_list=[]

        for manager, message_list in sender._oai_messages.items():
            
            for message in message_list:
                history_list.append(message['content'])  # 提取content字段
        json_plan_task = extract_json(history_list[1])
        task_for_this_agent = json_plan_task[sender.name]
        Misty_IP = json_plan_task['Misty_IP']
        API_KEY = json_plan_task['API_KEY']
        new_message = f"Misty_IP: {Misty_IP}\n\nAPI_KEY: {API_KEY}\n\nYOURTASK:\n" + "\n".join(f"{i+1}. {task}" for i, task in enumerate(task_for_this_agent))
        return new_message
    else:
        return message

ActionAgent = SocietyOfMindAgent(
    name="ActionAgent",
    chat_manager=misty_action_manager,
    llm_config=llm_config,
    response_preparer=ActionAgent_response_preparer,
)
ActionAgent.register_hook("process_message_before_send", my_hook)
