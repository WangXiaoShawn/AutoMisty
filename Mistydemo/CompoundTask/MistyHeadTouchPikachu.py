# filename: MistyHeadTouchPikachu.py

from CUBS_Misty import Robot
import time
from typing import Any, Dict

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
        self.ip = ip

    def trigger_pikachu_noise(self):
        '''
        Simulate Pikachu's noise by using Misty's text-to-speech for a mock "Pika Pika!" sound.
        '''
        # Display curiosity expression
        self.emotion_Admiration()
        time.sleep(1)  # Allow expression to be visible for a brief moment

        # Use text-to-speech to mock Pikachu's sound
        self.speak("Pika Pika!")

        # Reset Misty back to her neutral state
        self.return_to_normal()

    def register_head_touch(self) -> None:
        '''
        Registers the head touch event to trigger Pikachu's noise.
        '''
        # Define the callback function as a closure to access self
        def head_touch_callback(data):
            '''
            Callback function triggered by a head touch event.
            '''
            print("[INFO] Head touch detected. Playing Pikachu's noise.")
            self.trigger_pikachu_noise()

        self.register_event(
            event_type="TouchSensor",
            event_name="HeadTouchPikachu",
            condition=[
                event_filter("sensorPosition", "=", "HeadFront")
            ],
            debounce=500,
            keep_alive=True,
            callback_function=head_touch_callback
        )

    def register_and_run_event(self) -> None:
        '''
        Registers the head touch event and starts the robot's event loop.
        '''
        self.register_head_touch()
        self.start()

if __name__ == "__main__":
    # Misty robot's IP address
    misty_ip = "67.20.199.168"
    
    # Initialize the custom robot instance
    misty = CustomRobot(misty_ip)
    
    # Register and run the head touch event
    misty.register_and_run_event()