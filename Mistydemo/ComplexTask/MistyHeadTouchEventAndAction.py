# filename: MistyHeadTouchEventAndAction.py

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

def misty_countdown_and_cute_action(robot):
    '''
    Execute a countdown followed by a cute action by Misty upon detecting head touch.
    
    Parameters:
    - robot (Robot): An instance of the Misty robot.
    '''

    # Step 1: Count down from 3 to 1
    robot.speak("Three")
    time.sleep(1)
    robot.speak("Two")
    time.sleep(1)
    robot.speak("One")
    time.sleep(1)

    # Step 2: Capture a photo (Simulate as we don't have vision API functions here)
    print("[INFO] Taking a photo...")

    # Step 3: Perform a cute action: Waving arms with a joyful expression and bright LED color
    robot.emotion_Joy()
    robot.change_led(255, 182, 193)  # Set LED color to pink for cuteness
    robot.sound_Joy()
    
    # Cute action: waving arms
    for _ in range(2):
        robot.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=0.5)
        time.sleep(0.5)
        robot.move_arms(leftArmPosition=90, rightArmPosition=90, duration=0.5)
        time.sleep(0.5)

    # Restore Misty to neutral state
    robot.return_to_normal()

class CustomRobot(Robot):
    def __init__(self, ip):
        '''
        Initializes the CustomRobot with the given IP address.
        
        :param ip: The IP address of the Misty robot.
        '''
        super().__init__(ip)

    def register_head_touch_event(self) -> None:
        '''
        Registers the head touch event to trigger the countdown and cute action.
        '''
        # Define the callback function as a closure to access self
        def head_touch_callback(data):
            '''
            Callback function triggered by a head touch event.
            '''
            print("[INFO] Head touch detected. Initiating countdown and cute action.")
            misty_countdown_and_cute_action(self)

        self.register_event(
            event_type="TouchSensor",
            event_name="HeadTouchEvent",
            condition=[
                event_filter("sensorPosition", "=", "HeadFront")
            ],
            debounce=500,
            keep_alive=True,
            callback_function=head_touch_callback
        )

    def register_and_run_events(self) -> None:
        '''
        Registers the head touch event and starts the robot's event loop.
        '''
        self.register_head_touch_event()
        self.start()

if __name__ == "__main__":
    # Misty robot's IP address
    misty_ip = "67.20.201.57"
    
    # Initialize the custom robot instance
    misty = CustomRobot(misty_ip)
    
    # Register and run the events
    misty.register_and_run_events()