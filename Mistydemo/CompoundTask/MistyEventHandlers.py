# filename: MistyEventHandlers.py

from CUBS_Misty import Robot
from typing import Any, Dict
import time

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

    def display_adorable_excited_expression(self):
        """
        Display an adorable, excited expression on Misty when her head is touched.
        """
        # Step 1: Change Misty's expression to joyful and excited
        self.emotion_JoyGoofy2()
        
        # Step 2: Change LED color to a happy green
        self.change_led(0, 255, 0)
        
        # Step 3: Play a sound expressing joy
        self.sound_Joy()
        time.sleep(1)

        # Step 4: Add head and arm movement for excitement
        self.move_head(pitch=-10, yaw=0, roll=0, duration=1.0)
        self.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=1.0)
        time.sleep(1)

        # Reset to normal after completion
        self.return_to_normal()

    def display_extremely_angry_expression(self):
        """
        Show an extremely angry expression on Misty when the back of her head is touched.
        """
        # Step 1: Display an extremely angry expression
        self.emotion_Rage4()
        
        # Step 2: Change LED to a furious red
        self.change_led(255, 0, 0)
        
        # Step 3: Play a sound expressing intense anger
        self.sound_Rage()
        time.sleep(1)

        # Step 4: Add head shaking and arm movements for anger
        for _ in range(3):
            self.move_head(yaw=30, duration=0.2)
            self.move_arms(leftArmPosition=90, rightArmPosition=-29, duration=0.2)
            time.sleep(0.2)
            self.move_head(yaw=-30, duration=0.2)
            self.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=0.2)
            time.sleep(0.2)

        # Reset to normal after showing anger
        self.return_to_normal()

    def register_head_front_touch(self) -> None:
        '''
        Registers the front head touch event to display an excited expression.
        '''
        # Define the callback function as a closure to access self
        def head_front_touch_callback(data):
            '''
            Callback function triggered by a front head touch event.
            '''
            print("[INFO] Front head touch detected. Displaying excited expression.")
            self.display_adorable_excited_expression()

        self.register_event(
            event_type="TouchSensor",
            event_name="HeadFrontTouch",
            condition=[
                event_filter("sensorPosition", "=", "HeadFront")
            ],
            debounce=500,
            keep_alive=True,
            callback_function=head_front_touch_callback
        )

    def register_head_back_touch(self) -> None:
        '''
        Registers the back head touch event to show an angry expression.
        '''
        # Define the callback function as a closure to access self
        def head_back_touch_callback(data):
            '''
            Callback function triggered by a back head touch event.
            '''
            print("[INFO] Back head touch detected. Displaying angry expression.")
            self.display_extremely_angry_expression()

        self.register_event(
            event_type="TouchSensor",
            event_name="HeadBackTouch",
            condition=[
                event_filter("sensorPosition", "=", "HeadBack")
            ],
            debounce=500,
            keep_alive=True,
            callback_function=head_back_touch_callback
        )

    def register_and_run_events(self) -> None:
        '''
        Registers head touch events and starts the robot's event loop.
        '''
        self.register_head_front_touch()
        self.register_head_back_touch()
        self.start()

if __name__ == "__main__":
    # Misty robot's IP address
    misty_ip = "67.20.199.168"

    # Initialize the custom robot instance
    misty = CustomRobot(misty_ip)

    # Register and run the events
    misty.register_and_run_events()