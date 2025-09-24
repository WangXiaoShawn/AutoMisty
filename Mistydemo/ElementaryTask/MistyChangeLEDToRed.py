# filename: MistyChangeLEDToRed.py

import time
from CUBS_Misty import Robot

def change_led_to_red(robot_ip):
    '''
    Change Misty's LED to red for 3 seconds.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Change LED to red
    misty.change_led(red=255, green=0, blue=0)
    
    # Keep the LED red for 3 seconds
    time.sleep(3)

    # Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    change_led_to_red("67.20.199.168")