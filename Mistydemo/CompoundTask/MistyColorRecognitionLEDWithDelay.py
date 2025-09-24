# filename: MistyColorRecognitionLEDWithDelay.py

from CUBS_Misty import Robot
import time

def misty_recognize_and_display_color_with_delay(robot_ip, recognized_color):
    '''
    Change Misty's LED to the color recognized by the PerceptionAgent and keep it for 2 seconds.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    - recognized_color (str): The color recognized, expected to be one of "red", "green", "blue", "yellow", "purple", or "white".
    '''
    
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
    

if __name__ == "__main__":
    # Example usage: Change LED to green if the recognized color is "green"
    misty_recognize_and_display_color_with_delay("67.20.201.57", "green")