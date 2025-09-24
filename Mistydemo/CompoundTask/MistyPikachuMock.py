# filename: MistyPikachuMock.py

from CUBS_Misty import Robot
import time

def mock_pikachu_noise(robot_ip):
    '''
    Simulate Pikachu's noise by using Misty's text-to-speech for a mock "Pika Pika!" sound when her head is touched.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display curiosity expression
    misty.emotion_Admiration()
    time.sleep(1)  # Allow expression to be visible for a brief moment

    # Step 2: Use text-to-speech to mock Pikachu's sound
    misty.speak("Pika Pika!")

    # Step 3: Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    mock_pikachu_noise("67.20.201.57")