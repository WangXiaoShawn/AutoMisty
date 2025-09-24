# filename: MistyHelloSmileWave.py

from CUBS_Misty import Robot
import time

def misty_hello_smile_wave(robot_ip):
    '''
    Execute actions for Misty to smile and wave when greeted with "Hello".
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display a joyful expression and play a welcoming sound
    misty.emotion_Joy()
    misty.sound_PhraseHello()
    
    # Step 2: Perform a waving action with arm
    for _ in range(2):  # Repeat twice for a waving gesture
        misty.move_arms(leftArmPosition=30, rightArmPosition=-29, duration=0.5)
        time.sleep(0.5)
        misty.move_arms(leftArmPosition=-29, rightArmPosition=30, duration=0.5)
        time.sleep(0.5)
    
    # Step 3: Keep a cheerful facial expression after waving
    misty.emotion_JoyGoofy()
    
    # Step 4: Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_hello_smile_wave("67.20.201.57")