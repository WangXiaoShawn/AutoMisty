# filename: MistyComfortUser.py

from CUBS_Misty import Robot
import time

def misty_comfort_user(robot_ip):
    '''
    Comfort the user through facial expressions, sounds, and text-to-speech.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display a comforting expression and play a gentle sound
    misty.emotion_Love()
    misty.sound_Love()

    # Short pause for effect
    time.sleep(2)

    # Step 2: Speak comforting words to the user
    comforting_message = "I'm here for you. It's okay to feel sad sometimes. You're not alone."
    misty.speak(comforting_message, speechRate=1.0)

    # Step 3: Reset Misty to her normal state after comforting actions
    misty.return_to_normal()
    

if __name__ == "__main__":
    misty_comfort_user("67.20.199.168")