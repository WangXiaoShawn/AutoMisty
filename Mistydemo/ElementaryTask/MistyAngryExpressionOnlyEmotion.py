# filename: MistyAngryExpressionOnlyEmotion.py

from CUBS_Misty import Robot
import time

def misty_angry_expression_only_emotion(robot_ip):
    '''
    Display only an angry expression on Misty the robot without sound or LED change.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Set an angry facial expression
    misty.emotion_Anger()
    
    # Maintain the expression for 2 seconds
    time.sleep(2)

    # Return Misty to normal state after the expression
    misty.return_to_normal()

if __name__ == "__main__":
    misty_angry_expression_only_emotion("67.20.199.168")