# filename: MistyRaiseRightHand.py

from CUBS_Misty import Robot
import time

def raise_right_hand(robot_ip):
    '''
    Raises Misty's right hand to the highest point.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Move the right arm to the highest point (maximum upward position: -29 degrees)
    misty.move_arms(leftArmPosition=None, rightArmPosition=-29, duration=1.0)

    # Wait for the movement to complete
    time.sleep(1)

    # Step 2: Return Misty to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    raise_right_hand("67.20.199.168")