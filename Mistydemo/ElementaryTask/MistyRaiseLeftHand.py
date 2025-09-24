# filename: MistyRaiseLeftHand.py

from CUBS_Misty import Robot
import time
def raise_left_hand(robot_ip):
    '''
    Move Misty's left hand to the highest point.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Move left arm to the maximum upward position
    misty.move_arms(leftArmPosition=-29, rightArmPosition=None, duration=1.0)
    
    time.sleep(1)


    # Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    raise_left_hand("67.20.199.168")