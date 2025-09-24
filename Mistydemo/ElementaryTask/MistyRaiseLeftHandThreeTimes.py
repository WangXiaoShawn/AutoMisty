# filename: MistyRaiseLeftHandThreeTimes.py

from CUBS_Misty import Robot
import time

def raise_left_hand_three_times(robot_ip):
    '''
    Move Misty's left hand to the highest point three times.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Move left arm to the maximum upward position three times
    for _ in range(3):
        misty.move_arms(leftArmPosition=-29, rightArmPosition=None, duration=0.5)
        time.sleep(1.0)  # Hold for a moment before resetting
        
        # Reset the left arm to a neutral position
        misty.move_arms(leftArmPosition=0, rightArmPosition=None, duration=0.5)
        time.sleep(1.0)  # Pause before the next movement
    
    # Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    raise_left_hand_three_times("67.20.199.168")