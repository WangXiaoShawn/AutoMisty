# filename: MistyTurnHeadLeftThriceWithLongerReset.py

from CUBS_Misty import Robot
import time

def misty_turn_head_left_thrice_with_longer_reset(robot_ip):
    '''
    Turn Misty's head to the maximum left angle three times consecutively with longer reset in between.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Perform the action three times consecutively with reset
    for _ in range(3):
        # Step 1: Turn Misty's head to the maximum left angle
        misty.move_head(yaw=81, pitch=0, roll=0, velocity=100, units="degrees")

        # Step 2: Reset Misty back to her neutral state
        misty.return_to_normal()
        time.sleep(4)  # Wait longer before the next iteration

if __name__ == "__main__":
    misty_turn_head_left_thrice_with_longer_reset("67.20.199.168")