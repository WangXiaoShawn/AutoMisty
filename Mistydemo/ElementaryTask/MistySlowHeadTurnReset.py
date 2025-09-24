# filename: MistySlowHeadTurnReset.py

from CUBS_Misty import Robot
import time

def misty_slow_head_turn_with_reset(robot_ip):
    '''
    Execute a slow left and right head turn with Misty, and ensure a slow reset to neutral position.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Slowly turn head to the left
    misty.move_head(yaw=81, duration=3.0, units="degrees")
    time.sleep(3.0)  # Wait for the movement to complete
    
    # Slowly turn head to the right
    misty.move_head(yaw=-81, duration=3.0, units="degrees")
    time.sleep(3.0)  # Wait for the movement to complete
    
    # Slowly return Misty back to her neutral state
    misty.move_head(pitch=0, yaw=0, roll=0, duration=3.0, units="degrees")
    time.sleep(3.0)  # Wait for the movement to complete

    # Set Misty's LED to a neutral color and display the default content expression
    misty.return_to_normal()

if __name__ == "__main__":
    misty_slow_head_turn_with_reset("67.20.199.168")