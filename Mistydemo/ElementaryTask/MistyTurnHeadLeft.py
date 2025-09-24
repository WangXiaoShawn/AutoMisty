# filename: MistyTurnHeadLeft.py

from CUBS_Misty import Robot

def misty_turn_head_left(robot_ip):
    '''
    Turn Misty's head to the maximum left angle.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Step 1: Turn Misty's head to the maximum left angle
    misty.move_head(yaw=81, pitch=0, roll=0, velocity=100, units="degrees")

    # Step 2: Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_turn_head_left("67.20.201.57")