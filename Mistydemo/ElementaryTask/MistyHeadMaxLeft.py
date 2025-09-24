# filename: MistyHeadMaxLeft.py

from CUBS_Misty import Robot

def turn_head_max_left(robot_ip):
    '''
    Turn Misty's head to the maximum left angle.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Turn head to the maximum left angle
    misty.move_head(yaw=81, pitch=0, roll=0, units="degrees")

    # Return Misty to her normal state
    misty.return_to_normal()

if __name__ == "__main__":
    turn_head_max_left("67.20.199.168")