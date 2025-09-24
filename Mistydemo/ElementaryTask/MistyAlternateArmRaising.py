# filename: MistyAlternateArmRaising.py

from CUBS_Misty import Robot
import time

def misty_alternate_arm_raising(robot_ip):
    '''
    Slowly alternate raising the left and right hands three times.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Begin sequence of arm movements
    for _ in range(3):
        # Step 1: Raise left hand slowly
        misty.move_arms(leftArmPosition=-29, rightArmPosition=0, leftArmVelocity=10, rightArmVelocity=10)
        time.sleep(2)  # Wait for the movement to complete
        
        # Step 2: Lower left hand slowly
        misty.move_arms(leftArmPosition=90, rightArmPosition=0, leftArmVelocity=10, rightArmVelocity=10)
        time.sleep(2)
        
        # Step 3: Raise right hand slowly
        misty.move_arms(leftArmPosition=0, rightArmPosition=-29, leftArmVelocity=10, rightArmVelocity=10)
        time.sleep(2)
        
        # Step 4: Lower right hand slowly
        misty.move_arms(leftArmPosition=0, rightArmPosition=90, leftArmVelocity=10, rightArmVelocity=10)
        time.sleep(2)
    
    # Reset Misty back to her normal state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_alternate_arm_raising("67.20.199.168")