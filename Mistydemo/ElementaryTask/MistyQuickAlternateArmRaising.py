# filename: MistyQuickAlternateArmRaising.py

from CUBS_Misty import Robot
import time

def misty_quick_alternate_arm_raising(robot_ip):
    '''
    Quickly alternate raising the left and right hands three times.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Begin sequence of rapid arm movements
    for _ in range(3):
        # Step 1: Raise left hand quickly
        misty.move_arms(leftArmPosition=-29, rightArmPosition=0, leftArmVelocity=100, rightArmVelocity=100)
        time.sleep(0.5)  # Wait for the movement to complete
        
        # Step 2: Lower left hand quickly
        misty.move_arms(leftArmPosition=90, rightArmPosition=0, leftArmVelocity=100, rightArmVelocity=100)
        time.sleep(0.5)
        
        # Step 3: Raise right hand quickly
        misty.move_arms(leftArmPosition=0, rightArmPosition=-29, leftArmVelocity=100, rightArmVelocity=100)
        time.sleep(0.5)
        
        # Step 4: Lower right hand quickly
        misty.move_arms(leftArmPosition=0, rightArmPosition=90, leftArmVelocity=100, rightArmVelocity=100)
        time.sleep(0.5)
    
    # Reset Misty back to her normal state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_quick_alternate_arm_raising("67.20.199.168")