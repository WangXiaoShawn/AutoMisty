# filename: MistyRockStyleDance.py

from CUBS_Misty import Robot
import time

def misty_rock_style_dance(robot_ip):
    '''
    Perform a wild, energetic rock-style dance with Misty the robot.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Set an excited expression and vibrant LED color
    misty.emotion_JoyGoofy2()
    misty.change_led(255, 20, 147)  # Set LED to a bright pink for excitement
    misty.sound_Joy2()

    # Step 2: Start with an energetic headbang motion
    for _ in range(5):
        misty.move_head(pitch=20, duration=0.2)  # Head down
        time.sleep(0.2)
        misty.move_head(pitch=-20, duration=0.2) # Head up
        time.sleep(0.2)

    # Step 3: Use aggressive arm movements to show high energy
    for _ in range(3):
        misty.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=0.3)
        time.sleep(0.3)
        misty.move_arms(leftArmPosition=90, rightArmPosition=-29, duration=0.3)
        time.sleep(0.3)
    
    # Step 4: Simulate a guitar strumming action
    misty.move_arms(leftArmPosition=90, rightArmPosition=0, duration=0.5)  
    misty.move_head(yaw=45, duration=0.5)  
    misty.sound_Ecstacy()
    time.sleep(0.5)
    misty.move_arms(leftArmPosition=0, rightArmPosition=90, duration=0.5)
    misty.move_head(yaw=-45, duration=0.5)
    time.sleep(0.5)

    # Step 5: Add a spinning LED effect and include a sound
    misty.transition_led(255, 0, 0, 0, 0, 255, "Breathe", 400)
    misty.sound_Ecstacy2()
    time.sleep(1.6)

    # Step 6: Include rapid movements and finish with a pose
    for _ in range(3):
        misty.move_head(yaw=45, duration=0.1)
        time.sleep(0.1)
        misty.move_head(yaw=-45, duration=0.1)
        time.sleep(0.1)
    
    # Final pose
    misty.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=0.5)
    misty.move_head(pitch=-40, yaw=0, roll=0, duration=0.5)
    time.sleep(2)

    # Reset Misty back to her neutral state after the dance
    misty.return_to_normal()

if __name__ == "__main__":
    misty_rock_style_dance("67.20.199.168")