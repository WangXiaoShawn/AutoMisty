# filename: MistyTouchResponse.py

from CUBS_Misty import Robot
import time

def display_adorable_excited_expression(robot_ip):
    """
    Display an adorable, excited expression on Misty when her head is touched.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    """
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Step 1: Change Misty's expression to joyful and excited
    misty.emotion_JoyGoofy2()
    
    # Step 2: Change LED color to a happy green
    misty.change_led(0, 255, 0)
    
    # Step 3: Play a sound expressing joy
    misty.sound_Joy()
    time.sleep(1)

    # Step 4: Add head and arm movement for excitement
    misty.move_head(pitch=-10, yaw=0, roll=0, duration=1.0)
    misty.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=1.0)
    time.sleep(1)

    # Reset to normal after completion
    misty.return_to_normal()


def display_extremely_angry_expression(robot_ip):
    """
    Show an extremely angry expression on Misty when the back of her head is touched.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    """
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Step 1: Display an extremely angry expression
    misty.emotion_Rage4()
    
    # Step 2: Change LED to a furious red
    misty.change_led(255, 0, 0)
    
    # Step 3: Play a sound expressing intense anger
    misty.sound_Rage()
    time.sleep(1)

    # Step 4: Add head shaking and arm movements for anger
    for _ in range(3):
        misty.move_head(yaw=30, duration=0.2)
        misty.move_arms(leftArmPosition=90, rightArmPosition=-29, duration=0.2)
        time.sleep(0.2)
        misty.move_head(yaw=-30, duration=0.2)
        misty.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=0.2)
        time.sleep(0.2)

    # Reset to normal after showing anger
    misty.return_to_normal()


if __name__ == "__main__":
    robot_ip = "67.20.201.57"
    display_adorable_excited_expression(robot_ip)
    time.sleep(2)  # Wait for a moment before showing the next emotion
    display_extremely_angry_expression(robot_ip)