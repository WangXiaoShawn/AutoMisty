# filename: MistyWaterReminder.py

from CUBS_Misty import Robot
import time

def misty_water_reminder(robot_ip):
    '''
    Continuously remind the user to drink water every 10 seconds using Misty's actions and voice.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    try:
        while True:
            # Step 1: Speak a reminder to drink water
            misty.speak("Hey, it's time to drink some water!")
            
            # Step 2: Enhance reminder with body movement
            # Light up LED to a refreshing color (e.g., light blue)
            misty.change_led(0, 255, 255)
            
            # Nod head to catch attention
            misty.move_head(pitch=15, duration=0.5)
            time.sleep(0.5)
            misty.move_head(pitch=-15, duration=0.5)
            time.sleep(0.5)
            
            # Arm movement as an additional gesture
            misty.move_arms(leftArmPosition=45, rightArmPosition=45, duration=1.0)
            time.sleep(1.0)
            misty.move_arms(leftArmPosition=0, rightArmPosition=0, duration=1.0)
            time.sleep(1.0)
            
            # Wait for 10 seconds before the next reminder
            time.sleep(10)
    
    except KeyboardInterrupt:
        # Step 3: Reset Misty back to her neutral state when manually interrupted
        misty.return_to_normal()
        print("Reminder process terminated.")

if __name__ == "__main__":
    misty_water_reminder("67.20.199.168")