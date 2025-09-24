# filename: MistyComfortAndCheerUp.py

from CUBS_Misty import Robot
import time

def misty_comfort_and_cheer_up(robot_ip):
    '''
    Comfort and cheer up the user with Misty the robot by displaying expressions, playing sounds, 
    speaking comforting words, performing a cute action, and telling a joke.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Step 1: Display a comforting expression and play a gentle sound
    misty.emotion_Love()
    misty.sound_Love()

    # Short pause for effect
    time.sleep(2)

    # Step 2: Speak comforting words to the user
    comforting_message = "I'm here for you. It's okay to feel sad sometimes. You're not alone."
    misty.speak(comforting_message, speechRate=1.0)

    # Short pause for effect
    time.sleep(2)
    
    # Step 3: Perform a cute action
    # Waving one arm and displaying a joyful sound and color
    misty.emotion_JoyGoofy()
    misty.move_arms(leftArmPosition=-29, rightArmPosition=30, duration=1.0)
    misty.change_led(0, 255, 255)  # Cyan color for playfulness
    misty.sound_Joy()

    # Short pause for effect
    time.sleep(2)
    
    # Step 4: Tell a joke to cheer up the user
    joke = "Why don't scientists trust atoms? Because they make up everything!"
    misty.speak(joke, speechRate=1.0)
    
    # Short pause after joke
    time.sleep(5)
    
    # Step 5: Reset Misty to her normal state after comforting and cheering actions
    misty.return_to_normal()

if __name__ == "__main__":
    misty_comfort_and_cheer_up("67.20.199.168")