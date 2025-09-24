# filename: MistyLoveResearchPoem.py

from CUBS_Misty import Robot
import time

def misty_love_research_poem(robot_ip):
    '''
    Create and recite a poem themed "I love Research", while performing synchronized actions.

    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Set the color and initial expression to indicate curiosity and excitement
    misty.emotion_Amazement()
    misty.change_led(0, 255, 255)  # Set LED light to cyan, symbolizing inspiration
    misty.sound_Amazement2()

    # Start the poem recitation with actions & speech
    poem_lines = [
        "In the vast ocean of knowledge, I set sail,",
        "With every discovery, my spirit prevails.",
        "I crave the truth, unveiling each layer,",
        "In research's embrace, I find my care.",
        "Each challenge is a puzzle, each problem a key,",
        "I dive into research, where I'd rather be."
    ]

    # Actions with movements for each line of the poem
    for i, line in enumerate(poem_lines):
        # Motion-gaze upwards for inspiration
        if i == 0:
            misty.move_head(pitch=-20, yaw=0, roll=0, duration=2.0)
            misty.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=1.0)
        # Enthusiastic arm movements for the excitement of discovery
        elif i == 1:
            misty.move_arms(leftArmPosition=90, rightArmPosition=60, duration=1.0)
        # Align emotionally with the essence of seeking truth
        elif i == 2:
            misty.emotion_Joy()
            misty.change_led(0, 255, 0)  # Change LED light to green for growth 
        # Signal internal bliss with research connection 
        elif i == 3:
            misty.emotion_ContentRight()
            misty.change_led(255, 165, 0)  # Change to orange to reflect contentment
        # Serious head nodding as handling challenges 
        elif i == 4:
            misty.move_head(pitch=0, yaw=81, roll=0, duration=0.5)
        # Final arms relaxed motion to savor contentment
        elif i == 5:
            misty.move_arms(leftArmPosition=0, rightArmPosition=0, duration=2.0)
            misty.return_to_normal()
        
        # Recite the line of the poem in default serious tone
        misty.speak(line)  # Removed pitch parameter
        time.sleep(3)  # Wait to ensure articulation completes

    # Reset Misty to a neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_love_research_poem("67.20.199.168")