# filename: MistyStoryTeller.py

from CUBS_Misty import Robot
import time

def misty_narrate_story(robot_ip, api_key, story_text):
    '''
    Make Misty narrate a story using voice and expression.

    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    - api_key (str): The API key for accessing Narration API (if used).
    - story_text (str): The text of the story to be narrated.
    '''

    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Step 1: Set an engaging expression for storytelling
    misty.emotion_JoyGoofy()
    misty.change_led(0, 255, 0)  # Set LED color to green for storytelling mode

    # Step 2: Narrate the story with voice using TTS
    misty.speak(text=story_text, speechRate=1.0, pitch=1.0)

    # Step 3: Accompany the narration with expressive arm movements
    misty.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=1.0)
    misty.move_head(pitch=10, yaw=20, roll=0, duration=1.0)
    time.sleep(1.0)
    misty.move_arms(leftArmPosition=90, rightArmPosition=90, duration=1.0)
    misty.move_head(pitch=-10, yaw=-20, roll=0, duration=1.0)
    time.sleep(1.0)

    # Step 4: Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    story_text = "Once upon a time in a magical forest, there was a little robot named Misty. " \
                 "Misty loved to explore and help her friends. One day, she found a hidden treasure, " \
                 "but only kindness could unlock it. Together with her friends, they shared happy moments " \
                 "and discovered the treasure's secret: the real treasure was friendship!"
    api_key = "YOUR_OPENAI_API_KEY_HERE"  # 请设置您的OpenAI API Key
    misty_narrate_story("67.20.201.57", api_key, story_text)