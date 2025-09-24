# filename: MistyThreeLittlePigsStoryWithMovement.py

from CUBS_Misty import Robot
import time

def tell_three_little_pigs_story_with_movement(robot_ip):
    '''
    Narrate the story of The Three Little Pigs with body movements, using Misty's 
    text-to-speech ability with a cute voice, suitable expressions, and sounds 
    to create an engaging storytelling experience.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Introduction
    misty.emotion_Joy()
    misty.speak("Once upon a time, there were three little pigs. One day, they decided to build their own houses.", pitch=2.0, speechRate=1.3)

    # House of straw
    misty.emotion_Amazement()
    misty.move_arms(leftArmPosition=-29, rightArmPosition=90, duration=2.0)
    misty.speak("The first little pig built his house of straw. It was quick and easy!", pitch=1.8, speechRate=1.2)
    time.sleep(3)

    # Wolf approaches the straw house
    misty.emotion_Terror()
    misty.move_head(yaw=45, duration=1.0)
    time.sleep(1)

    misty.speak("But soon, a big bad wolf came by. He huffed and puffed and blew the house down!", pitch=2.2, speechRate=1.0)
    misty.sound_Disapproval()
    time.sleep(3)

    # House of sticks
    misty.transition_led(255, 165, 0, 255, 69, 0, "Blink", 500)
    misty.emotion_Joy()
    misty.move_arms(leftArmPosition=-29, rightArmPosition=45, duration=2.0)
    misty.speak("The second little pig built his house with sticks. A bit sturdier, he thought.", pitch=1.7, speechRate=1.2)
    time.sleep(3)

    # Wolf approaches the stick house
    misty.emotion_Terror()
    misty.move_head(yaw=-45, duration=1.0)
    misty.speak("But then, the wolf huffed and puffed and blew the stick house down too!", pitch=2.2, speechRate=1.0)
    misty.sound_Disapproval()
    time.sleep(3)

    # House of bricks
    misty.change_led(0, 0, 255)
    misty.emotion_Disoriented()
    misty.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=2.0)
    misty.speak("The third little pig built his house of bricks. It was hard work, but very strong.", pitch=1.6, speechRate=1.2)
    time.sleep(3)

    # Wolf approaches the brick house
    misty.emotion_Aggressiveness()
    misty.move_head(yaw=0, pitch=10, duration=1.0)
    misty.speak("The wolf came to the house of bricks and tried to blow it down.", pitch=2.1, speechRate=1.0)

    misty.sound_PhraseUhOh()
    misty.emotion_Terror2()
    time.sleep(3)

    # Brick house stands strong
    misty.emotion_Joy2()
    misty.move_head(yaw=0, pitch=-10, duration=1.0)
    misty.speak("But no matter how hard he tried, the brick house stood strong!", pitch=1.5, speechRate=1.3)

    misty.sound_Joy()
    time.sleep(3)

    # Conclusion
    misty.return_to_normal()
    misty.speak("The three little pigs learned the value of hard work and security. They lived happily ever after. The end.", pitch=2.0, speechRate=1.3)

if __name__ == "__main__":
    tell_three_little_pigs_story_with_movement("67.20.199.168")