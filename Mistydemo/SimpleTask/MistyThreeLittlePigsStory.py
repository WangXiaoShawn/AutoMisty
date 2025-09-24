# filename: MistyThreeLittlePigsStory.py

from CUBS_Misty import Robot
import time

def tell_three_little_pigs_story(robot_ip):
    '''
    Narrate the story of The Three Little Pigs, using Misty's text-to-speech ability 
    with suitable expressions and sounds to create an engaging storytelling experience.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)

    # Introduction
    misty.emotion_Joy()
    misty.speak("Once upon a time, there were three little pigs. One day, they decided to build their own houses.")

    # House of straw
    misty.emotion_Amazement()
    misty.speak("The first little pig built his house of straw. It was quick and easy!")
    time.sleep(3)

    # Wolf approaches the straw house
    misty.emotion_Terror()
    misty.speak("But soon, a big bad wolf came by. He huffed and puffed and blew the house down!")
    misty.sound_Disapproval()
    time.sleep(3)

    # House of sticks
    misty.transition_led(255, 165, 0, 255, 69, 0, "Blink", 500)
    misty.emotion_Joy()
    misty.speak("The second little pig built his house with sticks. A bit sturdier, he thought.")
    time.sleep(3)

    # Wolf approaches the stick house
    misty.emotion_Terror()
    misty.speak("But then, the wolf huffed and puffed and blew the stick house down too!")
    misty.sound_Disapproval()
    time.sleep(3)

    # House of bricks
    misty.change_led(0, 0, 255)
    misty.emotion_Disoriented()
    misty.speak("The third little pig built his house of bricks. It was hard work, but very strong.")
    time.sleep(3)

    # Wolf approaches the brick house
    misty.emotion_Aggressiveness()
    misty.speak("The wolf came to the house of bricks and tried to blow it down.")
    misty.sound_PhraseUhOh()
    misty.emotion_Terror2()
    time.sleep(3)

    # Brick house stands strong
    misty.emotion_Joy2()
    misty.speak("But no matter how hard he tried, the brick house stood strong!")
    misty.sound_Joy()
    time.sleep(3)

    # Conclusion
    misty.return_to_normal()
    misty.speak("The three little pigs learned the value of hard work and security. They lived happily ever after. The end.")

if __name__ == "__main__":
    tell_three_little_pigs_story("67.20.201.57")