# filename: MistySlowPacedIntro.py

from CUBS_Misty import Robot

def misty_slow_intro(robot_ip):
    '''
    Make Misty say "Hello, I am Misty" at a slow pace using the onboard text-to-speech engine.
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    '''
    
    # Initialize Misty robot
    misty = Robot(robot_ip)
    
    # Use Misty's TTS to say "Hello, I am Misty" with a slower speech rate
    misty.speak("Hello, I am Misty", speechRate=0.5)
    
    # Reset Misty back to her neutral state
    misty.return_to_normal()

if __name__ == "__main__":
    misty_slow_intro("67.20.199.168")