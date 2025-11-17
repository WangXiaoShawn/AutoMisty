# filename: MistyPoemResearch.py

from CUBS_Misty import Robot
import time

def recite_poem_and_research(robot_ip: str):
    """
    Recite a themed poem "I Love Research" and enact actions that reflect the love of research.
    
    This function:
    - Instantiates Misty locally using the provided IP.
    - Recites a 12-line poem line-by-line using Misty's TTS.
    - Accompanies each line with synchronized Misty actions (LED color shifts, head/arm gestures, and sounds)
      to reflect curiosity, exploration, and the scientific method.
    - Resets Misty to normal state at the end by calling return_to_normal().
    
    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    """
    
    # Initialize Misty robot locally
    misty = Robot(robot_ip)

    # A 12-line poem about loving research
    poem_lines = [
        "I love research, a lantern in the night of doubt.",
        "I follow questions wherever bright ideas sprout.",
        "With notebooks open, I chase the glow of truth.",
        "I test again and again, patient as a sleuth.",
        "I sketch the patterns, I map the unknowns with care.",
        "I cradle evidence, I treat each result fair.",
        "I code, I measure, I repeat, I refine.",
        "I learn from failure, and from failure I shine.",
        "I share what I discover, to lift others up.",
        "I linger in quiet hours when the world is hushed and stuff.",
        "I trust the method, and let curiosity guide the way.",
        "I love research, for it makes the yearning minds sway."
    ]

    # Optional: a short intro via TTS and a mood setup
    misty.change_led(0, 128, 255)  # soft blue for focus
    misty.emotion_ContentLeft()       # look attentive
    misty.speak(text="I love research. A poem of curiosity begins.")
    time.sleep(1.0)

    # Iterate through lines, recite, and perform actions
    for idx, line in enumerate(poem_lines):
        # Speak the line
        misty.speak(text=line, flush=False)

        # Synchronize a small action with each line
        if idx % 3 == 0:
            # Gentle head tilt and scan movement to imply surveying data
            misty.move_head(pitch=0, yaw=15, roll=0, velocity=40, units="degrees")
        elif idx % 3 == 1:
            # Subtle arm gesture as if annotating in a notebook
            misty.move_arms(leftArmPosition=-29, rightArmPosition=29, duration=0.8, leftArmVelocity=40, rightArmVelocity=40)
        else:
            # Play a light positive sound to reflect excitement about findings
            misty.sound_Joy(volume=40)

        # Change LED to reflect evolving discovery
        led_r = (idx * 20) % 256
        led_g = 100
        led_b = (255 - idx * 20) % 256
        misty.change_led(red=led_r, green=led_g, blue=led_b)

        # Small pause to let actions settle and maintain rhythm
        time.sleep(0.8)

    # Closing action: a moment of celebration and then revert to normal
    misty.speak(text="The pursuit continues, one finding at a time.")
    misty.transition_led(0, 255, 0, 0, 0, 0, "Blink", 400)  # a quick green blink to celebrate
    time.sleep(1.0)
    misty.return_to_normal()  # reset to a neutral state

if __name__ == "__main__":
    # Demonstration run with Misty at the provided IP
    recite_poem_and_research("192.168.1.100")