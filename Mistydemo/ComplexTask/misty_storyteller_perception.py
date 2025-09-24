# filename: misty_storyteller_perception.py

from CUBS_Misty import Robot
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import cv2
import queue
import time

def misty_narrate_story(robot_ip, api_key, story_text):
    '''
    Make Misty narrate a story using voice and expression.

    Parameters:
    - robot_ip (str): The IP address of the Misty robot.
    - api_key (str): The API key for accessing any necessary services.
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

class MistyStoryTellerPerception(Robot):
    def __init__(self, ip: str, api_key: str) -> None:
        super().__init__(ip)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,
            temperature=0.7,  # Use higher temperature for more creativity
        )
        self.story_generation_task = '''
            Generate a short and engaging story based on the given keyword.
            Ensure the story is imaginative and suitable for all ages.
        '''

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Display the video frame
            cv2.imshow("MistyVideo", frame)

            # Fetch and process the latest transcription
            while not self.transcript_queue.empty():
                _, self.last_transcript = self.transcript_queue.get()

            if self.last_transcript:
                print(f"Keyword detected: {self.last_transcript}")
                
                # Use the transcription (keyword) to generate a story
                story_text = self.generate_story_with_keyword(self.last_transcript)
                
                # Reset the transcript
                self.last_transcript = ""

                # Use ActionAgent's function for narration
                misty_narrate_story(self.ip, api_key, story_text)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def generate_story_with_keyword(self, keyword: str) -> str:
        with self._analysis_lock:  # Use lock for thread safety
            try:
                messages = [
                    AIMessage(content=self.story_generation_task),  # AI task prompt
                    HumanMessage(content=keyword)  # Keyword provided by user
                ]

                # Call LangChain's ChatOpenAI model to generate a story
                response = self.llm.invoke(messages)

                return response.content.strip()  # Return the generated story

            except Exception as e:
                print(f"[GPT] Error calling invoke: {e}")
                return "I'm sorry, I couldn't create a story right now."

if __name__ == "__main__":
    ip_address = "67.20.199.168"
    api_key = "YOUR_OPENAI_API_KEY_HERE"  # 请设置您的OpenAI API Key

    misty = MistyStoryTellerPerception(ip=ip_address, api_key=api_key)
    misty.start_perception_loop()