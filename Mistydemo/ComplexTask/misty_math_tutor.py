# filename: misty_math_tutor.py

from CUBS_Misty import Robot
import cv2
import queue
import threading
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

class MistyMathTutor(Robot):
    def __init__(self, ip: str, api_key: str) -> None:
        super().__init__(ip)
        # Initialize ChatOpenAI model with given API key for math problem-solving
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,
            temperature=0,
        )
        # Define task for conversational AI to focus on math problem solving
        self.chat_ai_task = '''
            You are a math tutor! Solve math questions precisely and clearly explain each step.
        '''

    def process_multimodal_task(self) -> None:
        '''
        Process video frames and interpret spoken math questions for solving.
        '''
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                # Capture the latest video frame
                frame = self.frame_queue.get(timeout=0.1)

                # Check for new voice transcriptions in the queue
                while not self.transcript_queue.empty():
                    _, self.last_transcript = self.transcript_queue.get()

                if self.last_transcript:
                    # Send the spoken math question to GPT for solving
                    response = self.chat_with_gpt(self.last_transcript)
                    # Speak out the solution in an easy-to-understand manner
                    self.speak(response)
                    # Reset last transcript to avoid repeat processing
                    self.last_transcript = ""

                # Display the video feed
                cv2.imshow("MistyVideo", frame)

            except queue.Empty:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def chat_with_gpt(self, user_input: str) -> str:
        '''
        Interact with GPT to solve math problems provided by the user's voice input.

        Args:
            user_input (str): The math question spoken by the user.

        Returns:
            str: The math solution provided by the AI.
        '''
        with self._analysis_lock:  
            try:
                # Clear the conversation history to focus solely on the math problem
                messages = [AIMessage(content=self.chat_ai_task), HumanMessage(content=user_input)]
                
                # Request a response from GPT based on the math question
                response = self.llm.invoke(messages)
                return response.content.strip()

            except Exception as e:
                print(f"[GPT] Error while solving math problem: {e}")
                return "I'm sorry, I couldn't process your request."

if __name__ == "__main__":
    # Initialize Misty as a math tutor with IP and API key
    misty = MistyMathTutor("67.20.199.168", 
                           api_key=api_key)
    # Start the main perception loop
    misty.start_perception_loop()