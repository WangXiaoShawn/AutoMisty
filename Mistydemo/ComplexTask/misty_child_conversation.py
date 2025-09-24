# filename: misty_child_conversation.py

from CUBS_Misty import Robot
import cv2
import queue
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

class MistyChildConversation(Robot):
    def __init__(self, ip: str, api_key: str) -> None:
        super().__init__(ip)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,
            temperature=0.7,
        )
        # Set a unique characteristic for Misty as a child
        self.chat_ai_task = '''
            You are a playful and imaginative child named Misty! Respond with curiosity and enthusiasm, 
            using a child's language and perspective.
        '''

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)

                while not self.transcript_queue.empty():
                    _, self.last_transcript = self.transcript_queue.get()

                if self.last_transcript:
                    response = self.chat_with_gpt(self.last_transcript)
                    self.speak(response)
                    self.last_transcript = ""

                cv2.imshow("MistyVideo", frame)

            except queue.Empty:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def chat_with_gpt(self, user_input: str) -> str: #-1
        '''
        Sends user input to GPT and returns the response in child's identity.
        '''
        with self._analysis_lock:  
            try:
                self.conversation_history.append(HumanMessage(content=user_input))
                messages = [AIMessage(content=self.chat_ai_task)] + self.conversation_history

                response = self.llm.invoke(messages)
                self.conversation_history.append(AIMessage(content=response.content.strip()))

                return response.content.strip()

            except Exception as e:
                print(f"[GPT] Error calling invoke: {e}")  
                return "Wow, I don't know what happened!" 

if __name__ == "__main__":
    # Instantiate Misty with the child's conversational ability
    misty = MistyChildConversation("67.20.199.168", 
                                   api_key=api_key)
    # Start the perception loop
    misty.start_perception_loop()