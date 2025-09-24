# filename: misty_translation_perception.py

from CUBS_Misty import Robot
import queue
import cv2
import threading
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

class MistyTranslation(Robot):
    def __init__(self, ip: str, api_key: str) -> None:
        super().__init__(ip)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,
            temperature=0,
        )
        # Define the task for GPT to translate English to Chinese
        self.translation_task = '''
            You are a translation assistant. Translate the following English text into Chinese.
        '''

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)

                while not self.transcript_queue.empty():
                    _, self.last_transcript = self.transcript_queue.get()

                if self.last_transcript:
                    translation = self.translate_to_chinese(self.last_transcript)
                    print(f"Translation: {translation}")
                    self.speak(translation)
                    self.last_transcript = ""

                cv2.imshow("MistyVideo", frame)

            except queue.Empty:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def translate_to_chinese(self, english_text: str) -> str:
        '''
        Translates English text into Chinese using GPT.
        '''
        with self._analysis_lock:
            try:
                # Create messages for GPT to translate the text
                messages = [
                    AIMessage(content=self.translation_task),
                    HumanMessage(content=english_text)
                ]

                # Get the translation response from GPT
                response = self.llm.invoke(messages)
                return response.content.strip()

            except Exception as e:
                print(f"[GPT] Error in translation: {e}")
                return "Translation failed."

if __name__ == "__main__":

    misty = MistyTranslation("67.20.197.87",
                             api_key=api_key)
    misty.start_perception_loop()