# filename: misty_sign_analysis.py

from CUBS_Misty import Robot
import cv2
import time
import base64
import threading
import queue
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage

class MistySignAnalysis(Robot):
    def __init__(self, ip: str, api_key: str) -> None:
        super().__init__(ip)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=api_key,
            temperature=0.7,
        )
        # Define the vision analysis task
        self.vision_analysis_task = '''
            Analyze the sign carefully and extract the text written on it.
            Provide an accurate reading of the sign's content.
        '''

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MistyVideo", 640, 480)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue 

            while not self.transcript_queue.empty():
                _, self.last_transcript = self.transcript_queue.get()

            if self.last_transcript:
                cv2.putText(
                    frame, self.last_transcript, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA
                )

            if "now" in self.last_transcript.lower():
                photo_filename = f"sign_photo_{int(time.time())}.jpg"
                cv2.imwrite(photo_filename, frame)
                print(f"[PHOTO] {photo_filename} saved. Analyzing sign...")

                result = self.analyze_image_with_gpt(photo_filename)

                print("[GPT RESULT]", result)
                self.speak(result)
                self.last_transcript = ""

            cv2.imshow("MistyVideo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def analyze_image_with_gpt(self, image_path: str) -> str:  # -1
        with self._analysis_lock: # Ensure thread safety with a lock
            try:
                with open(image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode('utf-8')

                messages = [
                    AIMessage(content="You are an expert in analyzing image content."),
                    HumanMessage(content=[
                        {"type": "text", "text": self.vision_analysis_task},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ])
                ]

                response = self.llm.invoke(messages)
                return response.content.strip()

            except Exception as e:
                print(f"[GPT] Error calling invoke: {e}")
                return "I couldn't analyze the sign."

if __name__ == "__main__":
    misty_ip = "67.20.199.168"
    api_key = "YOUR_OPENAI_API_KEY_HERE"  # 请设置您的OpenAI API Key

    misty = MistySignAnalysis(misty_ip, api_key)
    misty.start_perception_loop()