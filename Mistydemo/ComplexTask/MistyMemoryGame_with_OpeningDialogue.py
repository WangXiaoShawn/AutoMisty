# filename: MistyMemoryGame_with_OpeningDialogue.py

from CUBS_Misty import Robot
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import cv2
import queue
import time
import random
import threading
from difflib import SequenceMatcher # Use difflib for fuzzy matching

class MistyMemoryGame(Robot):
    def __init__(self, ip: str) -> None:
        super().__init__(ip)
        self.actions = ["up", "down", "left", "right"]
        self.selected_action = None

    def deliver_opening_dialog(self) -> None:
        '''Introduce the memory game to the player.'''
        self.speak("Welcome to the Misty Memory Game! I'll perform a random action: up, down, left, or right.")
        self.speak("Try to remember the action because I'll ask you to guess it soon after. Let's start!")
        time.sleep(8)  # Wait for 8 seconds before starting the first action

    def select_and_perform_action(self) -> None:
        '''
        Select and perform a random action, announcing it to the player.
        '''
        self.selected_action = random.choice(self.actions)

        # Original sequence of actions
        if self.selected_action == "up":
            self.speak("Up!")
            self.emotion_Joy()
            self.change_led(0, 255, 0)  # LED green
            self.move_arms(leftArmPosition=-29, rightArmPosition=-29, duration=1.0)
        elif self.selected_action == "down":
            self.speak("Down!")
            self.emotion_Sadness()
            self.change_led(0, 0, 255)  # LED blue
            self.move_arms(leftArmPosition=90, rightArmPosition=90, duration=1.0)
        elif self.selected_action == "left":
            self.speak("Left!")
            self.emotion_Contempt()
            self.change_led(255, 255, 0)  # LED yellow
            self.move_head(yaw=-45, duration=1.0)
        elif self.selected_action == "right":
            self.speak("Right!")
            self.emotion_Amazement()
            self.change_led(255, 165, 0)  # LED orange
            self.move_head(yaw=45, duration=1.0)

        time.sleep(3)
        self.return_to_normal()
        self.speak("Now, can you remember which action I performed?")
        time.sleep(1)  # Pause for a moment before processing voice input
        self.last_transcript = ""  # Clear transcript before capturing new input

    def play_memory_game(self) -> None:
        '''
        Runs the memory game once with Misty delivering the opening dialog and selecting/performing an action.
        '''
        self.deliver_opening_dialog()
        self.select_and_perform_action()

    @staticmethod
    def fuzzy_match(a: str, b: str) -> float:
        '''Returns the similarity ratio between two strings.'''
        return SequenceMatcher(None, a, b).ratio()

    def process_multimodal_task(self) -> None:
        '''
        Process the voice input to guess the action and provide feedback.
        '''
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)
        self.play_memory_game()

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            while not self.transcript_queue.empty():
                _, self.last_transcript = self.transcript_queue.get()

            if self.last_transcript:
                print(f"User guessed: {self.last_transcript}")

                # Fuzzy matching to determine correctness of the guess
                scores = {action: self.fuzzy_match(action, self.last_transcript.lower()) for action in self.actions}
                best_guess = max(scores, key=scores.get)

                print(f"Fuzzy matching scores: {scores}")

                if scores[best_guess] > 0.3 and best_guess == self.selected_action:  # Adjust threshold for fuzzy matching
                    time.sleep(2)  # Add a 2-second pause before confirming the correct guess
                    self.speak("Correct! Let's play another round.")
                    self.select_and_perform_action()
                
                self.last_transcript = ""

            cv2.imshow("MistyVideo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def start_perception_loop(self) -> None: #-1
        '''
        Starts the perception loop of the robot.
        '''
        self._listen_for_ctrl_x()
        print("[INFO] Stopping existing streams...")
        self.stop_av_streaming()

        print("[INFO] Starting AV stream...")
        self.start_av_stream()

        print("[INFO] Loading Whisper speech recognition model...")
        self.load_whisper_model()

        threads = [
            threading.Thread(target=self._read_audio_stream, daemon=True),
            threading.Thread(target=self._process_audio, daemon=True),
            threading.Thread(target=self._video_reader_thread, daemon=True)
        ]

        for t in threads:
            t.start()

        print("[INFO] System is running. Press Ctrl+X to exit.")
        self.process_multimodal_task()

        self._stop_event.set()
        print("[MAIN] System shutdown.")

if __name__ == "__main__":
    misty = MistyMemoryGame("67.20.199.168")
    misty.start_perception_loop()