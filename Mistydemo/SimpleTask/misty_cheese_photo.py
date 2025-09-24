# filename: misty_cheese_photo.py

from CUBS_Misty import Robot
import cv2
import queue
import time
import numpy as np

class MistyCheesePhoto(Robot):
    def __init__(self, ip: str) -> None:
        super().__init__(ip)
        self._photo_taken: bool = False

    def process_multimodal_task(self) -> None:
        self.return_to_normal()
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            while not self.transcript_queue.empty():
                _, self.last_transcript = self.transcript_queue.get()
                self._photo_taken = False

            if self.last_transcript:
                cv2.putText(
                    frame, self.last_transcript, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA
                )

                # Detect the verbal cue 'Cheese'
                if "cheese" in self.last_transcript.lower() and not self._photo_taken:
                    self.take_photo(frame)
                    self._photo_taken = True

            cv2.imshow("MistyVideo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        cv2.destroyAllWindows()

    def take_photo(self, frame: np.ndarray) -> None:
        # Save the captured frame as an image file
        photo_filename = f"photo_{int(time.time())}.jpg"
        cv2.imwrite(photo_filename, frame)
        print(f"[PHOTO] Photo saved as {photo_filename}")

if __name__ == "__main__":
    # Instantiate the MistyCheesePhoto class with the provided IP
    misty = MistyCheesePhoto("67.20.199.168")

    # Begin the perception loop
    misty.start_perception_loop()