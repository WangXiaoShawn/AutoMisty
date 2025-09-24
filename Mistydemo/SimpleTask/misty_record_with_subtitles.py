# filename: misty_record_with_subtitles.py

from CUBS_Misty import Robot
import cv2
import numpy as np
import time
import queue

class MistyRecordWithSubtitles(Robot):
    def __init__(self, ip: str) -> None:
        super().__init__(ip)
        self.video_writer = None
        self.recording = False
        self.subtitle_queue = queue.Queue()

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                while not self.transcript_queue.empty():
                    _, transcript = self.transcript_queue.get()
                    self.subtitle_queue.put(transcript)

                    if "go" in transcript.lower() and not self.recording:
                        self.start_recording(frame)
                    elif "done" in transcript.lower() and self.recording:
                        self.stop_recording()
                
                if self.recording:
                    self.overlay_subtitles(frame)
                    self.video_writer.write(frame)

                cv2.imshow("MistyVideo", frame)

            except queue.Empty:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                if self.recording:
                    self.stop_recording()
                break

        cv2.destroyAllWindows()

    def start_recording(self, frame: np.ndarray) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f"recording_{int(time.time())}.mp4"
        height, width, _ = frame.shape
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        self.recording = True
        print(f"[RECORD] Recording started and saving as {filename}")

    def stop_recording(self) -> None:
        if self.recording:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("[RECORD] Recording stopped")

    def overlay_subtitles(self, frame: np.ndarray) -> None:
        if not self.subtitle_queue.empty():
            subtitle = self.subtitle_queue.get()
            cv2.putText(
                frame, subtitle, (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA
            )

if __name__ == "__main__":
    misty = MistyRecordWithSubtitles("67.20.199.168")
    misty.start_perception_loop()