# filename: misty_voice_controlled_recording.py

from CUBS_Misty import Robot
import cv2
import numpy as np
import time
import queue

class VideoRecordingRobot(Robot):
    def __init__(self, ip: str, api_key: str) -> None:
        super().__init__(ip)
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording: bool = False
        self.api_key = api_key
        self.recorded_frames = []
        self.last_transcript = ""

    def process_multimodal_task(self) -> None:
        cv2.namedWindow("MistyVideo", cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)

                while not self.transcript_queue.empty():
                    _, self.last_transcript = self.transcript_queue.get()

                    if "go" in self.last_transcript.lower() and not self.recording:
                        self.start_recording(frame)
                    elif "stop" in self.last_transcript.lower() and self.recording:
                        self.stop_recording()

                if self.last_transcript:
                    height, _, _ = frame.shape
                    cv2.putText(
                        frame, self.last_transcript, (30, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA
                    )

                if self.recording:
                    cv2.putText(
                        frame, "RECORDING", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    self.video_writer.write(frame)
                    self.recorded_frames.append((frame, self.last_transcript))

                cv2.imshow("MistyVideo", frame)

            except queue.Empty:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._stop_event.set()
                break

        if self.recording:
            self.stop_recording()

        cv2.destroyAllWindows()

    def start_recording(self, frame: np.ndarray) -> None:
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = f"recording_{int(time.time())}.mp4"
            height, width, _ = frame.shape
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.recording = True
            self.recorded_frames = []
            print(f"[RECORD] Recording started, saving as {filename}")

    def stop_recording(self) -> None:
        if self.recording:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("[RECORD] Recording stopped")
            self.add_subtitles_to_video()

    def add_subtitles_to_video(self) -> None:
        if not self.recorded_frames:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = f"recording_with_subtitles_{int(time.time())}.mp4"
        height, width, _ = self.recorded_frames[0][0].shape
        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

        for frame, transcript in self.recorded_frames:
            if transcript:
                cv2.putText(frame, transcript, (30, height - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            video_writer.write(frame)

        video_writer.release()
        print(f"[RECORD] Saved video with subtitles as {filename}")

if __name__ == "__main__":
    misty = VideoRecordingRobot("67.20.199.168", 
                                api_key=api_key)
    misty.start_perception_loop()