from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# OpenCV colors use BGR order.
HAND_COLORS = {"Left": (180, 70, 255), "Right": (255, 220, 60)}
FINGERTIPS = ((4, "Thumb"), (8, "Index"))

model_path = Path(__file__).with_name("hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(model_path)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    camera = cv2.VideoCapture(0)
    last_timestamp_ms = -1

    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open the webcam.")

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a webcam frame.")
                break

            timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(image, timestamp_ms)

            height, width = frame.shape[:2]
            fingertip_positions = {}
            hand_labels = [categories[0].category_name
                           for categories in result.handedness]
            ambiguous = len(hand_labels) != len(set(hand_labels))

            for hand_index, (landmarks, categories) in enumerate(
                    zip(result.hand_landmarks, result.handedness)):
                hand_name = categories[0].category_name
                hand_color = HAND_COLORS.get(hand_name, (255, 255, 255))

                for tip_row, (index, finger_name) in enumerate(FINGERTIPS):
                    lm = landmarks[index]
                    if not ambiguous:
                        fingertip_positions[(hand_name, finger_name)] = (lm.x, lm.y)
                    point = (round(lm.x * width), round(lm.y * height))
                    cv2.circle(frame, point, 10, hand_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(frame, f"{hand_name} {finger_name}",
                                (point[0] + 12, point[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                hand_color, 1, cv2.LINE_AA)
                    row = hand_index * 2 + tip_row
                    cv2.putText(frame, f"{hand_name} {finger_name}: {point}",
                                (20, 70 + row * 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, hand_color, 1, cv2.LINE_AA)

            ready = len(fingertip_positions) == 4
            if ready:
                corner_keys = (
                    ("Left", "Index"), ("Right", "Index"),
                    ("Right", "Thumb"), ("Left", "Thumb"),
                )
                corners = np.array([
                    (round(fingertip_positions[key][0] * width),
                     round(fingertip_positions[key][1] * height))
                    for key in corner_keys
                ], dtype=np.int32).reshape(-1, 1, 2)
                # Preserve finger connections even when the edges cross.
                cv2.polylines(frame, [corners], True,
                              (230, 100, 255), 3, cv2.LINE_AA)

            if ambiguous:
                status = "Hand identity unclear - separate hands"
            elif ready:
                status = "Frame detected"
            else:
                status = f"Hands detected: {len(result.hand_landmarks)}/2"
            color = (0, 255, 0) if ready else (0, 180, 255)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            cv2.imshow("Spider-Gwen - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()



