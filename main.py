from pathlib import Path
import time
import math

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# OpenCV colors use BGR order.
HAND_COLORS = {"Left": (180, 70, 255), "Right": (255, 220, 60)}
FINGERTIPS = ((4, "Thumb"), (8, "Index"))

# Larger values reduce jitter but add more lag. Units: seconds.
SMOOTHING_TAU = 0.06

# Map brightness to navy shadows, magenta mids, cyan lights, and pale highlights.
PALETTE_STOPS = np.array([0, 90, 180, 255])
PALETTE_BGR = np.array([(30, 12, 24), (150, 45, 210),
                        (240, 210, 90), (255, 245, 255)])
COLOR_LUT = np.stack([
    np.interp(np.arange(256), PALETTE_STOPS, PALETTE_BGR[:, channel])
    for channel in range(3)
], axis=1).astype(np.uint8).reshape(256, 1, 3)

model_path = Path(__file__).with_name("hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(model_path)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    camera = cv2.VideoCapture(0)
    last_timestamp_ms = -1
    smoothing_state = {}
    smoothing_enabled = True
    effect_enabled = True

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
            debug_markers = []
            hand_labels = [categories[0].category_name
                           for categories in result.handedness]
            ambiguous = len(hand_labels) != len(set(hand_labels))

            for hand_index, (landmarks, categories) in enumerate(
                    zip(result.hand_landmarks, result.handedness)):
                hand_name = categories[0].category_name
                hand_color = HAND_COLORS.get(hand_name, (255, 255, 255))

                for tip_row, (index, finger_name) in enumerate(FINGERTIPS):
                    lm = landmarks[index]
                    key = (hand_name, finger_name)
                    x, y = lm.x, lm.y
                    if not ambiguous:
                        if smoothing_enabled and key in smoothing_state:
                            previous_x, previous_y, previous_ms = smoothing_state[key]
                            dt = (timestamp_ms - previous_ms) / 1000.0
                            alpha = 1.0 - math.exp(-dt / SMOOTHING_TAU)
                            x = previous_x + alpha * (x - previous_x)
                            y = previous_y + alpha * (y - previous_y)
                        smoothing_state[key] = (x, y, timestamp_ms)
                        fingertip_positions[key] = (x, y)
                    point = (round(x * width), round(y * height))
                    row = hand_index * 2 + tip_row
                    debug_markers.append((point, hand_name, finger_name, hand_color, row))

            # Discard history for missing hands or ambiguous identities.
            smoothing_state = {key: state for key, state in smoothing_state.items()
                               if key in fingertip_positions}

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
                # Bypass all color processing when comparing tracking performance.
                if effect_enabled:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [corners], 255)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    tinted = cv2.applyColorMap(gray, COLOR_LUT)
                    stylized = cv2.addWeighted(frame, 0.35, tinted, 0.65, 0)
                    frame[mask > 0] = stylized[mask > 0]

                # Preserve finger connections even when the edges cross.
                cv2.polylines(frame, [corners], True,
                              (230, 100, 255), 3, cv2.LINE_AA)

            # Draw diagnostics last so they remain above the video effect.
            for point, hand_name, finger_name, hand_color, row in debug_markers:
                cv2.circle(frame, point, 10, hand_color, 2, cv2.LINE_AA)
                cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, f"{hand_name} {finger_name}",
                            (point[0] + 12, point[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            hand_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{hand_name} {finger_name}: {point}",
                            (20, 70 + row * 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, hand_color, 1, cv2.LINE_AA)

            if ambiguous:
                status = "Hand identity unclear - separate hands"
            elif ready:
                status = "Frame detected"
            else:
                status = f"Hands detected: {len(result.hand_landmarks)}/2"
            color = (0, 255, 0) if ready else (0, 180, 255)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            mode = 'ON' if smoothing_enabled else 'OFF'
            effect_mode = 'ON' if effect_enabled else 'OFF'
            cv2.putText(frame, f"S: smoothing {mode} | V: color {effect_mode} | Q: quit",
                        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Spider-Gwen - Webcam", frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord("q"):
                break
            if key_pressed == ord("s"):
                smoothing_enabled = not smoothing_enabled
                smoothing_state.clear()
            if key_pressed == ord('v'):
                effect_enabled = not effect_enabled
    finally:
        camera.release()
        cv2.destroyAllWindows()






