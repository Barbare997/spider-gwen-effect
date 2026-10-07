# Spider-Gwen-Inspired Webcam Effect

An evolving Python computer-vision project that uses hand tracking
to control a visual frame over live webcam video.

## Current features

- Real-time webcam capture
- Two-hand tracking with MediaPipe
- Thumb and index fingertip tracking on each hand
- A finger-controlled frame that can transition into a crossed bow-tie shape

## Technology

Python, OpenCV, MediaPipe, and NumPy.

## Status

Work in progress. Planned additions include temporal smoothing,
comic-style video inside the frame, animated trails, and glow.

## Running locally

Requires Python 3.12 and a webcam.

On Windows, create a virtual environment and install dependencies:

    python -m venv .venv
    .venv\Scripts\activate.bat
    python -m pip install mediapipe opencv-contrib-python numpy

Download hand_landmarker.task from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Place it beside main.py, then run:

    python main.py

Click the video window and press Q to exit.

## Learning focus

Built incrementally to explore hand tracking, frame geometry,
temporal filtering, and real-time visual compositing.