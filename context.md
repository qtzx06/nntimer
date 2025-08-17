# Project Plan: nntimer

## 1. Project Objective

The primary goal of this project is to develop **nntimer**, a hands-free Rubik's Cube timer. The application will use a standard webcam to visually track the solving process, automatically starting the timer when a solve begins and stopping it upon completion.

## 2. Core Functionality & User Experience

- A simple web-based UI served locally using FastAPI.
- The UI displays a live webcam feed.
- The application visually identifies a Rubik's Cube in the frame.
- A timer starts when the user's hands touch a stationary, unsolved cube.
- The timer stops the moment the cube is visually identified as "solved."
- Solve times are sent to the UI in real-time via WebSockets and logged to a file.

## 3. Technology Stack

- **Primary Language:** Python 3.10
- **Package Manager:** `uv`
- **Computer Vision:** OpenCV, MediaPipe Hands
- **Machine Learning:** PyTorch (with ROCm), Ultralytics (YOLOv8)
- **Web Stack:** FastAPI, Uvicorn, WebSockets

## 4. AI Components

- **Cube Detector (YOLOv8n):** An ONNX model (`best.onnx`) fine-tuned to find the cube's bounding box and classify it as `SolvedCube` or `UnsolvedCube`.
- **Cube State Classifier (ResNet18):** A PyTorch model (`cube_classifier.pth`) trained to provide a high-accuracy binary classification of the cube's state (`SOLVED` vs. `UNSOLVED`).

## 5. Project Status & Journey

- **[DONE]** Initial project setup, including dependencies and ROCm-enabled PyTorch.
- **[DONE]** `train_classifier.py`: Successfully trained the ResNet18 model to **100% validation accuracy**.
  - *Troubleshooting:* Resolved a `DataLoader` crash by implementing a `spawn` start method for multiprocessing and added `tqdm` for a better UX.
- **[DONE]** `train_detector.py`: Successfully trained an initial YOLOv8n model and exported to ONNX.
- **[DONE]** `main.py`: Integrated both models and MediaPipe for real-time visualization.
- **[PROBLEM]** The initial detector model is not robust. It confidently misidentifies other objects (like a keyboard) as the cube. This indicates the training data lacks diversity.
- **[SOLUTION]** Created `capture_data.py` to gather more varied training images, including "negative" examples of the background without a cube.
- **[CURRENT BLOCKER]** The capture script cannot access the webcam due to a system permissions issue.
  - *Diagnosis:* The user `qtzx` is not in the `video` group, which is required by OpenCV for direct camera access.
  - *Resolution:* The user needs to run `sudo usermod -aG video $USER` and then log out and log back in.

## 6. Next Steps

1.  **[PENDING]** **Resolve Permissions:** User to log out and back in to apply the `video` group membership.
2.  **[NEXT]** **Gather Data:** Run `capture_data.py` to capture new positive and negative training images.
3.  **Retrain Detector:** Upload and annotate the new images in Roboflow, then train a new, more robust detector model.
4.  **Integrate New Model:** Update `main.py` with the new detector model.
5.  **Implement State Machine:** Build the core timer logic based on the model outputs and hand tracking.
