import cv2
import time
import numpy as np
import onnxruntime
import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import threading
import math
import torch
from torchvision import models, transforms
from PIL import Image

# ==================================================================================
# --- values to change the timer's behavior ---
# ==================================================================================
CLASSIFIER_WEIGHT = 0.7
DETECTOR_WEIGHT = 0.3
CONFIDENCE_THRESHOLD = 0.25
SOLVED_STATE_CONFIDENCE_THRESHOLD = 0.80
HAND_STEADY_DURATION_REQ = 0.4
HAND_STEADY_THRESHOLD = 0.015
HAND_MOTION_START_THRESHOLD = 0.08
PROCESS_EVERY_N_FRAMES = 2
DETECTION_GRACE_PERIOD = 5
SOLVED_CONFIRMATION_FRAMES = 3
COLOR_VARIATION_THRESHOLD = 7.0
# ==================================================================================

# --- config ---
DETECTOR_PATH = "runs/detect/train/weights/best.onnx"
CLASSIFIER_PATH = "cube_classifier.pth"
CLASSIFIER_CLASS_NAMES = ["solved", "unsolved"]
DETECTOR_CLASS_NAMES = ["SolvedCube", "UnsolvedCube"]

# --- fastapi app setup ---
app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# --- model loading ---
print("loading models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ort_session = onnxruntime.InferenceSession(DETECTOR_PATH)
detector_input_name = ort_session.get_inputs()[0].name
classifier_model = models.resnet18()
classifier_model.fc = torch.nn.Linear(
    classifier_model.fc.in_features, len(CLASSIFIER_CLASS_NAMES)
)
classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier_model = classifier_model.to(device)
classifier_model.eval()
print(f"models loaded successfully on device: {device}")

# --- classifier transforms ---
classifier_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# --- state machine & state management ---
class TimerState:
    IDLE, READY, SET, SOLVING, SOLVED = "IDLE", "READY", "SET", "SOLVING", "SOLVED"


class StateManager:
    def __init__(self):
        self.latest_frame, self.current_status, self.solve_time = (
            None,
            TimerState.IDLE,
            0.0,
        )
        self._lock = threading.Lock()

    def set_frame(self, frame):
        with self._lock:
            self.latest_frame = frame

    def get_frame(self):
        with self._lock:
            return self.latest_frame

    def set_status(self, status, solve_time=None):
        with self._lock:
            self.current_status = status
            if solve_time is not None:
                self.solve_time = solve_time

    def get_status(self):
        with self._lock:
            return {"status": self.current_status, "time": f"{self.solve_time:.2f}"}


state_manager = StateManager()


# --- helper functions ---
def preprocess_for_detector(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def postprocess_detector_output(output, frame_shape):
    output = output[0].T
    best_box, detector_probs = None, {name: 0.0 for name in DETECTOR_CLASS_NAMES}
    best_confidence = 0.0
    for row in output:
        class_confidence = row[4:].max()
        if class_confidence > best_confidence:
            best_confidence = class_confidence
            if class_confidence > CONFIDENCE_THRESHOLD:
                detector_probs = {
                    name: prob for name, prob in zip(DETECTOR_CLASS_NAMES, row[4:])
                }
                cx, cy, w, h = row[:4]
                h_orig, w_orig = frame_shape
                x1, y1 = (
                    int((cx - w / 2) * w_orig / 640),
                    int((cy - h / 2) * h_orig / 640),
                )
                x2, y2 = (
                    int((cx + w / 2) * w_orig / 640),
                    int((cy + h / 2) * h_orig / 640),
                )
                best_box = [x1, y1, x2, y2]
    return best_box, detector_probs


def classify_cube_state(image_crop, model):
    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    input_tensor = classifier_transforms(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    classifier_probs = {
        name: prob.item() for name, prob in zip(CLASSIFIER_CLASS_NAMES, probabilities)
    }
    return classifier_probs


def has_enough_color_variation(crop, threshold):
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_std_dev = np.std(hsv[:, :, 0])
    print(f"color variation (hue std dev): {hue_std_dev:.2f}")
    return hue_std_dev > threshold


# --- background video processing thread ---
def video_processing_thread():
    state, start_time, solve_time, frame_count, detection_miss_counter = (
        TimerState.IDLE,
        0,
        0,
        0,
        0,
    )
    last_hand_pos, steady_start_time = None, 0
    last_known_bbox, last_known_class_name, last_known_confidence = None, None, 0.0
    last_known_display_probs = {name: 0.0 for name in CLASSIFIER_CLASS_NAMES}
    last_landmarks = None
    solved_confirmation_counter = 0

    mp_hands = mp.solutions.hands
    hands = mp.solutions.hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2
    )

    # --- custom drawing styles for mediapipe ---
    drawing_spec_dots = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 0, 255), thickness=3, circle_radius=3
    )  # bright red dots
    drawing_spec_lines = mp.solutions.drawing_utils.DrawingSpec(
        color=(255, 255, 255), thickness=3
    )  # bright white lines

    cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(
                f"webcam opened successfully at index {i} with resolution {width}x{height}."
            )
            break
    if not cap or not cap.isOpened():
        print("error: could not open any webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("error: can't receive frame from webcam.")
            break

        frame_count += 1
        current_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        hands_present = hand_results.multi_hand_landmarks is not None

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            detector_input = preprocess_for_detector(frame)
            detector_output = ort_session.run(
                None, {detector_input_name: detector_input}
            )[0]
            cube_bbox, detector_probs = postprocess_detector_output(
                detector_output, frame.shape[:2]
            )

            if cube_bbox:
                detection_miss_counter = 0
                last_known_bbox = cube_bbox
                x1, y1, x2, y2 = cube_bbox
                cropped_cube = frame[y1:y2, x1:x2]

                if has_enough_color_variation(cropped_cube, COLOR_VARIATION_THRESHOLD):
                    classifier_probs = classify_cube_state(
                        cropped_cube, classifier_model
                    )
                    combined_solved_confidence = (
                        classifier_probs["solved"] * CLASSIFIER_WEIGHT
                    ) + (detector_probs["SolvedCube"] * DETECTOR_WEIGHT)
                    last_known_class_name = (
                        "solved" if combined_solved_confidence > 0.5 else "unsolved"
                    )
                    last_known_confidence = combined_solved_confidence
                    last_known_display_probs = classifier_probs
            else:
                detection_miss_counter += 1

        cube_is_present = detection_miss_counter <= DETECTION_GRACE_PERIOD

        # --- state machine logic ---
        if state == TimerState.IDLE:
            if cube_is_present and last_known_class_name == "unsolved":
                state = TimerState.READY
        elif state == TimerState.READY:
            if not cube_is_present:
                state = TimerState.IDLE
            elif hands_present:
                wrist_pos = hand_results.multi_hand_landmarks[0].landmark[
                    mp_hands.HandLandmark.WRIST
                ]
                current_hand_pos = (wrist_pos.x, wrist_pos.y)
                if last_hand_pos is None:
                    last_hand_pos, steady_start_time = current_hand_pos, current_time
                movement = math.sqrt(
                    (current_hand_pos[0] - last_hand_pos[0]) ** 2
                    + (current_hand_pos[1] - last_hand_pos[1]) ** 2
                )
                if movement < HAND_STEADY_THRESHOLD:
                    if current_time - steady_start_time > HAND_STEADY_DURATION_REQ:
                        state, last_landmarks = (
                            TimerState.SET,
                            hand_results.multi_hand_landmarks,
                        )
                else:
                    steady_start_time = current_time
                last_hand_pos = current_hand_pos
            else:
                last_hand_pos = None
        elif state == TimerState.SET:
            if not cube_is_present:
                state = TimerState.IDLE
            elif not hands_present:
                state, start_time, solve_time = TimerState.SOLVING, current_time, 0
            elif hands_present and last_landmarks:
                total_velocity = sum(
                    math.sqrt(
                        (
                            hand_landmarks.landmark[lm_idx].x
                            - last_landmarks[i].landmark[lm_idx].x
                        )
                        ** 2
                        + (
                            hand_landmarks.landmark[lm_idx].y
                            - last_landmarks[i].landmark[lm_idx].y
                        )
                        ** 2
                    )
                    for i, hand_landmarks in enumerate(
                        hand_results.multi_hand_landmarks
                    )
                    if i < len(last_landmarks)
                    for lm_idx in [
                        mp_hands.HandLandmark.THUMB_TIP,
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    ]
                )
                print(f"hand motion velocity: {total_velocity:.4f}")
                if total_velocity > HAND_MOTION_START_THRESHOLD:
                    state, start_time, solve_time = TimerState.SOLVING, current_time, 0
                last_landmarks = hand_results.multi_hand_landmarks
        elif state == TimerState.SOLVING:
            solve_time = current_time - start_time
            if (
                last_known_class_name == "solved"
                and hands_present
                and last_known_confidence > SOLVED_STATE_CONFIDENCE_THRESHOLD
            ):
                solved_confirmation_counter += 1
                if solved_confirmation_counter >= SOLVED_CONFIRMATION_FRAMES:
                    state = TimerState.SOLVED
            else:
                solved_confirmation_counter = 0
        elif state == TimerState.SOLVED:
            if not hands_present:
                state = TimerState.READY

        state_manager.set_status(state, solve_time)

        # --- drawing logic ---
        display_text = state
        if state == TimerState.SOLVING:
            display_text = f"solving: {solve_time:.2f}"
        elif state == TimerState.SOLVED:
            display_text = f"solved: {solve_time:.2f}"

        if detection_miss_counter > DETECTION_GRACE_PERIOD:
            last_known_bbox = None
        if last_known_bbox:
            x1, y1, x2, y2 = last_known_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                display_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        y_offset = 30
        for name, prob in last_known_display_probs.items():
            text = f"{name}: {prob:.2f}"
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec_dots,
                    connection_drawing_spec=drawing_spec_lines,
                )

        _, buffer = cv2.imencode(".jpg", frame)
        state_manager.set_frame(buffer.tobytes())

    cap.release()
    hands.close()


# --- api endpoints ---
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=video_processing_thread, daemon=True).start()


@app.get("/")
async def read_root():
    with open("frontend/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/status")
async def get_status():
    return state_manager.get_status()


def frame_generator():
    while True:
        frame_bytes = state_manager.get_frame()
        if frame_bytes:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.01)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
