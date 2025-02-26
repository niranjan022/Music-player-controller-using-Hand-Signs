import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import keyboard as ky
import pygetwindow as gw

# Load the trained model
clf = joblib.load("gesture_model.pkl")
GESTURES = ["next_song", "prev_song", "increase_volume", "decrease_volume","pause_play"]
INVALID_MESSAGE = "Not valid"

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
COMMANDS = {
         "increase_volume": 'nircmd changesysvolume 1000',
          "decrease_volume": 'nircmd changesysvolume -1000'  # Toggle play/pause using space key
    }

# Function to execute the command
def execute_command(command):
    if command in COMMANDS:
        os.system(COMMANDS[command])
    else:
        print("Command not recognized:", command)
def execute_other_commands(command):
        # Get a list of all open windows
    windows = gw.getAllTitles()
        # Set the target window title
    target_window_title = "Windows Media Player Legacy"

        # Check if the target window exists and bring it to the foreground
    for window in windows:
      
      if target_window_title in window:
            win = gw.getWindowsWithTitle(target_window_title)[0]
            win.activate()
            break
    else:
        print(f"Window with title '{target_window_title}' not found.")
        exit()

    comm = {
        "next_song": 'ctrl+f',
        "prev_song": 'ctrl+b',
        "pause_play": 'ctrl+p'
        }
   
    ky.press_and_release(comm[command])

# Function to start Groove Music (if not already running)
def start_music():
    os.system("start wmplayer")
    print("Windows Media Player Legacy opened.")

cap = cv2.VideoCapture(0)

# Start Groove Music automatically when the script starts
start_music()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Detect gesture
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            try:
                prediction_probabilities = clf.predict_proba(landmarks)[0]  # Get probabilities for all classes
                max_prob = max(prediction_probabilities)                   # Get the highest probability
                predicted_class = np.argmax(prediction_probabilities)      # Get the class index

                if max_prob > 0.8:  # Confidence threshold
                    gesture = GESTURES[predicted_class]
                    print(f"Recognized Gesture: {gesture}")
                    if gesture not in ["increase_volume", "decrease_volume"]:
                        execute_other_commands(gesture)
                    else:
                        execute_command(gesture)
                    time.sleep(0.2)
                else:
                    gesture = INVALID_MESSAGE

            except Exception as e:
                gesture = INVALID_MESSAGE
                print("Error:", e)

            # Display gesture
            cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    else:
        # If no hand is detected
        cv2.putText(frame, f"Gesture: {INVALID_MESSAGE}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
