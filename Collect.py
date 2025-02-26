import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define gestures
GESTURES = ["next_song", "prev_song", "increase_volume", "decrease_volume", "play_pause"]
DATA_PATH = "gesture_data"

# Create folders for each gesture
os.makedirs(DATA_PATH, exist_ok=True)
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

print("Press 'n' to save data for the 'next_song' gesture")
print("Press 'p' to save data for the 'prev_song' gesture")
print("Press 'i' to save data for the 'increase_volume' gesture")
print("Press 'd' to save data for the 'decrease_volume' gesture")
print("Press 'a' to save data for the 'play_pause' gesture")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw landmarks and collect data
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to a flattened array
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save data based on key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Next Song
                np.save(os.path.join(DATA_PATH, "next_song", f"gesture_{len(os.listdir(os.path.join(DATA_PATH, 'next_song')))}.npy"), landmarks)
                print("Saved 'next_song' gesture")
            elif key == ord('p'):  # Previous Song
                np.save(os.path.join(DATA_PATH, "prev_song", f"gesture_{len(os.listdir(os.path.join(DATA_PATH, 'prev_song')))}.npy"), landmarks)
                print("Saved 'prev_song' gesture")
            elif key == ord('i'):  # Increase Volume
                np.save(os.path.join(DATA_PATH, "increase_volume", f"gesture_{len(os.listdir(os.path.join(DATA_PATH, 'increase_volume')))}.npy"), landmarks)
                print("Saved 'increase_volume' gesture")
            elif key == ord('d'):  # Decrease Volume
                np.save(os.path.join(DATA_PATH, "decrease_volume", f"gesture_{len(os.listdir(os.path.join(DATA_PATH, 'decrease_volume')))}.npy"), landmarks)
                print("Saved 'decrease_volume' gesture")
            elif key == ord('a'):  # Play/Pause
                np.save(os.path.join(DATA_PATH, "play_pause", f"gesture_{len(os.listdir(os.path.join(DATA_PATH, 'play_pause')))}.npy"), landmarks)
                print("Saved 'play_pause' gesture")

    # Display the frame
    cv2.imshow("Gesture Capture", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
