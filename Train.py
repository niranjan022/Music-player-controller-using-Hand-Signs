import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Define paths and gestures
GESTURES = ["next_song", "prev_song", "increase_volume", "decrease_volume", "play_pause"]
DATA_PATH = "gesture_data"

# Load data
X, y = [], []
for label, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(DATA_PATH, gesture)
    for file in os.listdir(gesture_path):
        data = np.load(os.path.join(gesture_path, file))
        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(clf, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")
