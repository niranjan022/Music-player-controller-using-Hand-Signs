# Music-player-controller-using-Hand-Signs

## 1) Collect.py
The `Collect.py` file is used to get the gesture of the user for the following actions:
- **Increase volume**
- **Decrease volume**
- **Next Song**
- **Previous Song**
- **Play/Pause Song**

### Key Mappings:
- **"n"** - To store the next song gesture
- **"P"** - To store the previous song gesture
- **"i"** - To store the increase volume gesture
- **"d"** - To store the decrease volume gesture
- **"a"** - To store the play/pause gesture

## 2) Train.py
The `train.py` file is used to train the **Random Forest Classifier** algorithm in order to classify and recognize the gestures.

- It takes the `gesture.pkl` file as input, which is automatically created during the gesture collection step (Step 1).

## 3) app.py
The `app.py` file is used to:
- Run the program for **image capturing**
- Send the data for **gesture classification**
- Communicate with the **operating system** to perform the intended music player tasks based on recognized gestures.

---

This project leverages **OpenCV** and **MediaPipe** for hand gesture recognition and integrates machine learning models to control music playback intuitively!

