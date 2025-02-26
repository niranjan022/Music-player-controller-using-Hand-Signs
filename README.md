# Music-player-controller-using-Hand-Signs

1) Collect.py
=> The Collect.py file is used the get the gesture of the user for the actions such as Increase volume, Decrease volume, Next Song, Previous Song, Play/Pause Song.
Key Mappings:
1."n" - To store the next song gesture
2."P" - To store the previous song gesture
3."i" - To store the increase volume gesture
4."d" - To store the decrease volume gesture
5."a" - To store the play/pause gesture


2) Train.py
=> The train.py file is used to train the random forest classifier algorithm inorder to classify and recognise the gesture.
=> It takes the "gesture.pkl" file as input which will be created automatically in the gesture collection step (1st step).

3) app.py
=> The app.py file is used to run the programs for image capturing and sending the data to classify the gesture and atlast to communicate with the operating system to perform the intented tasks.
