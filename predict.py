from string import ascii_uppercase

import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = 370
    y1 = 10
    x2 = 600
    y2 = 240
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # _, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # # _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", gray)
    # Batch of 1
    result = loaded_model.predict(gray.reshape(1, 64, 64, 1))
    prediction = {}
    prediction = {'A': result[0][0],
                  'B': result[0][1],
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6],
                  'H': result[0][7],
                  'I': result[0][8],
                  'J': result[0][9],
                  'K': result[0][10],
                  'L': result[0][11],
                  'M': result[0][12],
                  'N': result[0][13],
                  'O': result[0][14],
                  'P': result[0][15],
                  'Q': result[0][16],
                  'R': result[0][17],
                  'S': result[0][18],
                  'T': result[0][19],
                  'U': result[0][20],
                  'V': result[0][21],
                  'W': result[0][22],
                  'X': result[0][23],
                  'Y': result[0][24],
                  'Z': result[0][25],
                  'blank': result[0][26],
                  'space': result[0][27]}
    # # Sorting based on top prediction
    # prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)


    # inde = 0
    # for i in ascii_uppercase:
    #     prediction[i] = result[0][inde]
    #     inde += 1
    # # LAYER 1
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break

cap.release()
cv2.destroyAllWindows()