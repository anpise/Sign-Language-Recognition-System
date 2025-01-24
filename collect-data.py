import cv2
import numpy as np
import os
import string

# Create the greyscaleDirectory structure
if not os.path.exists("greyscale/data"):
    os.makedirs("greyscale/data")
if not os.path.exists("greyscale/data/train"):
    os.makedirs("greyscale/data/train")
if not os.path.exists("greyscale/data/test"):
    os.makedirs("greyscale/data/test")

if not os.path.exists("gaussian/data"):
    os.makedirs("gaussian/data")
if not os.path.exists("gaussian/data/train"):
    os.makedirs("gaussian/data/train")
if not os.path.exists("gaussian/data/test"):
    os.makedirs("gaussian/data/test")

for i in string.ascii_uppercase:
    if not os.path.exists("greyscale/data/train/" + i):
        os.makedirs("greyscale/data/train/" + i)
    if not os.path.exists("greyscale/data/test/" + i):
        os.makedirs("greyscale/data/test/" + i)

for i in string.ascii_uppercase:
    if not os.path.exists("gaussian/data/train/" + i):
        os.makedirs("gaussian/data/train/" + i)
    if not os.path.exists("gaussian/data/test/" + i):
        os.makedirs("gaussian/data/test/" + i)

# Train or test
mode = 'train'
greyscaleDirectory = 'greyscale/data/' + mode + '/'
gaussianDirectory = 'gaussian/data/' + mode + '/'
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {
        # 'blank': len(os.listdir(greyscaleDirectory + "/blank")),
        'a': len(os.listdir(greyscaleDirectory + "/A")),
        'b': len(os.listdir(greyscaleDirectory + "/B")),
        'c': len(os.listdir(greyscaleDirectory + "/C")),
        'd': len(os.listdir(greyscaleDirectory + "/D")),
        'e': len(os.listdir(greyscaleDirectory + "/E")),
        'f': len(os.listdir(greyscaleDirectory + "/F")),
        'g': len(os.listdir(greyscaleDirectory + "/G")),
        'h': len(os.listdir(greyscaleDirectory + "/H")),
        'i': len(os.listdir(greyscaleDirectory + "/I")),
        'j': len(os.listdir(greyscaleDirectory + "/J")),
        'k': len(os.listdir(greyscaleDirectory + "/K")),
        'l': len(os.listdir(greyscaleDirectory + "/L")),
        'm': len(os.listdir(greyscaleDirectory + "/M")),
        'n': len(os.listdir(greyscaleDirectory + "/N")),
        'o': len(os.listdir(greyscaleDirectory + "/O")),
        'p': len(os.listdir(greyscaleDirectory + "/P")),
        'q': len(os.listdir(greyscaleDirectory + "/Q")),
        'r': len(os.listdir(greyscaleDirectory + "/R")),
        's': len(os.listdir(greyscaleDirectory + "/S")),
        't': len(os.listdir(greyscaleDirectory + "/T")),
        'u': len(os.listdir(greyscaleDirectory + "/U")),
        'v': len(os.listdir(greyscaleDirectory + "/V")),
        'w': len(os.listdir(greyscaleDirectory + "/W")),
        'x': len(os.listdir(greyscaleDirectory + "/X")),
        'y': len(os.listdir(greyscaleDirectory + "/Y")),
        'z': len(os.listdir(greyscaleDirectory + "/Z"))

    }

    # Printing the count in each set to the screen

    cv2.putText(frame, "a : " + str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "b : " + str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "c : " + str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "d : " + str(count['d']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "e : " + str(count['e']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "f : " + str(count['f']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "g : " + str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "h : " + str(count['h']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "i : " + str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "k : " + str(count['k']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "l : " + str(count['l']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "m : " + str(count['m']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "n : " + str(count['n']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "o : " + str(count['o']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "p : " + str(count['p']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "q : " + str(count['q']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "r : " + str(count['r']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "s : " + str(count['s']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "t : " + str(count['t']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "u : " + str(count['u']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "v : " + str(count['v']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "w : " + str(count['w']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "x : " + str(count['x']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "y : " + str(count['y']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "z : " + str(count['z']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    # Coordinates of the ROI

    x1 = int(0.65 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 5
    y2 = int(0.45 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    test_image = cv2.resize(test_image, (300, 300))
    # cv2.imshow("test", test_image)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GrayScale", gray)

    # cv2.imshow("ROI", roi)
    # roi = frame
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(gaussianDirectory + 'A/' + str(count['a']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'A/' + str(count['a']) + '.jpg', gray)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(gaussianDirectory + 'B/' + str(count['b']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'B/' + str(count['b']) + '.jpg', gray)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(gaussianDirectory + 'C/' + str(count['c']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'C/' + str(count['c']) + '.jpg', gray)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(gaussianDirectory + 'D/' + str(count['d']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'D/' + str(count['d']) + '.jpg', gray)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(gaussianDirectory + 'E/' + str(count['e']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'E/' + str(count['e']) + '.jpg', gray)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(gaussianDirectory + 'F/' + str(count['f']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'F/' + str(count['f']) + '.jpg', gray)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(gaussianDirectory + 'G/' + str(count['g']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'G/' + str(count['g']) + '.jpg', gray)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(gaussianDirectory + 'H/' + str(count['h']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'H/' + str(count['h']) + '.jpg', gray)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(gaussianDirectory + 'I/' + str(count['i']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'I/' + str(count['i']) + '.jpg', gray)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(gaussianDirectory + 'J/' + str(count['j']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'J/' + str(count['j']) + '.jpg', gray)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(gaussianDirectory + 'K/' + str(count['k']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'K/' + str(count['k']) + '.jpg', gray)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(gaussianDirectory + 'L/' + str(count['l']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'L/' + str(count['l']) + '.jpg', gray)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(gaussianDirectory + 'M/' + str(count['m']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'M/' + str(count['m']) + '.jpg', gray)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(gaussianDirectory + 'N/' + str(count['n']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'N/' + str(count['n']) + '.jpg', gray)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(gaussianDirectory + 'O/' + str(count['o']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'O/' + str(count['o']) + '.jpg', gray)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(gaussianDirectory + 'P/' + str(count['p']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'P/' + str(count['p']) + '.jpg', gray)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(gaussianDirectory + 'Q/' + str(count['q']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'Q/' + str(count['q']) + '.jpg', gray)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(gaussianDirectory + 'R/' + str(count['r']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'R/' + str(count['r']) + '.jpg', gray)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(gaussianDirectory + 'S/' + str(count['s']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'S/' + str(count['s']) + '.jpg', gray)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(gaussianDirectory + 'T/' + str(count['t']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'T/' + str(count['t']) + '.jpg', gray)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(gaussianDirectory + 'U/' + str(count['u']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'U/' + str(count['u']) + '.jpg', gray)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(gaussianDirectory + 'V/' + str(count['v']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'V/' + str(count['v']) + '.jpg', gray)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(gaussianDirectory + 'W/' + str(count['w']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'W/' + str(count['w']) + '.jpg', gray)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(gaussianDirectory + 'X/' + str(count['x']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'X/' + str(count['x']) + '.jpg', gray)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(gaussianDirectory + 'Y/' + str(count['y']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'Y/' + str(count['y']) + '.jpg', gray)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(gaussianDirectory + 'Z/' + str(count['z']) + '.jpg', test_image)
        cv2.imwrite(greyscaleDirectory + 'Z/' + str(count['z']) + '.jpg', gray)

cap.release()
cv2.destroyAllWindows()

