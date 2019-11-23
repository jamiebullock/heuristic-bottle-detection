import cv2
import math
import numpy as np

class ColourBounds:
    def __init__(self, rgb):
        hsv = cv2.cvtColor(np.uint8([[[rgb[2], rgb[1], rgb[0]]]]), cv2.COLOR_BGR2HSV).flatten()

        lower = [hsv[0] - 10]
        upper = [hsv[0] + 10]

        if lower[0] < 0:
            lower.append(179 + lower[0]) # + negative = - abs
            upper.append(179)
            lower[0] = 0
        elif upper[0] > 179:
            lower.append(0)
            upper.append(upper[0] - 179)
            upper[0] = 179

        self.lower = [np.array([h, 100, 100]) for h in lower]
        self.upper = [np.array([h, 255, 255]) for h in upper]

def contains_vertical(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    return x1 <= x2 < x1 + w1 and x1 <= x2 + w2 < x1 + w1

def drawLabel(w, h, x, y, text, frame):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(120,0,0),2)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

colourMap = {
        "Pepsi": ColourBounds((0, 92, 184)),
        "Mountain Dew": ColourBounds((156, 161, 10)),
        "Coke": ColourBounds((244, 0, 0))
        }

cap = cv2.VideoCapture(0)

while(True):

    _, frame = cap.read()

    frame = cv2.resize(frame, (500, 280))
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    rects = {}

    for name, colour in colourMap.items():
        mask = cv2.inRange(hsv, colour.lower[0], colour.upper[0])

        if len(colour.lower) == 2:
            mask = mask | cv2.inRange(hsv, colour.lower[1], colour.upper[1])

        conts, heirarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if (len(conts) == 0):
            continue

        biggest = sorted(conts, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.boundingRect(biggest)
        x, y, w, h = rect

        if w < 50 or h < 50:
            continue

        if name == "Coke":
            if any([contains_vertical(rects[n], rect) for n in rects]):
                continue

        rects[name] = rect
        drawLabel(w, h, x, y, name, frame)

    cv2.imshow('image',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cv2.waitKey(1)

