import cv2
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 25
eraserThickness = 100
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
detector = htm.handDetector(detectionCon=0.85)

for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]
header_height, header_width, _ = header.shape

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 728)

fingers = []  # Initialize fingers variable

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the frame horizontally

    # Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()
        # print(fingers)

    # Check which fingers are up
    # If selection mode, two fingers are up, then select
    if len(fingers) >= 3 and fingers[1] and fingers[2]:
        # print("Selection mode")
        if y1 < 125 and header_height <= 125:
            if 150 < x1 < 220:
                header = overlayList[0]
                drawColor = (255, 0, 255)
            elif 300 < x1 < 350:
                header = overlayList[1]
                drawColor = (255, 0, 0)
            elif 400 < x1 < 460:
                header = overlayList[2]
                drawColor = (0, 255, 255)
            elif 500 < x1 < 580:
                header = overlayList[3]
                drawColor = (0, 0, 0)
        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

    # If index finger is up, draw
    if len(fingers) >= 2 and fingers[1] and not fingers[2]:
        xp, yp = 0, 0

        cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
        # print("Drawing mode")
        if xp == 0 and yp == 0:
            xp, yp = x1, y1
        if drawColor == (0, 0, 0):
            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
        else:
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
        xp, yp = x1, y1

    imGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
     # Resize imgCanvas to match img size

    # img = cv2.bitwise_and(img, imgInv)
    # img = cv2.bitwise_or(img, imgCanvasResized)

    img_height, img_width, _ = img.shape
    header_resized = cv2.resize(header, (img_width, header_height))
    img[0:header_height, 0:img_width] = header_resized


    cv2.imshow("img", img)
    cv2.imshow("imgcanvas", imgCanvas)


    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break





cap.release()
cv2.destroyAllWindows()
