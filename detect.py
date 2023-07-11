import cv2
import numpy as np

cap = cv2.VideoCapture("cars.mp4")
if not cap.isOpened():
    # Если камера не обнаружена, сообщите об ошибке
    raise Exception('Check if the camera is on.')

fgbg = cv2.createBackgroundSubtractorKNN(history=50, detectShadows=True)



ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

pfgmask = fgbg.apply(prev_gray)
fgmask = cv2.medianBlur(pfgmask, 5)
prev_thresh = cv2.adaptiveThreshold(pfgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)



while True:
    img, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_gray)

    prev_frame = frame
    prev_gray = frame_gray

    fgmask = cv2.medianBlur(fgmask, 5)
    thresh = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('thresh', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    old_thresh = thresh.copy()

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y + h)), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()