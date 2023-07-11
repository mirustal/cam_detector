import cv2
import numpy as np


def calculate_optical_flow(prev_frame, curr_frame, prev_points):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # вычисляем перемещение точек на текущем кадре
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)

    # отбрасываем точки, для которых перемещение не удалось определить
    #curr_points = curr_points[status == 1].reshape(-1, 2)

    #print('---------------\n', curr_points)
    # определяем смещение центра объекта
    delta = np.mean(curr_points- prev_points.reshape(-1, 2), axis=0)


    return curr_points


cap = cv2.VideoCapture("cars.mp4")
if not cap.isOpened():
    # Если камера не обнаружена, сообщите об ошибке
    raise Exception('Check if the camera is on.')

fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# Lucas-Kanade parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# cv2.GaussianBlur(prev_frame, (5, 5), 5)
prev_mask = None
object_point = []
while True:
    ret, frame = cap.read()
    mask = np.zeros_like(frame)
    if not ret:
        break
    # cv2.imshow("orig_frame", frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame_gray)


    prev_frame = frame
    prev_gray = frame_gray
    fgmask = cv2.medianBlur(fgmask, 5)
    # cv2.imshow("fgmask", fgmask)
    thresh = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # cv2.imshow("adaptiveThreshol", thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("morphologyEx", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 250:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        object_contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])

        # проверяем, была ли уже обработана область, содержащая контур
        if mask[y:y + h, x:x + w].any():
            continue
        # отмечаем область на маске как обработанную
        mask[y:y + h, x:x + w] = 255

    #mask = np.ones(thresh.shape[:2], dtype=bool)  # создаем маску, все пиксели которой равны True

    for contour in contours:
        if cv2.contourArea(contour) < 250:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        # проверяем, не был ли этот контур уже найден
        if not mask[y:y + h, x:x + w].any():
            continue

        # устанавливаем значения пикселей маски, соответствующих этому контуру, в 0
        mask[y:y + h, x:x + w] = False

        object_contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
        # определяем точки контура в предыдущем кадре
        prev_points = np.array([[x, y] for [[x, y]] in object_contour], dtype=np.float32).reshape(-1, 1, 2)

        delta = calculate_optical_flow(prev_frame, frame, prev_points)
        delta = delta.reshape(delta.shape[0], delta.shape[2])
        # result = np.concatenate(prev_points, delta)
        # print('\npoint1:', *prev_points[1], 'point3:', *prev_points[3])
        (x, y) = delta[0]
        (xw, yh) = delta[2]
        cv2.rectangle(frame, (int(x), int(y)), (int(xw), int(yh)), (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()