import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    width = int(cap.get(3))
    height = int(cap.get(4))

    for (x, y, w, h) in faces:
        cv2.circle(frame, (x + w//2, y + h//2), w//2, (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 1)

    image = np.zeros(frame.shape, np.uint8)

    flipped_frame = cv2.flip(frame, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(faces) > 0:
        text = cv2.putText(flipped_frame, 'face detected!', (50, height - 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', flipped_frame)

    if cv2.waitKey(1) == ord('q'):
        break
        

cap.release()
cv2.destroyAllWindows()
