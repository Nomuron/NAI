"""
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
opencv-python package
mediapipes package

Link to install python: https://www.python.org/downloads/
To run script you need to run command "python squid_games"

This code do two things, similar to Squid Games shooting robot:
- detecting somebodies face
- if is moving, drawing target on somebodies face, but if is not moving, it's not drawing anything
"""
import cv2

# initialisation of the camera
cap = cv2.VideoCapture(0)

# download face detection filter
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# variable for saving face position
prev_face_position = None

while True:
    ret, frame = cap.read()

    # conversion of colors to grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # checking position of face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        current_face_position = (x, y, w, h)

        if prev_face_position is not None:
            # calculating difference of face positions
            delta_x = abs(prev_face_position[0] - current_face_position[0])
            delta_y = abs(prev_face_position[1] - current_face_position[1])

            # if movement were detected, draw target on a face
            if delta_x > 5 or delta_y > 5:
                # calculating center of the circle
                center = (x + w // 2, y + h // 2)
                # calculating radius of the circle
                radius = min(w, h) // 2

                # drawing circle
                cv2.circle(frame, center, radius, (0, 0, 255), 5)

                # calculating size of the cross in the circle
                cross_size = min(w, h) // 4

                # drawing vertical line
                cv2.line(frame, (center[0], center[1] - cross_size),
                         (center[0], center[1] + cross_size), (0, 0, 255), 5)

                # drawing horizontal line
                cv2.line(frame, (center[0] - cross_size, center[1]),
                         (center[0] + cross_size, center[1]), (0, 0, 255), 5)

        # update previous position
        prev_face_position = current_face_position

    # display camera view with target
    cv2.imshow('Baba Jaga Patrzy', frame)

    # ending by clicking 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close window and release resources
cap.release()
cv2.destroyAllWindows()
