import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Define FACE_CONNECTIONS based on landmark indices
FACE_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), 
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), 
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), 
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), 
    (103, 67), (67, 109), (109, 10)
]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extracting eye landmarks
            left_eye_landmarks = np.array([[lm.x * img_w, lm.y * img_h] for lm in face_landmarks.landmark[33:263]])
            right_eye_landmarks = np.array([[lm.x * img_w, lm.y * img_h] for lm in face_landmarks.landmark[130:362]])

            # Calculating eye center
            left_eye_center = np.mean(left_eye_landmarks, axis=0).astype(int)
            right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(int)

            # Calculating gaze vector
            gaze_vector = right_eye_center - left_eye_center
            gaze_angle = np.arctan2(gaze_vector[1], gaze_vector[0]) * 180 / np.pi

            # Display gaze direction
            if gaze_angle < -10:
                text = "Looking left"
            elif gaze_angle > 10:
                text = "Looking right"
            else:
                text = "Forward"

        # Drawing eye center
        cv2.circle(image, tuple(left_eye_center), 5, (0, 255, 0), -1)
        cv2.circle(image, tuple(right_eye_center), 5, (0, 255, 0), -1)
        # Display text on image
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=FACE_CONNECTIONS,  # Use the defined FACE_CONNECTIONS
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('Gaze and Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
