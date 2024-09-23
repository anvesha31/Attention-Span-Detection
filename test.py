import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
time_duration = int(input("Enter the time duration in seconds\n"))
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # Return the eye aspect ratio
    return ear

# Constants for drowsiness detection
EAR_THRESHOLD = 0.2  # Adjust this threshold based on your environment
EAR_CONSEC_FRAMES = 20  # Number of consecutive frames the eye must be below threshold to trigger drowsiness
COUNTER = 0
ALARM_ON = False

# Your existing code continues from here
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

forward_count = 0
forward_duration = 0
start_time = time.time()

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = []
            right_eye = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # 2d coordinates
                    face_2d.append([x, y])

                    # 3d coordinates
                    face_3d.append([x, y, lm.z])

                if idx in [159, 145, 33, 133, 153, 144, 362, 374, 386, 263, 362, 384, 385, 386, 387]:
                    left_eye.append([int(lm.x * img_w), int(lm.y * img_h)])
                if idx in [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158]:
                    right_eye.append([int(lm.x * img_w), int(lm.y * img_h)])

            # Calculate eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            # Average eye aspect ratio
            ear = (left_ear + right_ear) / 2.0

            # Check if the eye aspect ratio is below the threshold
            if ear < EAR_THRESHOLD:
                COUNTER += 1
                # If the eyes have been closed for a sufficient number of consecutive frames,
                # trigger the alarm
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        print("Drowsiness detected!")
                        # Add your alerting mechanism here (e.g., sound alarm)
            else:
                COUNTER = 0
                ALARM_ON = False

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # see where the user's head is tilting
            if y < -10:
                text = "Looking left"
                forward_count += 0
                forward_duration = 0
            elif y > 10:
                text = "Looking right"
                forward_count += 0
                forward_duration = 0
            elif x < -10:
                text = "Looking Down"
                forward_count += 0
                forward_duration = 0
            elif x > 10:
                text = "Looking UP"
                forward_count += 0
                forward_duration = 0
            else:
                text = "forward"
                forward_count += 1
                forward_duration += time.time() - start_time

            # display nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # text on image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('head pose estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    # Check if the specified duration has passed
    if time.time() - start_time >= time_duration:
        total_time = (time.time() - start_time)  # Calculate total frames processed
        print(total_time)
        print("Forward duration:", forward_duration)
        current_frequency = forward_count / time_duration
        with open('max_frequency.txt', "r") as file:
            max_frequency = file.readline()
            max_frequency = float(max_frequency)

        if current_frequency > max_frequency:
            max_frequency = current_frequency
            with open('max_frequency.txt', 'w') as file2:
                file2.write(str(max_frequency))
        print("Frequency of looking forward:", current_frequency)
        print("Estimated Attention Span:", current_frequency/max_frequency * 100)
        break

cap.release()
# full attention span value of 1 second is 28.0 (looking forward continuously)
