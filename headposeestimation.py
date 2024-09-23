import cv2
import mediapipe as mp
import numpy as np
import time
time_duration = int(input("Enter the time duration in seconds\n"))
fps_array = []

# import face mesh library from mediapipe
mp_face_mesh = mp.solutions.face_mesh

# initialize the face mesh model with confidence levels of 0.5
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# import drawing assets to display on screen
mp_drawing = mp.solutions.drawing_utils
# specify thickness of lines and circle radius drawn on the screen.
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# initialize video capturing with device id 0 (default camera)
cap = cv2.VideoCapture(0)
# variables to keep track of attention span and time elapsed
forward_count = 0
forward_duration = 0
start_time = time.time()

# while camera is open,
while cap.isOpened():
    # capture frames from camera
    success, image = cap.read()
    start = time.time()
    #flip image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # prevent accidental modification of image frame (write = false)
    image.flags.writeable = False
    # extract face mesh from image
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # height, width and number of channels
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    # condition to check for multiple faces
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # iterate over face landmarks
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        # extract coordinates of nose by scaling normalized lm values (multiplying by image dimensions)
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    # x and y coordinates of nose
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # 2d coordinates
                    face_2d.append([x, y])

                    # 3d coordinates
                    face_3d.append([x, y, lm.z])
            # store landmarks of face
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            # focal length is related to horizontal fov, to simplify calculations we take img_w
            focal_length = 1 * img_w
            # camera matrix to use for perspective and point calculations
            #principle point (center of camera)
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            # distortion matrix initialized by zeros (to indicate no distortion)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            # estimate the pose(rotation and translation of face)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            # rotation matrix and jacobian matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Decompose matrix into 3 euler angles.
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
            # display the lines on screen
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

    with open('max_frequency.txt', "r") as file:
        max_frequency = file.readline()
        max_frequency = float(max_frequency)

    # Check if 15 seconds have passed
    if time.time() - start_time >= time_duration:
        total_time = (time.time() - start_time)  # Calculate total frames processed
        current_frequency = forward_count / time_duration

        if current_frequency > max_frequency:
            max_frequency = current_frequency
            with open('max_frequency.txt', 'w') as file2:
                file2.write(str(max_frequency))
        print("Frequency of looking forward:", current_frequency)
        print("Estimated Attention Span:", current_frequency/max_frequency * 100)
        break

cap.release()
