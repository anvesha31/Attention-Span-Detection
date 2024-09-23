import cv2
from gazetimation import Gazetimation

# Define a function to determine the direction based on gaze coordinates

def determine_direction(gaze_direction):
    x, y = gaze_direction
    print(x, y)


# Define a custom handler function
def my_handler(data):
    if data is None:
        print("No data received")
        return

    frame, left_pupil, right_pupil, gaze_left_eye, gaze_right_eye = data

    # Calculate the average gaze direction from left and right eyes
    avg_gaze_direction = (gaze_left_eye + gaze_right_eye) / 2

    # Determine the direction based on the average gaze direction
    direction_text = determine_direction(avg_gaze_direction)

    # Draw the direction text onto the frame
    cv2.putText(frame, direction_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Gaze Estimation", frame)
    cv2.waitKey(1)


import cv2

# Create a blank OpenCV window
cv2.namedWindow("Gaze Estimation")

# Initialize Gazetimation object
gz = Gazetimation(device=0)

# Run Gazetimation with the custom handler
gz.run(handler=my_handler)
