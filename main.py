import mediapipe as mp
import cv2
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Load the wristwatch image
wristwatch_image = cv2.imread('wristwatch.png')

cap = cv2.VideoCapture(0)

# Get the initial size of the wristwatch image
initial_width = wristwatch_image.shape[1]
initial_height = wristwatch_image.shape[0]

# Create a resizable window
cv2.namedWindow("Wristwatch Virtual Try On", cv2.WINDOW_NORMAL)

# Define the function to calculate wrist angle
def wrist_angle(image, hand):
    # fix the point where the camera is located, the upper middle point of the screen
    a = np.array([1000, 0]) 
    # take nodes 5 (INDEX_FINGER_MCP) and 9 (MIDDLE_FINGER_MCP)
    b = np.array([hand.landmark[9].x, hand.landmark[9].z])
    c = np.array([hand.landmark[5].x, hand.landmark[5].z])

    # Radian calculation
    y1 = c[1] - b[1]
    y2 = a[1] - b[1]
    x1 = c[0] - b[0]
    x2 = a[0] - b[0]
    radians = np.arctan2(y1, x1) - np.arctan2(y2, x2)

    if radians < 0:
        radians += 2 * np.pi  # Adjust the angle to be within 0-360 degrees

    # Convert to degrees
    wrist_ang = radians * 180.0 / np.pi
    wrist_ang = round(wrist_ang, 2)

    coords = tuple(
        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x * 640 + 20, 
                  hand.landmark[mp_hands.HandLandmark.WRIST].y * 480 + 20)).astype(int))
    
    cv2.putText(image, f'Wrist Angle: {wrist_ang} degrees',
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    return image

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                continue

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make Detections with holistic model
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Right hand landmarks
            right_hand_landmarks = results.right_hand_landmarks
            if right_hand_landmarks is not None:
                # Get the wrist landmark
                wrist_landmark = right_hand_landmarks.landmark[0]

                # Calculate the depth of the wrist landmark
                wrist_depth = wrist_landmark.z
                #print(wrist_depth)

                # Calculate the scaling factor based on depth
                scale_factor = 0.1 + (abs(wrist_depth) * 2)

                # Calculate the scaled dimensions of the wristwatch image
                scaled_width = int(initial_width * scale_factor)
                scaled_height = int(initial_height * scale_factor)

                # Calculate the x, y position for overlaying the wristwatch
                x_position = int(wrist_landmark.x * frame.shape[1] - scaled_width // 2)
                y_position = int(wrist_landmark.y * frame.shape[0] - scaled_height // 2)

                # Ensure the overlay stays within the frame boundaries
                if x_position < 0:
                    x_position = 0
                if y_position < 0:
                    y_position = 0
                if x_position + scaled_width > frame.shape[1]:
                    x_position = frame.shape[1] - scaled_width
                if y_position + scaled_height > frame.shape[0]:
                    y_position = frame.shape[0] - scaled_height

                # Resize the wristwatch image to the calculated dimensions
                wristwatch_image_resized = cv2.resize(wristwatch_image, (scaled_width, scaled_height))

                # Overlay the wristwatch image onto the frame
                frame[y_position:y_position + scaled_height, x_position:x_position + scaled_width] = wristwatch_image_resized

                # Call the wrist_angle function to calculate and display wrist angle
                frame = wrist_angle(frame, right_hand_landmarks)

            cv2.imshow('Wristwatch Virtual Try On', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
