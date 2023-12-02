import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Load the wristwatch image
wristwatch_image = cv2.imread('watch1.png')

cap = cv2.VideoCapture(0)

# Get the initial size of the wristwatch image
initial_width = wristwatch_image.shape[1]
initial_height = wristwatch_image.shape[0]

# Define the background color to be treated as transparent
background_color = [255, 255, 255]  # Set this to the color of your background

# Create a resizable window
cv2.namedWindow("Wristwatch Virtual Try On", cv2.WINDOW_NORMAL)

def rotate_image(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Rotate the image
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

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

def wrist_orientation(hand):
    wrist_point = np.array([hand.landmark[0].x, hand.landmark[0].y])
    middle_finger_mcp_point = np.array([hand.landmark[9].x, hand.landmark[9].y])

    hand_vector = middle_finger_mcp_point - wrist_point
    reference_vector = np.array([1, 0])

    hand_vector_normalized = hand_vector / np.linalg.norm(hand_vector)
    reference_vector_normalized = reference_vector / np.linalg.norm(reference_vector)

    cross_product = np.cross(np.append(hand_vector_normalized, 0), np.append(reference_vector_normalized, 0))
    cross_product_magnitude = np.linalg.norm(cross_product)

    if cross_product_magnitude == 0:
        rotation_angle = 0
    else:
        cross_product_sign = np.sign(cross_product[2])
        rotation_angle = cross_product_sign * np.arccos(np.dot(hand_vector_normalized, reference_vector_normalized))
        rotation_angle_degrees = np.degrees(rotation_angle)

        if rotation_angle_degrees > 180:
            rotation_angle_degrees -= 360

    return float(rotation_angle_degrees)

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
            
            # right hand landmarks
            right_hand_landmarks = results.right_hand_landmarks
            if right_hand_landmarks is not None:
                # Get the wrist landmark
                wrist_landmark = right_hand_landmarks.landmark[0]

                # Calculate the depth of the wrist landmark
                wrist_depth = wrist_landmark.z

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

                # Rotate the wristwatch image based on wrist orientation
                rotated_wristwatch = rotate_image(wristwatch_image_resized, wrist_orientation(right_hand_landmarks))

                # Create a mask based on the background color
                mask = np.all(rotated_wristwatch[:, :, :3] != background_color, axis=-1)

                # Apply the mask to the frame
                frame[y_position:y_position + scaled_height, x_position:x_position + scaled_width][mask] = rotated_wristwatch[mask]

                # Call the wrist_angle function to calculate and display wrist angle
                frame = wrist_angle(frame, right_hand_landmarks)

            cv2.imshow('Wristwatch Virtual Try On', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
