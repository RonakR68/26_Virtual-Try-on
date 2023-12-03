import mediapipe as mp
import cv2
import numpy as np
import math

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Load the wristwatch image
img_path = 'watch1.png'
wristwatch_image = cv2.imread(img_path)

cap = cv2.VideoCapture(0)

# Get the initial size of the wristwatch image
initial_width = wristwatch_image.shape[1]
initial_height = wristwatch_image.shape[0]

# Define the background color to be treated as transparent
background_color = [255, 255, 255]

# Create a resizable window
cv2.namedWindow("Wristwatch Virtual Try On", cv2.WINDOW_NORMAL)

def rotate_image(image, angle, background_color):
    image_height, image_width = image.shape[:2]
    diagonal_square = (image_width * image_width) + (image_height * image_height)
    
    # Calculate the diagonal of the bounding box
    diagonal = round(np.sqrt(diagonal_square))
    
    # Calculate the amount of padding needed on each side
    padding_top = round((diagonal - image_height) / 2)
    padding_bottom = round((diagonal - image_height) / 2)
    padding_left = round((diagonal - image_width) / 2)
    padding_right = round((diagonal - image_width) / 2)
    
    # Apply padding to the image
    padded_image = cv2.copyMakeBorder(
        image,
        top=padding_top,
        bottom=padding_bottom,
        left=padding_left,
        right=padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color
    )
    
    # Get the new dimensions after padding
    padded_height = padded_image.shape[0]
    padded_width = padded_image.shape[1]
    
    # Calculate the center of the padded image
    center = (padded_width / 2, padded_height / 2)
    
    # Get the rotation matrix
    transform_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the padded image
    rotated_image = cv2.warpAffine(
        padded_image,
        transform_matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color
    )
    
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
    #print(wrist_ang)
    # coords = tuple(
    #     np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x * 640 + 20, 
    #               hand.landmark[mp_hands.HandLandmark.WRIST].y * 480 + 20)).astype(int))
    
    # cv2.putText(image, f'Wrist Angle: {wrist_ang} degrees',
    #             (10,80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

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

def get_direction_vector(hand, point1, point2):
    p1 = np.array([hand.landmark[point1].x, hand.landmark[point1].y])
    p2 = np.array([hand.landmark[point2].x, hand.landmark[point2].y])
    direction_vector = p2 - p1
    return direction_vector

def get_dist(hand,point1,point2):
    x1 = hand.landmark[point1].x*100
    y1 = hand.landmark[point1].y*100
    x2 = hand.landmark[point2].x*100
    y2 = hand.landmark[point2].y*100
    #print(f"({x1},{y1}), ({x2},{y2})")
    dist = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
    return dist

def get_wrist_width(frame, hand):
    # Get the coordinates of point 0 (wrist)
    wrist_x, wrist_y = int(hand.landmark[0].x * frame.shape[1]), int(hand.landmark[0].y * frame.shape[0])

    # Get the coordinates of point 1 (INDEX_FINGER_MCP)
    index_finger_mcp_x = int(hand.landmark[1].x * frame.shape[1])
    index_finger_mcp_y = int(hand.landmark[1].y * frame.shape[0])

    # Calculate the Euclidean distance between point 0 and point 1
    wrist_width = np.sqrt((index_finger_mcp_x - wrist_x)**2 + (index_finger_mcp_y - wrist_y)**2)
    wrist_width *= 2.0

    # Calculate perpendicular lines passing through point 0
    perpendicular_upward = (wrist_x, int(wrist_y - wrist_width / 2))
    perpendicular_downward = (wrist_x, int(wrist_y + wrist_width / 2))

    # Draw lines on the frame
    cv2.line(frame, (wrist_x, wrist_y), perpendicular_upward, (0, 255, 0), 2)
    cv2.line(frame, (wrist_x, wrist_y), perpendicular_downward, (0, 255, 0), 2)

    return frame,wrist_width


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

                # Get the direction vector between wrist and index finger MCP (point 5)
                direction_vector = get_direction_vector(right_hand_landmarks, 0, 5)
                #print(direction_vector)

                # Calculate the depth of the wrist landmark
                wrist_depth = wrist_landmark.z

                # Calculate the scaling factor based on depth
                scale_factor = 0.1 + (abs(wrist_depth) * 2)
                scale_factor = (scale_factor * get_dist(right_hand_landmarks,5,17)/12)
                scale_factor = max(0.1, scale_factor)
                #print(scale_factor)
                
                #wrist_width = get_wrist_width(frame, right_hand_landmarks)[1]
                # Calculate the scaled dimensions of the wristwatch image
                scaled_width = int(initial_width * scale_factor)
                #scaled_height = int(wrist_width)
                scaled_height = int(initial_height * scale_factor)
                #print(f"scaled ht: {scaled_height}")

                # Calculate the x, y position for overlaying the wristwatch
                x_position = int(wrist_landmark.x * frame.shape[1] - scaled_width // 2)
                y_position = int(wrist_landmark.y * frame.shape[0] - scaled_height // 2)

                # Adjust the x_position based on the direction vector
                x_offset = 45 
                x_position -= int((direction_vector[0] * scaled_height) + x_offset)
                #print(x_position)

                # Adjust the y_position based on the direction vector
                y_offset = 10 
                y_position += int(wrist_orientation(right_hand_landmarks) + y_offset)
                #print(y_position)

                # Ensure the overlay stays within the frame boundaries
                x_position = max(0, min(x_position, frame.shape[1] - scaled_width))
                y_position = max(0, min(y_position, frame.shape[0] - scaled_height))

                # Resize the wristwatch image to the calculated dimensions
                wristwatch_image_resized = cv2.resize(wristwatch_image, (scaled_width, scaled_height))

                # Rotate the wristwatch image based on wrist orientation
                rotation_angle_degrees = wrist_orientation(right_hand_landmarks)
                rotated_wristwatch = rotate_image(wristwatch_image_resized, rotation_angle_degrees, background_color)

                # Calculate the dimensions of the region of interest (roi)
                roi_width = min(rotated_wristwatch.shape[1], frame.shape[1] - x_position)
                roi_height = min(rotated_wristwatch.shape[0], frame.shape[0] - y_position)

                # Ensure the dimensions are valid
                if roi_width > 0 and roi_height > 0:
                    # Create a mask based on the background color
                    mask = np.all(rotated_wristwatch[:roi_height, :roi_width, :3] != background_color, axis=-1)

                    # Apply the mask to the frame
                    roi = frame[y_position:y_position + roi_height, x_position:x_position + roi_width]
                    roi[mask] = rotated_wristwatch[:roi_height, :roi_width][mask]

                    # Call the wrist_angle function to calculate and display wrist angle
                    frame = wrist_angle(frame, right_hand_landmarks)

                    #frame = get_wrist_width(frame, right_hand_landmarks)

            cv2.imshow('Wristwatch Virtual Try On', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()