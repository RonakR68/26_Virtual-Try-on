from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

wristwatch_image = cv2.imread('watch1.png')
initial_width = wristwatch_image.shape[1]
initial_height = wristwatch_image.shape[0]
background_color = [255, 255, 255]

cap = cv2.VideoCapture(0)

# Set the cap object in the app.config
app.config['cap'] = cap

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
        value=background_color  # Set this to the color of your background
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
        borderValue=background_color  # Set this to the color of your background
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

    coords = tuple(
        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x * 640 + 20, 
                  hand.landmark[mp_hands.HandLandmark.WRIST].y * 480 + 20)).astype(int))
    
    #cv2.putText(image, f'Wrist Angle: {wrist_ang} degrees',
               # (10,80),
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

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

def generate_frames(cap):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                right_hand_landmarks = results.right_hand_landmarks
                if right_hand_landmarks is not None:
                    wrist_landmark = right_hand_landmarks.landmark[0]
                    direction_vector = get_direction_vector(right_hand_landmarks, 0, 5)
                    wrist_depth = wrist_landmark.z
                    scale_factor = 0.1 + (abs(wrist_depth) * 2)
                    scaled_width = int(initial_width * scale_factor)
                    scaled_height = int(initial_height * scale_factor)

                    x_position = int(wrist_landmark.x * frame.shape[1] - scaled_width // 2)
                    y_position = int(wrist_landmark.y * frame.shape[0] - scaled_height // 2)
                    x_offset = 45
                    x_position -= int((direction_vector[0] * scaled_height) + x_offset)
                    y_offset = 10
                    y_position += int(wrist_orientation(right_hand_landmarks) + y_offset)

                    x_position = max(0, min(x_position, frame.shape[1] - scaled_width))
                    y_position = max(0, min(y_position, frame.shape[0] - scaled_height))

                    wristwatch_image_resized = cv2.resize(wristwatch_image, (scaled_width, scaled_height))
                    rotation_angle_degrees = wrist_orientation(right_hand_landmarks)
                    rotated_wristwatch = rotate_image(wristwatch_image_resized, rotation_angle_degrees, background_color)

                    roi_width = min(rotated_wristwatch.shape[1], frame.shape[1] - x_position)
                    roi_height = min(rotated_wristwatch.shape[0], frame.shape[0] - y_position)

                    if roi_width > 0 and roi_height > 0:
                        mask = np.all(rotated_wristwatch[:roi_height, :roi_width, :3] != background_color, axis=-1)
                        roi = frame[y_position:y_position + roi_height, x_position:x_position + roi_width]
                        roi[mask] = rotated_wristwatch[:roi_height, :roi_width][mask]

                        frame = wrist_angle(frame, right_hand_landmarks)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('indexst.html')

@app.route('/video_feed')
def video_feed():
    cap = app.config['cap']
    return Response(generate_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    #cap = cv2.VideoCapture(0)

    # Set the cap object in the app.config
    app.config['cap'] = cap

    app.run(debug=True, threaded=True, use_reloader=False, port=5000, passthrough_errors=False)
