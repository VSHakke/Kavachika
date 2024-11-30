import cv2
import time
import mediapipe as mp
import numpy as np
from mtcnn import MTCNN
from Person_Detection import detect_person
from Centroid_Tracker import CentroidTracker
from SOS_Condition import is_female_surrounded
from Telebot_alert import send_telegram_alert  # Ensure this import is correct
from facial_expression import classify_face, draw_selected_landmarks
from gesture_detection import classify_gesture
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv

# Load gender detection model
gender_model = load_model('gender_detection.h5')
gender_classes = ['man', 'woman']

# Initialize video capture and tracker
webcam = cv2.VideoCapture(0)
tracker = CentroidTracker()
# Multi-task Cascaded Convolutional Networks. It is a popular deep learning framework used for face detection and alignment in images
detector = MTCNN()

mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

if not webcam.isOpened():
    print("Could not open video")
    exit()

# Create a named window and resize it
cv2.namedWindow("Webcam/Video Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam/Video Feed", 1280, 720)

# Initialize UI Colors and Fonts
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_COLOR_HIGH = (0, 0, 255)  # Red for high alerts
ALERT_COLOR_MEDIUM = (0, 165, 255)  # Orange for medium alerts
ALERT_COLOR_LOW = (0, 255, 0)  # Green for low alerts
TEXT_COLOR = (255, 255, 255)  # White for normal text
BOX_BG_COLOR = (0, 0, 0)  # Black background for text box
SIDEBAR_BG_COLOR = (50, 50, 50)  # Dark gray for sidebar
SIDEBAR_TEXT_COLOR = (0, 255, 0)  # Green for normal text

# Sidebar Configuration
SIDEBAR_HEIGHT = 100  # Increased sidebar height for better visibility
TEXT_BOX_PADDING = 5  # Padding around the text in the text box
TEXT_SCALE = 0.6  # Text scale
TEXT_THICKNESS = 2  # Text thickness

try:
    skip_frame = 7
    frame_count = -1
    alert_message = None
    alert_priority = None

    while True:
        status, frame = webcam.read()
        if not status:
            print("Failed to read frame from video")
            break

        frame_count += 1
        if frame_count % skip_frame != 0:
            continue

        # Detect persons in the frame
        person_boxes = detect_person(frame)
        n = len(person_boxes)

        # Reset gender counts for the current frame
        male_count = 0
        female_count = 0

        # Update tracker with detected person bounding boxes
        objects = tracker.update(person_boxes)

        for i, (objectID, centroid) in enumerate(objects.items()):
            if objectID < len(person_boxes):
                x1, y1, x2, y2 = map(int, person_boxes[i])
                person_img = frame[y1:y2, x1:x2]

                # Detect faces within the person bounding box using MTCNN
                faces = detector.detect_faces(person_img)

                if faces:
                    face = faces[0]
                    x, y, width, height = face['box']
                    face_img = person_img[y:y + height, x:x + width]

                    # Detect and classify facial expression
                    results = mp_holistic.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                    if results.face_landmarks:
                        face_class = classify_face(results.face_landmarks)

                        # New gender detection logic using the TensorFlow model
                        face_crop = np.copy(person_img[y:y + height, x:x + width])

                        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                            continue

                        # Preprocess face for gender detection
                        face_crop = cv2.resize(face_crop, (96, 96))
                        face_crop = face_crop.astype("float") / 255.0
                        face_crop = img_to_array(face_crop)
                        face_crop = np.expand_dims(face_crop, axis=0)

                        # Perform gender detection
                        conf = gender_model.predict(face_crop)[0]
                        gender_label_idx = np.argmax(conf)
                        gender_label = gender_classes[gender_label_idx]

                        # Perform gesture detection
                        gesture_label = classify_gesture(frame, results)
                        # gender_label = 'woman'
                        # Update gender counts
                        if gender_label == 'man':
                            male_count += 1
                        elif gender_label == 'woman':
                            female_count += 1

                            # SOS gesture detection
                            if gesture_label in ["Gesture 1", "Gesture 2- Person signaling vulnerability.", "Gesture 3- Person is in danger.", "Gesture 4", "Gesture 5", "Gesture 6", "Gesture 7"]:
                                alert_message = f"{gesture_label} detected!"
                                alert_priority = "high"
                                send_telegram_alert(frame, alert_message)  # Send alert to Telegram

                            # Alert condition: Female detected alone at night
                            if n == 1 and gender_label == 'woman' and (time.localtime().tm_hour >= 18 or time.localtime().tm_hour < 6):
                                alert_message = f"Female detected alone at night! Gesture: {gesture_label}"
                                alert_priority = "high"
                                send_telegram_alert(frame, alert_message)  # Send alert to Telegram
                        # Combine all detected features into the label
                        label = f'ID {objectID}: {gender_label}, {face_class}, {gesture_label}'

                        # Draw facial landmarks
                        draw_selected_landmarks(face_img, results.face_landmarks)

                        # Draw a small text box for the label
                        text_size = cv2.getTextSize(label, FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
                        text_x = x1
                        text_y = y1 - text_size[1] - TEXT_BOX_PADDING

                        # Create the background rectangle for the text box
                        cv2.rectangle(frame,
                                      (text_x - TEXT_BOX_PADDING, text_y - TEXT_BOX_PADDING),
                                      (text_x + text_size[0] + TEXT_BOX_PADDING, text_y + text_size[1] + TEXT_BOX_PADDING),
                                      BOX_BG_COLOR, -1)

                        # Put the label text in the text box
                        cv2.putText(frame, label, (text_x, text_y + text_size[1]), FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

                    else:
                        label = f'ID {objectID}: Unknown'
                else:
                    print(f"Warning: No face detected for object ID {objectID}")

        # Create the sidebar at the bottom of the frame
        sidebar = np.zeros((SIDEBAR_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
        sidebar[:] = SIDEBAR_BG_COLOR

        # Sidebar content: Information like gender counts, total persons, etc.
        sidebar_text = f'Males: {male_count}  Females: {female_count}  Total Persons: {n}'
        cv2.putText(sidebar, sidebar_text, (15, 30), FONT, 0.8, SIDEBAR_TEXT_COLOR, 2)

        # Show alerts in the sidebar with different colors based on priority
        if alert_message:
            alert_text = f"ALERT: {alert_message}"
            if alert_priority == "high":
                cv2.putText(sidebar, alert_text, (20, 70), FONT, 0.8, ALERT_COLOR_HIGH, 2)
            elif alert_priority == "medium":
                cv2.putText(sidebar, alert_text, (20, 70), FONT, 0.8, ALERT_COLOR_MEDIUM, 2)
            else:
                cv2.putText(sidebar, alert_text, (20, 70), FONT, 0.8, ALERT_COLOR_LOW, 2)

        # Draw a border around the frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (255, 255, 255), 3)

        # Combine the original frame and the sidebar vertically
        combined_frame = cv2.vconcat([frame, sidebar])

        # Display the final frame with the sidebar at the bottom
        cv2.imshow("Webcam/Video Feed", combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()

