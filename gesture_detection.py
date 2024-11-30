import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
# "E:\PLACEMENT\Project Section\Kavachika"
file_path = 'E:/PLACEMENT/Project Section/Kavachika/hand_landmarks.csv'
hand_landmarks = pd.read_csv(file_path)

# Filter relevant columns (hand landmarks and the gesture 'sign')
landmark_columns = [col for col in hand_landmarks.columns if 'hand_mark' in col]
data = hand_landmarks[landmark_columns + ['sign']]

# Clean dataset (remove NaN values)
data = data.dropna()

# Feature Engineering: Compute Euclidean distances between specific landmarks
def euclidean_distance(landmarks, index1, index2):
    return np.sqrt((landmarks[index1 * 3] - landmarks[index2 * 3])**2 + 
                   (landmarks[index1 * 3 + 1] - landmarks[index2 * 3 + 1])**2 + 
                   (landmarks[index1 * 3 + 2] - landmarks[index2 * 3 + 2])**2)

# Add distances between specific landmarks
data['distance_finger_08_12'] = data.apply(lambda row: euclidean_distance(row[landmark_columns].values, 8, 12), axis=1)
data['distance_finger_04_08'] = data.apply(lambda row: euclidean_distance(row[landmark_columns].values, 4, 8), axis=1)

# Prepare features and labels
X = data.drop('sign', axis=1)
y = data['sign']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'sos_gesture_model.pkl')

# Step 4: Test the model's accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Step 5: Plot the confusion matrix with accuracy
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.text(0.5, -0.1, f"Accuracy: {accuracy:.2f}", fontsize=12, ha='center', transform=plt.gca().transAxes)
plt.show()

# Step 6: Real-time gesture detection (optional, if you want to test with live video)
mp_hands = mp.solutions.hands

def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks).flatten()

def classify_gesture(frame, results):
    model = joblib.load('sos_gesture_model.pkl')
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = extract_landmarks(hand_landmarks)
                distance_finger_08_12 = euclidean_distance(landmarks, 8, 12)
                distance_finger_04_08 = euclidean_distance(landmarks, 4, 8)
                features = np.append(landmarks, [distance_finger_08_12, distance_finger_04_08])
                prediction = model.predict([features])

                gesture_mapping = {
                    1: "Gesture 1",
                    2: "Gesture 2 - Person signaling vulnerability.",
                    3: "Gesture 3 - Person is in danger.",
                    4: "Gesture 4",
                    5: "Gesture 5",
                    6: "Gesture 6",
                    7: "Gesture 7",
                }
                gesture_name = gesture_mapping.get(prediction[0], "Unknown Gesture")
                print("Gesture Detected:", gesture_name)
                return gesture_name

    return "No Gesture Detected"
