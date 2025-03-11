import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label mapping for predictions
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Store the detected sentence
sentence = ""
last_predicted_character = ""

# Timer for taking input every 2 seconds
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Hand landmarks
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Check if 2 seconds have passed before taking input
        if time.time() - last_time >= 2:
            last_time = time.time()  # Reset timer

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Extract normalized x, y coordinates
                for i, landmark in enumerate(hand_landmarks.landmark):
                    if i >= 21:  # Ensure only 21 landmarks are used
                        continue
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize coordinates
                for i in range(21):  
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))

                # Bounding box for visualization
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Avoid repeating the same character consecutively
             
                sentence += predicted_character
                last_predicted_character = predicted_character

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show detected sentence
    cv2.putText(frame, "Sentence: " + sentence, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit when 'q' is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()
