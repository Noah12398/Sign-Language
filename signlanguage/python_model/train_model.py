import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("americanSignLanguage.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def landmarks_to_image(landmarks):
    """Convert 21 hand landmarks into a 64x64 grayscale image."""
    if landmarks is not None:
        img = np.zeros((64, 64), dtype=np.uint8)  # Grayscale image

        # Normalize landmarks to 64x64
        landmarks = np.array(landmarks).reshape(21, 3)
        x_vals = (landmarks[:, 0] * 64).astype(int)
        y_vals = (landmarks[:, 1] * 64).astype(int)

        for x, y in zip(x_vals, y_vals):
            if 0 <= x < 64 and 0 <= y < 64:
                img[y, x] = 255  # Draw white dots for landmarks

        return img
    return None


cap = cv2.VideoCapture(0)  # Open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)  # Detect hands

    # Extract hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])  # Extract x, y, z

            if len(landmarks) == 63:  # Ensure correct shape
                hand_image = landmarks_to_image(landmarks)  # Convert to image

                if hand_image is not None:
                    hand_image = np.expand_dims(hand_image, axis=-1)  # Add channel dimension → (64, 64, 1)
                    input_data = np.expand_dims(hand_image, axis=0)   # Add batch dimension → (1, 64, 64, 1)
                    input_data = input_data / 255.0  # Normalize

                    # Reshape to match the model's expected input shape
                    input_data = input_data.reshape(1, 64, 64, 1)
                    print("Shape before resize:", input_data.shape)

                    input_data = cv2.resize(input_data[0, :, :, 0], (28, 28))  # Extract the 2D image first
                    input_data = np.expand_dims(input_data, axis=-1)  # Add channel dimension
                    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

                    # Predict ASL digit
                    prediction = model.predict(input_data)
                    predicted_label = np.argmax(prediction)

                    # Display prediction
                    cv2.putText(frame, f"Predicted: {predicted_label}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show output
    cv2.imshow("ASL Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
