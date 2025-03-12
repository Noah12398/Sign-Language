import os
from flask import Flask, jsonify, request
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model
try:
    model_dict = pickle.load(open('signlanguage/python_model/model.p', 'rb'))
    print(f"Current working directory: {os.getcwd()}")
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()

# Initialize Mediapipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Label mapping for predictions
labels_dict = {i: chr(65 + i) for i in range(24)}  # A-Y (Skipping J, Z by default)
labels_dict[24] = 'J'
labels_dict[25] = 'Z'

@app.route('/detect_sign_from_image', methods=['POST'])
def detect_sign_from_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected'}), 400

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux, x_, y_ = [], [], []

            for i, landmark in enumerate(hand_landmarks.landmark):
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            for i in range(21):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), '?')
            confidence = float(np.max(model.predict_proba([np.asarray(data_aux)]))) if hasattr(model, 'predict_proba') else 0.0
            
            return jsonify({'character': predicted_character, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500

@app.route('/detect_sign', methods=['GET'])
def detect_sign():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open webcam'}), 500

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected'}), 400

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux, x_, y_ = [], [], []

            for i, landmark in enumerate(hand_landmarks.landmark):
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            for i in range(21):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), '?')
            
            return jsonify({'character': predicted_character})
    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Starting Enhanced Sign Language Detection Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
