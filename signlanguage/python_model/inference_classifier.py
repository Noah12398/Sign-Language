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
    model_dict = pickle.load(open('./signlanguage/python_model/model.p', 'rb'))    
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()

# Initialize Mediapipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Label mapping for predictions
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Add J and Z that might be missing
if 24 not in labels_dict:
    labels_dict[24] = 'J'
if 25 not in labels_dict:
    labels_dict[25] = 'Z'

@app.route('/detect_sign_from_image', methods=['POST'])
def detect_sign_from_image():
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        
        # Convert to opencv format
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        
        # Process with mediapipe
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected'}), 400
        
        # Process hand landmarks
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
            
            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            # Optional: Add confidence score if model provides it
            confidence = 0.0
            try:
                confidence = float(np.max(model.predict_proba([np.asarray(data_aux)])))
            except:
                pass
            
            # For debug purposes, save the processed image
            debug_image = frame.copy()
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_image, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Save for debugging (optional)
            cv2.imwrite('debug_image.jpg', debug_image)
            
            return jsonify({
                'character': predicted_character,
                'confidence': confidence
            })
        
    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500

# Keep the original endpoint for compatibility
@app.route('/detect_sign', methods=['GET'])
def detect_sign():
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open webcam'}), 500
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        
        # Process with mediapipe
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected'}), 400
        
        # Process hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            
            # Extract normalized x, y coordinates
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i >= 21:
                    continue
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            # Normalize coordinates
            for i in range(21):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            
            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            return jsonify({'character': predicted_character})
        
    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Starting Enhanced Sign Language Detection Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)