from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__, static_folder='statics')

# URL of your Flask API for making predictions
api_url = 'http://0.0.0.0:4000/predict'

# Load the TensorFlow model and scaler
model = tf.keras.models.load_model('wine_model.keras')
scaler = joblib.load('scaler.pkl')

# Wine class labels
class_labels = ['Class 0 (Cultivar 1)', 'Class 1 (Cultivar 2)', 'Class 2 (Cultivar 3)']

# Feature names for the wine dataset
feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280_od315', 'proline'
]

@app.route('/')
def home():
    return "Welcome to the Wine Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            
            # Extract all 13 features
            features = []
            for feature in feature_names:
                features.append(float(data[feature]))
            
            # Prepare input data
            input_data = np.array(features)[np.newaxis, ]
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Perform the prediction
            prediction = model.predict(input_data_scaled)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = class_labels[predicted_class_idx]
            confidence = float(prediction[0][predicted_class_idx]) * 100
            
            # Return the predicted class and confidence
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "probabilities": {
                    class_labels[i]: f"{float(prediction[0][i]) * 100:.2f}%"
                    for i in range(len(class_labels))
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)})
    
    elif request.method == 'GET':
        return render_template('predict_wine.html')
    
    else:
        return "Unsupported HTTP method"

@app.route('/info')
def info():
    """Endpoint to get information about the wine dataset"""
    return jsonify({
        "dataset": "Wine Dataset",
        "features": feature_names,
        "classes": class_labels,
        "num_features": 13,
        "num_classes": 3,
        "description": "Wine recognition dataset containing chemical analysis of wines from three different cultivars in Italy"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)