from flask import Flask, send_file, jsonify
from optimization_api import preprocess_text, app
import pickle
import os

def load_model_and_vectorizer(model_path='models/optimized_model.pkl', vectorizer_path='data/vectorizer.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: Could not load {model_path} or {vectorizer_path}. Ensure files exist.")
        raise e

@app.route('/')
def serve_interface():
    try:
        return send_file('web_interface.html')
    except FileNotFoundError:
        return jsonify({'error': 'Web interface file not found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess input text
        processed_text = preprocess_text(text)
        X = vectorizer.transform([processed_text])
        
        # Predict sentiment
        prediction = model.predict(X)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative' if prediction == -1 else 'Neutral'
        
        return jsonify({'sentiment': sentiment, 'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Ensure required files exist
    if not os.path.exists('models/optimized_model.pkl') or not os.path.exists('data/vectorizer.pkl') or not os.path.exists('week5_web_interface.html'):
        print("Error: Required files (optimized_model.pkl, vectorizer.pkl, or week5_web_interface.html) not found.")
        exit(1)
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)