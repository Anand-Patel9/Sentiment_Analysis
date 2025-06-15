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

if __name__ == "__main__":
    # Ensure required files exist
    required_files = [
        'models/optimized_model.pkl',
        'data/vectorizer.pkl',
        'web_interface.html'
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found.")
            exit(1)
    
    # Load model and vectorizer
    global model, vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    # Print registered routes for debugging
    print("Registered routes:", [rule.rule for rule in app.url_map.iter_rules()])
    
    # Run Flask app
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)