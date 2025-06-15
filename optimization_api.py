import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import emoji
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert emojis to text
    text = emoji.demojize(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Lemmatize tokens
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back to string
    return ' '.join(tokens)

def load_data_and_model(data_dir='data', model_path='models/best_model.pkl', vectorizer_path='data/vectorizer.pkl'):
    try:
        with open(f'{data_dir}/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open(f'{data_dir}/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(f'{data_dir}/X_val.pkl', 'rb') as f:
            X_val = pickle.load(f)
        with open(f'{data_dir}/y_val.pkl', 'rb') as f:
            y_val = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return X_train, y_train, X_val, y_val, model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: Could not load files from {data_dir} or {model_path}. Ensure all files exist.")
        raise e

def optimize_model(X_train, y_train, X_val, y_val):
    model = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'lbfgs']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)
    
    # Evaluate on validation set
    val_accuracy, val_report = evaluate_model(grid_search.best_estimator_, X_val, y_val, "Validation")
    
    # Save optimized model
    os.makedirs('models', exist_ok=True)
    with open('models/optimized_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    return grid_search.best_estimator_

def evaluate_model(model, X, y, set_name="Validation"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    print(f"{set_name} Accuracy:", accuracy)
    print(f"{set_name} Report:\n", report)
    return accuracy, report

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Sentiment Analysis API',
        'endpoint': '/predict',
        'method': 'POST',
        'example': {
            'text': 'I love this product!'
        },
        'response_format': {
            'sentiment': 'Positive/Neutral/Negative',
            'prediction': '1/0/-1'
        }
    })

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
    # Load data and model
    X_train, y_train, X_val, y_val, model, vectorizer = load_data_and_model()
    
    # Optimize model
    model = optimize_model(X_train, y_train, X_val, y_val)
    
    # Run Flask API
    app.run(debug=True, host='0.0.0.0', port=5000)