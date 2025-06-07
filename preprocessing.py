import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import emoji
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = emoji.demojize(text)  # Convert emojis to text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_clean_data(file_path):
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please ensure it exists in the project directory.")
        return None
    
    # Apply preprocessing to text column
    df['text'] = df['text'].apply(preprocess_text)
    
    # Remove empty or invalid rows
    df = df.dropna(subset=['text', 'sentiment'])
    df = df[df['text'].str.strip() != '']
    
    return df

def prepare_data(df, output_dir='data'):
    if df is None:
        print("Error: No data to process. Exiting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features and labels
    X = df['text']
    y = df['sentiment'].astype(int)
    
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data into train (70%), validation (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_vectorized, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Save processed data and vectorizer
    try:
        with open(f'{output_dir}/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(f'{output_dir}/X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(f'{output_dir}/X_val.pkl', 'wb') as f:
            pickle.dump(X_val, f)
        with open(f'{output_dir}/X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(f'{output_dir}/y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(f'{output_dir}/y_val.pkl', 'wb') as f:
            pickle.dump(y_val, f)
        with open(f'{output_dir}/y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
        
        print(f"Data preprocessing complete. Saved vectorizer and split datasets to {output_dir}/")
    except Exception as e:
        print(f"Error saving files: {e}")

if __name__ == "__main__":
    # Load and preprocess the dataset
    df = load_and_clean_data('social_media_posts_10k.csv')
    prepare_data(df)