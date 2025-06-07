import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import os

def load_data(data_dir='data'):
    try:
        with open(f'{data_dir}/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open(f'{data_dir}/X_val.pkl', 'rb') as f:
            X_val = pickle.load(f)
        with open(f'{data_dir}/X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(f'{data_dir}/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(f'{data_dir}/y_val.pkl', 'rb') as f:
            y_val = pickle.load(f)
        with open(f'{data_dir}/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
        return X_train, X_val, X_test, y_train, y_val, y_test
    except FileNotFoundError as e:
        print(f"Error: Could not load data from {data_dir}. Ensure all pickle files exist.")
        raise e

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, X_val, y_val):
    model = LinearSVC(max_iter=1000)
    param_grid = {'C': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best SVM Parameters:", grid_search.best_params_)
    print("Best SVM Validation Score:", grid_search.best_score_)
    return grid_search.best_estimator_

def evaluate_model(model, X, y, set_name="Test"):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    print(f"{set_name} Accuracy:", accuracy)
    print(f"{set_name} Report:\n", report)
    return accuracy, report

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)
    svm_model = train_svm(X_train, y_train, X_val, y_val)
    
    # Evaluate models on validation set
    print("\nValidation Set Results:")
    lr_val_accuracy, lr_val_report = evaluate_model(lr_model, X_val, y_val, "Logistic Regression Validation")
    nb_val_accuracy, nb_val_report = evaluate_model(nb_model, X_val, y_val, "Naive Bayes Validation")
    svm_val_accuracy, svm_val_report = evaluate_model(svm_model, X_val, y_val, "SVM Validation")
    
    # Evaluate models on test set
    print("\nTest Set Results:")
    lr_test_accuracy, lr_test_report = evaluate_model(lr_model, X_test, y_test, "Logistic Regression Test")
    nb_test_accuracy, nb_test_report = evaluate_model(nb_model, X_test, y_test, "Naive Bayes Test")
    svm_test_accuracy, svm_test_report = evaluate_model(svm_model, X_test, y_test, "SVM Test")
    
    # Select the best model based on validation accuracy
    models = {
        'Logistic Regression': (lr_model, lr_val_accuracy),
        'Naive Bayes': (nb_model, nb_val_accuracy),
        'SVM': (svm_model, svm_val_accuracy)
    }
    best_model_name = max(models, key=lambda x: models[x][1])
    best_model, best_val_accuracy = models[best_model_name]
    
    # Save the best model
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    scores = cross_val_score(lr_model, X_train, y_train, cv=5)
    print("Cross-validation scores:", scores, "Mean:", scores.mean())
    print(f"Best model ({best_model_name}, Validation Accuracy: {best_val_accuracy:.2f}) saved to models/best_model.pkl")