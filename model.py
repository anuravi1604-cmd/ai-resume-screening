import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def train_model():
    """
    Train the TF-IDF vectorizer and the Logistic Regression classifier.
    Saves outputs via joblib.
    """
    if not os.path.exists("Resume.csv"):
        print("Dataset Resume.csv not found.")
        return
        
    print("Loading dataset...")
    df = pd.read_csv("Resume.csv")
    
    # Drop any nulls if present
    df = df.dropna(subset=['Resume_str', 'Category'])
    
    print("Cleaning text...")
    X_train = df["Resume_str"].apply(clean_text)
    y_train = df["Category"]
    
    print("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    print("Training Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    print("Saving models...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)
    print("Training complete.")

def load_models():
    """
    Load the trained model and vectorizer. Train first if they don't exist.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Models not found. Training model...")
        train_model()
        
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    
    return vectorizer, model

def predict_role(text):
    """
    Predict the job role for a given resume text.
    """
    vectorizer, model = load_models()
    
    # Clean the input text
    cleaned = clean_text(text)
    
    # Vectorize input text
    text_vec = vectorizer.transform([cleaned])
    
    # Predict probabilities to add a confidence score
    probs = model.predict_proba(text_vec)[0]
    best_idx = probs.argmax()
    best_class = model.classes_[best_idx]
    confidence = probs[best_idx]
    
    return best_class, confidence

if __name__ == "__main__":
    train_model()
