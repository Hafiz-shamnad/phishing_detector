import joblib
import os

# Define the paths to the model and vectorizer files
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'phishing_email_detector.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

# Check if the model and vectorizer files exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found. Please ensure the training script has been run and the files are in the correct location.")

# Load the trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_email(email_text):
    """
    Predicts whether the given email text is a phishing email or not.
    
    Args:
    email_text (str): The email text to be classified.
    
    Returns:
    str: 'Phishing Email' if the email is predicted to be a phishing email,
         'Safe Email' otherwise.
    """
    # Transform the email text using the loaded vectorizer
    email_text_vectorized = vectorizer.transform([email_text])
    
    # Predict using the loaded model
    prediction = model.predict(email_text_vectorized)
    
    # Return the corresponding label
    if prediction[0] == 1:
        return 'Phishing Email'
    else:
        return 'Safe Email'
