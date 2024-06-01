import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('Phishing_Email.csv')

# Check the first few rows of the dataset
print(df.head())

# Rename columns for simplicity
df.columns = ['index', 'text', 'label']
df = df.drop('index', axis=1)

# Handle missing values by dropping rows with NaN values in 'text' column
df = df.dropna(subset=['text'])

# Encode labels: 1 for Phishing Email, 0 for Safe Email
df['label'] = df['label'].map({'Phishing Email': 1, 'Safe Email': 0})

# Split the dataset into features and labels
X = df['text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the email texts
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the vectorizer and model to disk
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'phishing_email_detector.pkl')
