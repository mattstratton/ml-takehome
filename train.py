import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the training dataset from CSV (TODO: make this an argument at the CLI)
df = pd.read_csv('import.csv')

# Split data into features and labels (the message column of the CSV is the feature with the label column)
X = df['message']
y = df['label']

# Split data into training and testing sets
# test_size means to make 20% for testing and 80% for training
# random_state is just the seed for the RNG
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for TF-IDF vectorization and logistic regression
# TfidVectorizer transforms the text into TF-IDF features
# Tried different algorithims; LR seemed most accurate?
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model for later use
joblib.dump(pipeline, 'meeting_request_classifier.pkl')
print("Model training completed and saved.")
