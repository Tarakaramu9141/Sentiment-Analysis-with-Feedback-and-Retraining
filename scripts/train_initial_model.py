import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset from Excel
data = pd.read_excel('D:/Projects_for_resume/NLP_Senimtent_Analysis/data/sentiment_data.xlsx')

# Handle NaN values in 'text' by replacing with empty strings
data['text'] = data['text'].fillna('')

X = data['text']
y = data['sentiment']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Logistic Regression
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate on test set
accuracy = model_pipeline.score(X_test, y_test)
print(f"Initial Model Accuracy: {accuracy:.4f}")

# Define the absolute path for saving the model
model_dir = 'D:/Projects_for_resume/NLP_Senimtent_Analysis/models'
os.makedirs(model_dir, exist_ok=True)  # Creates directory if it doesn't exist

# Save the trained model to the specified path
model_path = os.path.join(model_dir, 'trained_model.pkl')
joblib.dump(model_pipeline, model_path)
print(f"Trained model saved as '{model_path}'")