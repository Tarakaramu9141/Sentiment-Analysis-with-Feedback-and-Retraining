import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Load feedback data
feedback_data = pd.read_csv('D:/Projects_for_resume/NLP_Senimtent_Analysis/feedback/feedback_data.csv')

if len(feedback_data) == 0:
    print("No feedback data available to retrain.")
    exit()

# Load the trained model
trained_model_path = 'D:/Projects_for_resume/NLP_Senimtent_Analysis/models/trained_model.pkl'
try:
    trained_model = joblib.load(trained_model_path)
    print("Trained model loaded successfully")
except Exception as e:
    print(f"Error loading trained model: {e}")
    raise

# Prepare feedback data
feedback_data = feedback_data.rename(columns={'corrected': 'sentiment'})

# Extract X and y from feedback
X_feedback = feedback_data['text']
y_feedback = feedback_data['sentiment']

# Retrain the loaded model with feedback data
trained_model.fit(X_feedback, y_feedback)

# Specify the desired path for saving the training model
model_save_path = 'D:/Projects_for_resume/NLP_Senimtent_Analysis/models/training_model.pkl'

# Ensure the directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the updated training model
joblib.dump(trained_model, model_save_path)
print(f"Training model retrained and saved as '{model_save_path}'")