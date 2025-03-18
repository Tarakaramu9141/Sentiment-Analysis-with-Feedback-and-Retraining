from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model from the specified path
trained_model_path = 'D:/Projects_for_resume/NLP_Senimtent_Analysis/models/trained_model.pkl' #Added explicit path
try:
    trained_model = joblib.load(trained_model_path)
    print("Trained model loaded successfully from", trained_model_path)
except Exception as e:
    print(f"Error loading trained model: {e}")
    raise

# Load or initialize the training model
training_model_path = 'D:/Projects_for_resume/NLP_Senimtent_Analysis/models/training_model.pkl'
if os.path.exists(training_model_path):
    training_model = joblib.load(training_model_path)
    print("Training model loaded from file")
else:
    training_model = trained_model
    joblib.dump(training_model, training_model_path)
    print("Training model initialized and saved")

# Feedback file
feedback_dir = 'D:/Projects_for_resume/NLP_Senimtent_Analysis/feedback'
feedback_file = os.path.join(feedback_dir, 'feedback_data.csv')
os.makedirs(feedback_dir, exist_ok=True)
if not os.path.exists(feedback_file):
    # Initialize with dummy data to ensure two classes (optional)
    initial_feedback = pd.DataFrame([
        {'text': 'dummy positive', 'predicted': 'Positive', 'corrected': 'Positive', 'category': 'Test'},
        {'text': 'dummy negative', 'predicted': 'Negative', 'corrected': 'Negative', 'category': 'Test'}
    ])
    initial_feedback.to_csv(feedback_file, index=False)
    print("Feedback file created with initial dummy data")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text:
            prediction = trained_model.predict([text])[0]
            print(f"Prediction for '{text}': {prediction}")
            return redirect(url_for('feedback', text=text, prediction=prediction))
    return render_template('index.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    text = request.args.get('text', '')
    predicted = request.args.get('prediction', '')

    if request.method == 'POST':
        corrected = request.form.get('corrected', '')
        category = request.form.get('category', '')

        # Append feedback to CSV
        feedback_df = pd.read_csv(feedback_file)
        new_feedback = pd.DataFrame([{
            'text': text,
            'predicted': predicted,
            'corrected': corrected,
            'category': category
        }])
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        feedback_df.to_csv(feedback_file, index=False)

        # Update training model only if there are at least 2 classes
        global training_model
        X_feedback = feedback_df['text']
        y_feedback = feedback_df['corrected']
        if len(y_feedback) > 0 and len(y_feedback.unique()) >= 2:  # Check for at least 2 classes
            training_model.fit(X_feedback, y_feedback)
            joblib.dump(training_model, training_model_path)
            print(f"Training model updated with {len(X_feedback)} feedback entries")
        else:
            print(f"Skipping model update: only {len(y_feedback.unique())} class(es) in feedback")

        return redirect(url_for('home', message='Feedback received! Model will update when both classes are present.'))

    return render_template('feedback.html', text=text, predicted=predicted)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)