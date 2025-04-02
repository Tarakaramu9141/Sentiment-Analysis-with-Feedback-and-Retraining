from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Configuration
TRAINED_MODEL_PATH = 'D:/Projects_for_resume/NLP_Sentiment_Analysis/models/trained_model.pkl'
TRAINING_MODEL_PATH = 'D:/Projects_for_resume/NLP_Sentiment_Analysis/models/training_model.pkl'
FEEDBACK_DIR = 'D:/Projects_for_resume/NLP_Sentiment_Analysis/feedback'
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'feedback_data.csv')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'pdf'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models with error handling
try:
    trained_model = joblib.load(TRAINED_MODEL_PATH)
    print("Trained model loaded successfully from", TRAINED_MODEL_PATH)
except Exception as e:
    print(f"Error loading trained model: {e}")
    raise

if os.path.exists(TRAINING_MODEL_PATH):
    try:
        training_model = joblib.load(TRAINING_MODEL_PATH)
        print("Training model loaded from file")
    except Exception as e:
        print(f"Error loading training model, creating new one: {e}")
        training_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
        ])
        joblib.dump(training_model, TRAINING_MODEL_PATH)
else:
    training_model = trained_model
    joblib.dump(training_model, TRAINING_MODEL_PATH)
    print("Training model initialized and saved")

# Ensure feedback directory exists
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Initialize feedback file if it doesn't exist
if not os.path.exists(FEEDBACK_FILE):
    initial_feedback = pd.DataFrame([
        {'text': 'dummy positive', 'predicted': 'Positive', 'corrected': 'Positive', 'category': 'Test'},
        {'text': 'dummy negative', 'predicted': 'Negative', 'corrected': 'Negative', 'category': 'Test'}
    ])
    initial_feedback.to_csv(FEEDBACK_FILE, index=False)
    print("Feedback file created with initial dummy data")

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.splitlines()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

def append_to_feedback(texts, sentiments, categories):
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        new_feedback = pd.DataFrame({
            'text': texts,
            'predicted': sentiments,
            'corrected': sentiments,  # Initially set corrected same as predicted
            'category': categories
        })
        # Clean data before saving
        new_feedback['text'] = new_feedback['text'].fillna('')
        new_feedback = new_feedback.dropna(subset=['predicted', 'category'])
        
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        feedback_df.to_csv(FEEDBACK_FILE, index=False)
        print(f"Appended {len(texts)} predictions to feedback file")
        return True
    except Exception as e:
        print(f"Error appending to feedback file: {e}")
        return False

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        file = request.files.get('file')

        if text and not file:
            try:
                prediction = trained_model.predict([text])[0]
                sentiment = prediction[0]
                probabilities = trained_model.predict_proba([text])
                sentiment_probs = probabilities[0]
                category_probs = probabilities[1][0]
                categories = trained_model.named_steps['clf'].estimators_[1].classes_
                fixed_categories = ['Child abuse', 'Depression/Suicidal', 'Political abuse', 
                                  'Sexual abuse', 'Religious abuse', 'Sarcasm']
                category_prob_dict = {cat: 0.0 for cat in fixed_categories}
                
                for cat, prob in zip(categories, category_probs):
                    if cat in fixed_categories:
                        category_prob_dict[cat] = float(np.round(prob * 100, 2))
                
                print(f"Prediction for '{text}': Sentiment={sentiment}, Category Probabilities={category_prob_dict}")
                return redirect(url_for('feedback', text=text, prediction=sentiment, **category_prob_dict))
            
            except Exception as e:
                flash(f"Error processing your text: {str(e)}", 'error')
                return redirect(url_for('home'))

        elif file and not text:
            if not file.filename:
                flash('No file selected', 'error')
                return redirect(url_for('home'))
            
            if not allowed_file(file.filename):
                flash('Unsupported file format. Please upload CSV, Excel, or PDF.', 'error')
                return redirect(url_for('home'))
            
            try:
                filename = secure_filename(file.filename.lower())
                if filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(file)
                elif filename.endswith('.pdf'):
                    lines = extract_text_from_pdf(file)
                    df = pd.DataFrame(lines, columns=['Text'])
                else:
                    flash("Unsupported file format. Use CSV, Excel, or PDF.", 'error')
                    return redirect(url_for('home'))

                if 'Text' not in df.columns:
                    flash("No 'Text' column found in the uploaded file.", 'error')
                    return redirect(url_for('home'))

                texts = df['Text'].fillna('').tolist()
                if not texts:
                    flash("No valid text found in the uploaded file.", 'error')
                    return redirect(url_for('home'))

                predictions = trained_model.predict(texts)
                sentiments = predictions[:, 0]
                categories_pred = predictions[:, 1]
                category_probs_all = trained_model.predict_proba(texts)[1]

                if not append_to_feedback(texts, sentiments, categories_pred):
                    flash("Error saving feedback data.", 'error')

                sentiment_counts = pd.Series(sentiments).value_counts(normalize=True) * 100
                sentiment_dist = {
                    'Positive': float(sentiment_counts.get('Positive', 0)),
                    'Negative': float(sentiment_counts.get('Negative', 0))
                }

                fixed_categories = ['Child abuse', 'Depression/Suicidal', 'Political abuse', 
                                  'Sexual abuse', 'Religious abuse', 'Sarcasm']
                category_dist = {cat: 0.0 for cat in fixed_categories}
                
                for i, probs in enumerate(category_probs_all):
                    for cat, prob in zip(trained_model.named_steps['clf'].estimators_[1].classes_, probs):
                        if cat in fixed_categories:
                            category_dist[cat] += prob * 100 / len(texts)

                # Round the values for display
                category_dist = {k: round(v, 2) for k, v in category_dist.items()}

                # Populate comments based on probability threshold (e.g., >20%)
                comments_by_category = {cat: [] for cat in fixed_categories}
                category_classes = trained_model.named_steps['clf'].estimators_[1].classes_
                
                for i, (text, probs) in enumerate(zip(texts, category_probs_all)):
                    for cat, prob in zip(category_classes, probs):
                        if cat in fixed_categories and prob > 0.2:  # Threshold of 20%
                            comments_by_category[cat].append(text)

                return redirect(url_for('feedback_file', 
                                     sentiment_dist=json.dumps(sentiment_dist), 
                                     category_dist=json.dumps(category_dist), 
                                     comments_by_category=json.dumps(comments_by_category)))
            
            except Exception as e:
                flash(f"Error processing file: {str(e)}", 'error')
                return redirect(url_for('home'))

        else:
            flash("Please provide either text or a file, not both.", 'error')
            return redirect(url_for('home'))
    
    return render_template('index.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    text = request.args.get('text', '')
    predicted = request.args.get('prediction', '')
    categories = ['Child abuse', 'Depression/Suicidal', 'Political abuse', 
                 'Sexual abuse', 'Religious abuse', 'Sarcasm']
    category_probs = {cat: float(request.args.get(cat, 0)) for cat in categories}

    if request.method == 'POST':
        corrected = request.form.get('corrected', '')
        category = request.form.get('category', 'Unknown').strip()

        if not corrected:
            flash("Please indicate if the prediction was correct.", 'error')
            return render_template('feedback.html', text=text, predicted=predicted, category_probs=category_probs)

        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            new_feedback = pd.DataFrame([{
                'text': text,
                'predicted': predicted,
                'corrected': corrected,
                'category': category if category else 'Unknown'
            }])
            
            feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
            feedback_df.to_csv(FEEDBACK_FILE, index=False)

            global training_model
            # Clean the data before training
            feedback_df = feedback_df.dropna(subset=['corrected', 'category'])
            feedback_df['text'] = feedback_df['text'].fillna('')
            
            X_feedback = feedback_df['text']
            y_feedback_sentiment = feedback_df['corrected']
            y_feedback_category = feedback_df['category']

            if len(y_feedback_sentiment) > 0 and len(y_feedback_sentiment.unique()) >= 2:
                if not isinstance(training_model.named_steps['clf'], MultiOutputClassifier):
                    print("Reconfiguring training model to MultiOutputClassifier")
                    training_model = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=5000)),
                        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
                    ])
                
                try:
                    training_model.fit(X_feedback, pd.concat([y_feedback_sentiment, y_feedback_category], axis=1))
                    joblib.dump(training_model, TRAINING_MODEL_PATH)
                    print(f"Training model updated with {len(X_feedback)} feedback entries")
                    flash("Thank you for your feedback! The model has been updated.", 'success')
                except Exception as e:
                    print(f"Error updating model: {e}")
                    flash("Feedback saved but model couldn't be updated. Please try again later.", 'warning')
            else:
                flash("Thank you for your feedback!", 'success')

            return redirect(url_for('home'))
        
        except Exception as e:
            print(f"Error processing feedback: {e}")
            flash("Error processing your feedback. Please try again.", 'error')
            return render_template('feedback.html', text=text, predicted=predicted, category_probs=category_probs)

    return render_template('feedback.html', text=text, predicted=predicted, category_probs=category_probs)

@app.route('/feedback_file')
def feedback_file():
    try:
        sentiment_dist = json.loads(request.args.get('sentiment_dist', '{}'))
        category_dist = json.loads(request.args.get('category_dist', '{}'))
        comments_by_category = json.loads(request.args.get('comments_by_category', '{}'))
        
        print(f"Rendering feedback_file with sentiment_dist={sentiment_dist}, category_dist={category_dist}")
        return render_template('feedback.html', 
                            sentiment_dist=sentiment_dist, 
                            category_dist=category_dist, 
                            comments_by_category=comments_by_category)
    except Exception as e:
        print(f"Error rendering feedback file results: {e}")
        flash("Error displaying results. Please try again.", 'error')
        return redirect(url_for('home'))

# Add these new routes to your existing api.py

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    # Load feedback data for visualization
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    
    # Sentiment distribution (convert to dict)
    sentiment_dist = feedback_df['corrected'].value_counts(normalize=True).to_dict()
    
    # Category distribution (convert to dict)
    category_dist = feedback_df['category'].value_counts(normalize=True).to_dict()
    
    # Recent feedback
    recent_feedback = feedback_df.tail(5).to_dict('records')
    
    # Convert percentages to actual percentages (0-100)
    sentiment_dist = {k: round(v * 100, 2) for k, v in sentiment_dist.items()}
    category_dist = {k: round(v * 100, 2) for k, v in category_dist.items()}
    
    return render_template('dashboard.html',
                         sentiment_dist=sentiment_dist,  # Pass as dict, not JSON
                         category_dist=category_dist,    # Pass as dict, not JSON
                         recent_feedback=recent_feedback)

@app.route('/api/feedback_stats')
def feedback_stats():
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    stats = {
        'total_feedback': len(feedback_df),
        'positive': len(feedback_df[feedback_df['corrected'] == 'Positive']),
        'negative': len(feedback_df[feedback_df['corrected'] == 'Negative']),
        'accuracy': "%.2f" % (len(feedback_df[feedback_df['predicted'] == feedback_df['corrected']]) / len(feedback_df) * 100)
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
