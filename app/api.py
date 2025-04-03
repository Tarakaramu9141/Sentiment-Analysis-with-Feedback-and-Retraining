import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from sklearn.base import InconsistentVersionWarning
import warnings

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Configuration
TRAINED_MODEL_PATH = 'D:/Projects_for_resume/NLP_Sentiment_Analysis/models/trained_model.pkl'
TRAINING_MODEL_PATH = 'D:/Projects_for_resume/NLP_Sentiment_Analysis/models/training_model.pkl'
FEEDBACK_DIR = 'D:/Projects_for_resume/NLP_Sentiment_Analysis/feedback'
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'feedback_data.csv')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'pdf'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        trained_model = joblib.load(TRAINED_MODEL_PATH)
        st.success("Trained model loaded successfully")
    except Exception as e:
        st.error(f"Error loading trained model: {e}")
        raise

    if os.path.exists(TRAINING_MODEL_PATH):
        try:
            training_model = joblib.load(TRAINING_MODEL_PATH)
            st.success("Training model loaded from file")
        except Exception as e:
            st.warning(f"Error loading training model, creating new one: {e}")
            training_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
            ])
            joblib.dump(training_model, TRAINING_MODEL_PATH)
    else:
        training_model = trained_model
        joblib.dump(training_model, TRAINING_MODEL_PATH)
        st.success("Training model initialized and saved")
    
    return trained_model, training_model

# Initialize feedback directory and file
def init_feedback():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    
    if not os.path.exists(FEEDBACK_FILE):
        initial_feedback = pd.DataFrame([
            {'text': 'dummy positive', 'predicted': 'Positive', 'corrected': 'Positive', 'category': 'Test'},
            {'text': 'dummy negative', 'predicted': 'Negative', 'corrected': 'Negative', 'category': 'Test'}
        ])
        initial_feedback.to_csv(FEEDBACK_FILE, index=False)
        st.info("Feedback file created with initial dummy data")

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.splitlines()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
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
        st.success(f"Appended {len(texts)} predictions to feedback file")
        return True
    except Exception as e:
        st.error(f"Error appending to feedback file: {e}")
        return False

def analyze_text(text, model):
    try:
        prediction = model.predict([text])[0]
        sentiment = prediction[0]
        probabilities = model.predict_proba([text])
        sentiment_probs = probabilities[0]
        category_probs = probabilities[1][0]
        categories = model.named_steps['clf'].estimators_[1].classes_
        fixed_categories = ['Child abuse', 'Depression/Suicidal', 'Political abuse', 
                          'Sexual abuse', 'Religious abuse', 'Sarcasm']
        category_prob_dict = {cat: 0.0 for cat in fixed_categories}
        
        for cat, prob in zip(categories, category_probs):
            if cat in fixed_categories:
                category_prob_dict[cat] = float(np.round(prob * 100, 2))
        
        st.session_state['last_analysis'] = {
            'text': text,
            'sentiment': sentiment,
            'category_probs': category_prob_dict
        }
        
        return sentiment, category_prob_dict
    
    except Exception as e:
        st.error(f"Error processing your text: {str(e)}")
        return None, None

def analyze_file(file, model):
    try:
        filename = file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif filename.endswith('.pdf'):
            lines = extract_text_from_pdf(file)
            df = pd.DataFrame(lines, columns=['Text'])
        else:
            st.error("Unsupported file format. Use CSV, Excel, or PDF.")
            return None, None, None

        if 'Text' not in df.columns:
            st.error("No 'Text' column found in the uploaded file.")
            return None, None, None

        texts = df['Text'].fillna('').tolist()
        if not texts:
            st.error("No valid text found in the uploaded file.")
            return None, None, None

        predictions = model.predict(texts)
        sentiments = predictions[:, 0]
        categories_pred = predictions[:, 1]
        category_probs_all = model.predict_proba(texts)[1]

        if not append_to_feedback(texts, sentiments, categories_pred):
            st.error("Error saving feedback data.")

        sentiment_counts = pd.Series(sentiments).value_counts(normalize=True) * 100
        sentiment_dist = {
            'Positive': float(sentiment_counts.get('Positive', 0)),
            'Negative': float(sentiment_counts.get('Negative', 0))
        }

        fixed_categories = ['Child abuse', 'Depression/Suicidal', 'Political abuse', 
                          'Sexual abuse', 'Religious abuse', 'Sarcasm']
        category_dist = {cat: 0.0 for cat in fixed_categories}
        
        for i, probs in enumerate(category_probs_all):
            for cat, prob in zip(model.named_steps['clf'].estimators_[1].classes_, probs):
                if cat in fixed_categories:
                    category_dist[cat] += prob * 100 / len(texts)

        # Round the values for display
        category_dist = {k: round(v, 2) for k, v in category_dist.items()}

        # Populate comments by category (including empty categories)
        comments_by_category = {cat: [] for cat in fixed_categories}  # Initialize all categories
        category_classes = model.named_steps['clf'].estimators_[1].classes_
        
        for i, (text, probs) in enumerate(zip(texts, category_probs_all)):
            for cat, prob in zip(category_classes, probs):
                if cat in fixed_categories and prob > 0.2:  # Threshold of 20%
                    comments_by_category[cat].append(text)
        
        # Store the results in session state
        st.session_state.file_results = {
            'sentiment_dist': sentiment_dist,
            'category_dist': category_dist,
            'comments_by_category': comments_by_category,
            'texts': texts
        }
        
        return sentiment_dist, category_dist, comments_by_category
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None, None

def show_file_results():
    if 'file_results' not in st.session_state:
        st.warning("No file analysis results available. Please analyze a file first.")
        return
    
    results = st.session_state.file_results
    
    st.subheader("File Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sentiment Distribution:**")
        st.bar_chart(results['sentiment_dist'])
    
    with col2:
        st.write("**Category Distribution:**")
        st.bar_chart(results['category_dist'])
    
    # Show comments by category
    st.subheader("Comments by Category")
    
    # Define all possible categories
    all_categories = [
        'Child abuse', 
        'Depression/Suicidal', 
        'Political abuse',
        'Sexual abuse', 
        'Religious abuse', 
        'Sarcasm'
    ]
    
    # Create selector with all categories
    selected_category = st.selectbox(
        "Select category to view comments", 
        all_categories,
        key='category_selector'
    )
    
    # Check if selected category exists in results and has comments
    if (selected_category in results['comments_by_category'] and 
        len(results['comments_by_category'][selected_category]) > 0):
        
        st.write(f"**Comments for {selected_category}:**")
        
        # Display each comment
        for i, comment in enumerate(results['comments_by_category'][selected_category]):
            # Calculate dynamic height (min 68px, max 300px)
            comment_height = max(68, min(300, 68 + (len(comment) // 3)))
            
            st.text_area(
                label=f"Comment {i+1}", 
                value=comment, 
                key=f"comment_{i}_{selected_category}",
                height=comment_height,
                disabled=True
            )
            
            # Add divider between comments (except after last one)
            if i < len(results['comments_by_category'][selected_category]) - 1:
                st.divider()
    else:
        st.info(f"No comments available in the '{selected_category}' category")

def save_feedback(text, predicted, corrected, category):
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
        st.success("Feedback saved successfully!")
        
        # Retrain model with new feedback
        retrain_model(feedback_df)
        
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

def retrain_model(feedback_df):
    try:
        # Clean the data before training
        feedback_df = feedback_df.dropna(subset=['corrected', 'category'])
        feedback_df['text'] = feedback_df['text'].fillna('')
        
        X_feedback = feedback_df['text']
        y_feedback_sentiment = feedback_df['corrected']
        y_feedback_category = feedback_df['category']

        if len(y_feedback_sentiment) > 0 and len(y_feedback_sentiment.unique()) >= 2:
            if not isinstance(st.session_state.training_model.named_steps['clf'], MultiOutputClassifier):
                st.info("Reconfiguring training model to MultiOutputClassifier")
                st.session_state.training_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
                ])
            
            try:
                st.session_state.training_model.fit(X_feedback, pd.concat([y_feedback_sentiment, y_feedback_category], axis=1))
                joblib.dump(st.session_state.training_model, TRAINING_MODEL_PATH)
                st.success(f"Model updated with {len(X_feedback)} feedback entries")
            except Exception as e:
                st.warning(f"Model couldn't be updated. Error: {e}")
    except Exception as e:
        st.error(f"Error during model retraining: {e}")

def show_dashboard():
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        
        st.subheader("Feedback Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", len(feedback_df))
        
        with col2:
            positive = len(feedback_df[feedback_df['corrected'] == 'Positive'])
            st.metric("Positive Feedback", positive)
        
        with col3:
            negative = len(feedback_df[feedback_df['corrected'] == 'Negative'])
            st.metric("Negative Feedback", negative)
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_dist = feedback_df['corrected'].value_counts(normalize=True) * 100
        st.bar_chart(sentiment_dist)
        
        # Category distribution
        st.subheader("Category Distribution")
        category_dist = feedback_df['category'].value_counts(normalize=True) * 100
        st.bar_chart(category_dist)
        
        # Recent feedback
        st.subheader("Recent Feedback")
        st.dataframe(feedback_df.tail(5))
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")

def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    
    # Initialize session state
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'file_results' not in st.session_state:
        st.session_state.file_results = None
    
    # Load models and initialize feedback
    st.session_state.trained_model, st.session_state.training_model = load_models()
    init_feedback()
    
    st.title("Sentiment Analysis with Feedback and Retraining")
    
    # Navigation
    menu = ["Home", "Dashboard", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.header("Analyze Text or File")
        
        tab1, tab2 = st.tabs(["Text Analysis", "File Analysis"])
        
        with tab1:
            text = st.text_area("Enter text to analyze:", height=150)
            if st.button("Analyze Text"):
                if text.strip():
                    sentiment, category_probs = analyze_text(text, st.session_state.trained_model)
                    if sentiment:
                        st.subheader("Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Sentiment", sentiment)
                        
                        with col2:
                            st.write("**Category Probabilities:**")
                            for cat, prob in category_probs.items():
                                st.progress(int(prob), text=f"{cat}: {prob}%")
                        
                        # Feedback form
                        st.subheader("Provide Feedback")
                        with st.form("text_feedback"):
                            corrected = st.radio("Is this correct?", 
                                               ["Correct", "Incorrect"], 
                                               index=0 if sentiment == "Positive" else 1)
                            category = st.selectbox("Category", 
                                                  ['Child abuse', 'Depression/Suicidal', 'Political abuse', 
                                                   'Sexual abuse', 'Religious abuse', 'Sarcasm', 'Other'])
                            
                            if st.form_submit_button("Submit Feedback"):
                                corrected_sentiment = "Positive" if corrected == "Correct" else "Negative"
                                save_feedback(text, sentiment, corrected_sentiment, category)
                else:
                    st.warning("Please enter some text to analyze")
        
        with tab2:
            uploaded_file = st.file_uploader("Upload a file (CSV, Excel, PDF)", type=['csv', 'xlsx', 'pdf'])
            if uploaded_file is not None:
                if st.button("Analyze File"):
                    with st.spinner("Analyzing file..."):
                        sentiment_dist, category_dist, comments_by_category = analyze_file(
                            uploaded_file, 
                            st.session_state.trained_model
                        )
            
            # Show results if available
            if 'file_results' in st.session_state and st.session_state.file_results:
                show_file_results()
    
    elif choice == "Dashboard":
        show_dashboard()
    
    elif choice == "About":
        st.header("About This Application")
        st.write("""
        This is a sentiment analysis application that can:
        - Analyze text for sentiment (Positive/Negative)
        - Categorize text into different categories
        - Learn from user feedback to improve predictions
        - Process files (CSV, Excel, PDF) in bulk
        """)
        st.write("The application uses machine learning models to make predictions.")

if __name__ == '__main__':
    main()
