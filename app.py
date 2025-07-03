from flask import Flask, request, render_template
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import os

# Flask setup
app = Flask(__name__)

# Download necessary resources
nltk.download('vader_lexicon')

# Cross-platform dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'uploads', 'dataset.csv')

# Load dataset (if it exists) and preprocess
if os.path.exists(DATASET_PATH):
    data = pd.read_csv(DATASET_PATH)
    data['cleaned_text'] = data['review_body'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
else:
    print("⚠️ Dataset not found — proceeding without loading CSV.")
    data = pd.DataFrame()

# VADER Sentiment Analyzer
vader = SentimentIntensityAnalyzer()

# Fine-tuned BERT Sentiment Model
bert_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Utility Functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def vader_predict(text):
    score = vader.polarity_scores(text)['compound']
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def bert_predict(text):
    result = bert_pipeline(text)[0]
    label = result['label']
    if '5' in label or '4' in label:
        return "positive"
    elif '1' in label or '2' in label:
        return "negative"
    else:
        return "neutral"

def ensemble_predict(text):
    predictions = [
        vader_predict(text),
        bert_predict(text)
    ]
    return max(set(predictions), key=predictions.count)

# Routes
@app.route('/')
def index():
    return render_template('index.html', predictions=None)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review = preprocess_text(review)

    vader_sentiment = vader_predict(review)
    bert_sentiment = bert_predict(review)
    ensemble_sentiment = ensemble_predict(review)

    predictions = {
        'vader': vader_sentiment,
        'bert': bert_sentiment,
        'ensemble': ensemble_sentiment
    }

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

@app.route('/health')
def health():
    return "OK", 200
