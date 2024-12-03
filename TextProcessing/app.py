from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from flair.models import TextClassifier
from flair.data import Sentence
import spacy


# Initialize the Flask app
app = Flask(__name__)

# Load SpaCy model for aspect and person extraction
nlp = spacy.load("en_core_web_sm")

# Load HuggingFace Emotion Model for detailed emotion analysis
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Load HuggingFace Sentiment Model (BERT)
bert_sentiment = pipeline("sentiment-analysis", return_all_scores=True)

# Load Flair sentiment model
flair_sentiment = TextClassifier.load('sentiment-fast')

# Function to extract persons from text using SpaCy
def extract_persons(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

# Function to extract noun-based aspects from text using SpaCy
def extract_aspects_spacy(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]

# Function for emotion analysis with emoji representation
def analyze_emotion_bert(text):
    emotions = emotion_model(text)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])
    
    # Map emotions to emojis
    emotion_to_emoji = {
        "joy": "😊",
        "anger": "😠",
        "sadness": "😢",
        "surprise": "😲",
        "fear": "😨",
        "disgust": "🤢",
        "neutral": "😐"
    }
    
    emoji = emotion_to_emoji.get(top_emotion['label'], "❓")  # Default to a question mark if no match
    return {
        "emotion": top_emotion['label'],
        "confidence": f"{top_emotion['score'] * 100:.2f}%",
        "emoji": emoji
    }

# Function for sentiment analysis using BERT
def analyze_sentiment_bert(text):
    sentiments = bert_sentiment(text)[0]
    top_sentiment = max(sentiments, key=lambda x: x['score'])
    return {"sentiment": top_sentiment['label'], "confidence": f"{top_sentiment['score'] * 100:.2f}%"}

def extract_relevant_aspects(text):
    doc = nlp(text)
    aspects = []
    for token in doc:
        if token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj", "pobj"]:
            aspects.append(token.text)
    return aspects


def consolidated_aspect_sentiment_analysis(text, valid_aspects=None):
    """Analyze aspect-based sentiment with optional filtering."""
    if valid_aspects is None:
        valid_aspects = ["camera", "battery", "screen", "resolution", "smartphone"]
    
    # Extract aspects
    aspects = extract_relevant_aspects(text)
    filtered_aspects = [aspect for aspect in aspects if aspect in valid_aspects]

    # Analyze sentiment for each aspect
    aspect_sentiments = {}
    sentences = preprocess_text(text)
    for aspect in filtered_aspects:
        context = " ".join([sentence for sentence in sentences if aspect in sentence])
        
        bert_result = analyze_sentiment_bert(context)
        flair_result = analyze_sentiment_flair(context)
        
        aspect_sentiments[aspect] = {"BERT Sentiment": bert_result, "Flair Sentiment": flair_result}
    return aspect_sentiments


def preprocess_text(text):
    # Basic cleanup: strip unnecessary whitespace
    text = text.strip()
    
    # Split into sentences (basic splitting on '.')
    sentences = text.split('.')
    
    # Remove empty strings and strip leading/trailing spaces from sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences


# Function to analyze aspect-based sentiment
def analyze_aspect_sentiment(aspect, text):
    sentiment_result = analyze_sentiment_bert(text)
    return {
        "aspect": aspect,
        "sentiment": sentiment_result["sentiment"],
        "confidence": sentiment_result["confidence"]
    }

def analyze_sentiment_flair(text):
    sentence = Sentence(text)
    flair_sentiment.predict(sentence)
    sentiment = sentence.labels[0]  # Flair provides label and confidence
    return {"sentiment": sentiment.value, "confidence": f"{sentiment.score * 100:.2f}%"}



def enhanced_person_emotion_detection(text):
    """Detect emotions for persons mentioned in the text."""
    persons = extract_persons(text)
    if not persons:
        return {"message": "No persons identified in the text."}
    
    person_emotions = {}
    sentences = preprocess_text(text)
    for person in persons:
        context = " ".join([sentence for sentence in sentences if person in sentence])
        if context.strip():  # Ensure context isn't empty
            bert_emotion = analyze_emotion_bert(context)
            flair_result = analyze_sentiment_flair(context)

            person_emotions[person] = {
                "BERT Emotion": bert_emotion,
                "Flair Sentiment": flair_result
            }
    return person_emotions





@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.json.get('text', '').strip()

        if not user_input:
            return jsonify({"error": "No input text provided"}), 400

        try:
            # Perform NLP analysis
            emotion_result = analyze_emotion_bert(user_input)
            sentiment_result = analyze_sentiment_bert(user_input)
            aspect_sentiments = consolidated_aspect_sentiment_analysis(user_input)
            persons = extract_persons(user_input)
            person_emotions = enhanced_person_emotion_detection(user_input)

            # Return JSON response
            return jsonify({
                'emotion': emotion_result,
                'sentiment': sentiment_result,
                'aspect_sentiments': aspect_sentiments,
                'persons': persons,
                'person_emotions': person_emotions
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('index.html')  # Serve the HTML page for GET requests

if __name__ == "__main__":
    app.run(debug=True)
