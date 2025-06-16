
from flask import Flask, render_template, request
import joblib
import numpy as np
from textblob import TextBlob
import random
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import pandas as pd
from deep_translator import GoogleTranslator
import os


app = Flask(__name__)

ridge = joblib.load("ridge_model.joblib")
tfidf = joblib.load("tfidf.joblib")
ohe = joblib.load("encoder.joblib")
corpus = joblib.load("commentary_corpus.joblib")

commentary_history = []

def sentiment_to_score(sentiment):
    if sentiment == "Positive":
        return 2
    elif sentiment == "Neutral":
        return 1
    else:
        return 0

def generate_summary(commentaries):
    sentiments = []
    scores = []

    for comm in commentaries:
        polarity = TextBlob(comm).sentiment.polarity
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        sentiments.append(label)
        scores.append(sentiment_to_score(label))
    combined_text = " ".join(commentaries)
    parser = PlaintextParser.from_string(combined_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, 3)
    summary_text = " ".join([str(sentence) for sentence in summary_sentences])
    avg_score = sum(scores) / len(scores)
    return f"""
📝 Over Summary:
• Avg Excitement Score: {avg_score:.2f} / 2.0 
• 🔥 Positive: {sentiments.count('Positive')} 
• 😐 Neutral: {sentiments.count('Neutral')} 
• 💔 Negative: {sentiments.count('Negative')}
                                                                  
📢 NLP Summary of Last 3 Balls:
• Key Highlights: {summary_text}

"""

@app.route("/", methods=["GET", "POST"])
def home():
    commentary = ""
    sentiment = ""
    sentiment_score = 0.0
    summary = ""
    missing_keywords = []

    if request.method == "POST":
        score = request.form["score"]
        ShotType = request.form["shot"]
        BallType = request.form["ball"]
        Length = request.form["length"]
        Line = request.form["line"]
        WagonWheel = request.form["wagon"]
        batsman = request.form["batsman"]
        bowler = request.form["bowler"]

        input_data = np.array([[score, score, score, ShotType, BallType, Length, Line, WagonWheel]])
        df_input = pd.DataFrame(input_data, columns=[
    'score', 'score_dup1', 'score_dup2', 'ShotType', 'BallType', 'Length', 'Line', 'WagonWheel'
])

        encoded = ohe.transform(df_input)

        pred_vec = ridge.predict(encoded)
        noise = np.random.normal(0, 0.01, pred_vec.shape)
        pred_vec_noisy = pred_vec + noise
        similarities = cosine_similarity(pred_vec_noisy, tfidf.transform(corpus)).flatten()
        top_indices = similarities.argsort()[-10:][::-1]

        if top_indices.size > 0:
            chosen = random.choice(top_indices.tolist())
            commentary = corpus[chosen]

            commentary = commentary.replace("Batsman", batsman).replace("Bowler", bowler)

            check_inputs = [score, ShotType, BallType, Length, Line, WagonWheel]
            for item in check_inputs:
                if str(item).strip().lower() not in commentary.lower():
                    missing_keywords.append(str(item))

           
            
            blob = TextBlob(commentary)
            sentiment_score = blob.sentiment.polarity

            if sentiment_score > 0.1:
                sentiment = "Positive"
            elif sentiment_score < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            engine = pyttsx3.init()
            if sentiment == "Positive":
                engine.setProperty('rate', 170)
                engine.setProperty('volume', 1.0)
            elif sentiment == "Negative":
                engine.setProperty('rate', 130)
                engine.setProperty('volume', 0.8)
            else:
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)

            
            engine = pyttsx3.init()
            engine.save_to_file(commentary, 'static/commentary.mp3')
            engine.runAndWait()

            commentary_history.append(commentary)
            if len(commentary_history) % 3 == 0:
                summary = generate_summary(commentary_history[-3:])
                   
        lang = request.form.get("lang")
        if lang and lang != "English":
            try:
                commentary = GoogleTranslator(source='auto', target=lang.lower()).translate(commentary)
            except:
                commentary += " [Translation failed]"

    return render_template("index.html", commentary=commentary, sentiment=sentiment,
                           summary=summary, sentiment_score=round(sentiment_score, 3))

if __name__ == "__main__":
    app.run(debug=True)