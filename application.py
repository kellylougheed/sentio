import nltk
from urllib import request as requestURL
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from flask import Flask, redirect, render_template, request, session
from flask_session import Session


# Configure application
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST", "GET"])
def analyze():
    if request.method == "GET":
        return render_template("index.html")

    if not request.files["file"] and not request.form.get("text"):
        return render_template("error.html", msg="Missing file or text")
    if request.files["file"]:
        try:
            text = request.files["file"].read().decode("utf-8")
        except:
            return render_template("error.html", msg="Invalid file")

    elif request.form.get("text"):
        text = request.form.get("text")

    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    res = sid.polarity_scores(text)
    res['neg'] *= 100
    res['neg'] = round(res['neg'], 2)
    res['pos'] *= 100
    res['pos'] = round(res['pos'], 2)
    res['neu'] *= 100
    res['neu'] = round(res['neu'], 2)

    return render_template("analyze.html", res=res)


@app.route("/freq", methods=["GET"])
def freq():
    return render_template("freq.html")


@app.route("/analyze-freq", methods=["POST", "GET"])
def analyze_freq():
    if request.method == "GET":
        return render_template("freq.html")
    if not request.form.get("url") and not request.form.get("text"):
        return render_template("error.html", msg="Missing URL or text")
    if request.form.get("url"):
        url = request.form.get("url")
        if url[-4:] != ".txt":
            return render_template("error.html", msg="Please submit a .txt URL")
        try:
            response = requestURL.urlopen(url)
        except:
            return render_template("error.html", msg="No response from URL")
        try:
            raw = response.read().decode('utf8')
        except:
            return render_template("error.html", msg="Unable to decode file")
        try:
            tokens = nltk.word_tokenize(raw)
        except:
            return render_template("error.html", msg="Error tokenizing text")
        try:
            text = nltk.Text(tokens)
        except:
            return render_template("error.html", msg="Error interpreting text")
    elif request.form.get("text"):
        text = request.form.get("text")
        tokens = nltk.word_tokenize(text)
        text = nltk.Text(tokens)

    words = request.form.get("word")
    words = words.split(",")
    words_results = []
    for w in words:
        count = text.count(w.strip())
        percentage = round((count/len(text))*100, 3)
        words_results.append({
            "word": w.strip(),
            "count": count,
            "percentage": percentage
        })

    return render_template("analyze-freq.html", words=words_results)
