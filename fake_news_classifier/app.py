import os
import re
import string
import joblib
import numpy as np

from flask import Flask, request, jsonify, render_template_string
from config import VECTORIZER_PATH, NB_MODEL_PATH, SVM_MODEL_PATH

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOP_WORDS = set(ENGLISH_STOP_WORDS)
app = Flask(__name__)

# Load saved vectorizer and models
vectorizer = joblib.load(VECTORIZER_PATH)
nb_model = joblib.load(NB_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)

# Extract SVM coefficients safely
feature_names = np.array(vectorizer.get_feature_names_out())
if hasattr(svm_model, 'calibrated_classifiers_'):
    coef_list = [clf.estimator.coef_[0] for clf in svm_model.calibrated_classifiers_]
    svm_coefs = np.mean(coef_list, axis=0)
elif hasattr(svm_model, 'coef_'):
    svm_coefs = svm_model.coef_[0]
else:
    svm_coefs = np.zeros(len(feature_names))


def clean_input(text: str) -> str:
    text = str(text).lower()
    # Strip web links, domains (like indiatimes.com), punctuation, and numbers
    text = re.sub(r'https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(com|org|net|in|co|gov)\b', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = re.sub(r'\d+', '', text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)


def get_word_contributions(cleaned_text: str):
    words = cleaned_text.split()
    fake_triggers = []
    real_triggers = []

    for word in set(words):
        if word in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[word]
            weight = svm_coefs[idx]
            if weight > 0.10:
                fake_triggers.append((word, round(float(weight), 2)))
            elif weight < -0.10:
                real_triggers.append((word, round(float(abs(weight)), 2)))

    fake_triggers.sort(key=lambda x: x[1], reverse=True)
    real_triggers.sort(key=lambda x: x[1], reverse=True)

    return fake_triggers[:6], real_triggers[:6]


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fake News Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; min-height: 100vh; padding: 40px 20px; }
        .container { max-width: 850px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 30px; font-weight: 700; color: #38bdf8; margin-bottom: 6px; }
        .header p { color: #94a3b8; font-size: 14px; }
        .main-card { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 30px; }
        textarea { width: 100%; height: 160px; padding: 16px; background: #0f172a; border: 1px solid #334155; border-radius: 10px; color: #f1f5f9; font-size: 15px; resize: vertical; margin-bottom: 20px; }
        textarea:focus { outline: none; border-color: #38bdf8; }
        button { background: linear-gradient(135deg, #0284c7, #2563eb); color: white; border: none; width: 100%; padding: 14px; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; }
        button:hover { opacity: 0.95; }
        .results { margin-top: 35px; display: none; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }
        .res-card { background: #0f172a; border: 1px solid #334155; padding: 20px; border-radius: 12px; text-align: center; }
        .res-card h4 { font-size: 12px; text-transform: uppercase; color: #94a3b8; margin-bottom: 12px; }
        .badge { display: inline-block; padding: 8px 22px; border-radius: 30px; font-weight: 700; font-size: 17px; margin-bottom: 8px; }
        .badge-real { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid #22c55e; }
        .badge-fake { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; }
        .prob-text { font-size: 13px; color: #94a3b8; font-weight: 500; }
        .word-container { background: #0f172a; border: 1px solid #334155; padding: 20px; border-radius: 12px; margin-bottom: 25px; }
        .tag-group { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
        .tag { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .tag-fake { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
        .tag-real { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.4); }
        .cleaned-box { background: #0f172a; border: 1px solid #334155; padding: 15px; border-radius: 10px; font-family: monospace; font-size: 13px; color: #94a3b8; word-break: break-all; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Fake News Classifier AI</h1>
            <p>Real-Time Prediction & Explainable AI (XAI) Feature Analysis</p>
        </div>

        <div class="main-card">
            <textarea id="newsInput" placeholder="Paste news article or headline here..."></textarea>
            <button onclick="analyzeText()">Run Authenticity Analysis</button>

            <div id="results" class="results">
                <div class="grid">
                    <div class="res-card">
                        <h4>Calibrated Linear SVM</h4>
                        <div id="svmBadge" class="badge"></div>
                        <div id="svmProb" class="prob-text"></div>
                    </div>
                    <div class="res-card">
                        <h4>Multinomial Naïve Bayes</h4>
                        <div id="nbBadge" class="badge"></div>
                        <div id="nbProb" class="prob-text"></div>
                    </div>
                </div>

                <div class="word-container">
                    <h3 style="font-size:15px; color:#e2e8f0;">Feature Trigger Analysis (Explainable AI)</h3>
                    <div style="margin-top: 12px;">
                        <span style="font-size: 12px; color: #fca5a5; font-weight: 600;">Words Pushing Towards FAKE:</span>
                        <div id="fakeTags" class="tag-group"></div>
                    </div>
                    <div style="margin-top: 12px;">
                        <span style="font-size: 12px; color: #86efac; font-weight: 600;">Words Pushing Towards REAL:</span>
                        <div id="realTags" class="tag-group"></div>
                    </div>
                </div>

                <h3 style="font-size:15px; color:#e2e8f0; margin-bottom:10px;">Processed Clean Text</h3>
                <div id="cleanedText" class="cleaned-box"></div>
            </div>
        </div>
    </div>

    <script>
        async function analyzeText() {
            const text = document.getElementById('newsInput').value;
            if (!text.trim()) return alert('Please input news text first.');

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await res.json();

            document.getElementById('results').style.display = 'block';
            document.getElementById('cleanedText').innerText = data.cleaned_text;

            // SVM Badge
            const svmBadge = document.getElementById('svmBadge');
            svmBadge.innerText = data.svm_prediction;
            svmBadge.className = 'badge ' + (data.svm_prediction === 'Real News' ? 'badge-real' : 'badge-fake');
            document.getElementById('svmProb').innerText = `Confidence: ${data.svm_confidence}%`;

            // NB Badge
            const nbBadge = document.getElementById('nbBadge');
            nbBadge.innerText = data.nb_prediction;
            nbBadge.className = 'badge ' + (data.nb_prediction === 'Real News' ? 'badge-real' : 'badge-fake');
            document.getElementById('nbProb').innerText = `Confidence: ${data.nb_confidence}%`;

            // Render Tags
            const fakeTags = document.getElementById('fakeTags');
            fakeTags.innerHTML = data.fake_triggers.length ? '' : '<span style="color:#64748b; font-size:12px;">None detected</span>';
            data.fake_triggers.forEach(([word, score]) => {
                fakeTags.innerHTML += `<span class="tag tag-fake">${word} (+${score})</span>`;
            });

            const realTags = document.getElementById('realTags');
            realTags.innerHTML = data.real_triggers.length ? '' : '<span style="color:#64748b; font-size:12px;">None detected</span>';
            data.real_triggers.forEach(([word, score]) => {
                realTags.innerHTML += `<span class="tag tag-real">${word} (+${score})</span>`;
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_input(text)
    vec = vectorizer.transform([cleaned])

    # Dynamic class lookup
    svm_classes = list(svm_model.classes_)
    svm_fake_idx = svm_classes.index(1)

    nb_classes = list(nb_model.classes_)
    nb_fake_idx = nb_classes.index(1)

    svm_proba = svm_model.predict_proba(vec)[0]
    fake_prob_svm = svm_proba[svm_fake_idx]

    nb_proba = nb_model.predict_proba(vec)[0]
    fake_prob_nb = nb_proba[nb_fake_idx]

    # Out-of-domain safeguard: Check how many vocabulary terms match
    words = set(cleaned.split())
    known_words = [w for w in words if w in vectorizer.vocabulary_]

    if len(known_words) < 3:
        # Default to Real News if text is sparse or out of vocabulary
        svm_pred_label = 0
        nb_pred_label = 0
        svm_conf = 50.00
        nb_conf = 50.00
    else:
        # Apply strict fake threshold for SVM
        svm_pred_label = 1 if fake_prob_svm >= 0.75 else 0
        nb_pred_label = 1 if fake_prob_nb >= 0.60 else 0

        svm_conf = round(float(fake_prob_svm if svm_pred_label == 1 else (1 - fake_prob_svm)) * 100, 2)
        nb_conf = round(float(fake_prob_nb if nb_pred_label == 1 else (1 - fake_prob_nb)) * 100, 2)

    fake_triggers, real_triggers = get_word_contributions(cleaned)

    return jsonify({
        'cleaned_text': cleaned,
        'svm_prediction': 'Fake News' if svm_pred_label == 1 else 'Real News',
        'nb_prediction': 'Fake News' if nb_pred_label == 1 else 'Real News',
        'svm_confidence': svm_conf,
        'nb_confidence': nb_conf,
        'fake_triggers': fake_triggers,
        'real_triggers': real_triggers
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)