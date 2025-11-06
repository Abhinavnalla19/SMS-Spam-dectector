## SMS Spam Detector

AI-powered SMS Spam Detector — a lightweight Flask web app that uses a scikit-learn model to classify SMS messages as Spam or Not Spam.

### What's inside
- `Spam_Detection.ipynb`: Data cleaning, TF-IDF feature extraction, model training, and evaluation.
- `app.py`: Flask app exposing a `/predict` API and serving the UI.
- `templates/index.html`: Modern UI to paste SMS text, get a prediction, and see a confidence score.
- `requirements.txt`: Python dependencies.

### Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000/ in your browser.

### Features
- Real-time SMS spam classification via a clean web UI
- TF-IDF + scikit-learn model
- Displays confidence score with simple visual feedback
- Reproducible training steps in the notebook

### Suggested topics
python, flask, machine-learning, scikit-learn, spam-detection, tfidf, sms

### Author
Abhinav