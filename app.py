from flask import Flask, request, jsonify, render_template
import joblib
import os
from pathlib import Path

app = Flask(__name__)

# Load artifacts relative to this file to avoid issues when the working
# directory is different (for example when running via a WSGI server).
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "Spam_Detection.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept JSON (from the single-page UI) or form-encoded data.
        data = request.get_json(silent=True) or {}
        # request.get_json can return None if the content-type isn't JSON;
        # fall back to form/query values as needed.
        if not isinstance(data, dict):
            data = {}

        msg = data.get("message") or request.values.get("message", "")

        if not isinstance(msg, str) or not msg.strip():
            return jsonify({"error": "Empty message."}), 400

        vec = vectorizer.transform([msg])

        # If vectorizer returns a sparse matrix it will have .nnz; if it's
        # a dense numpy array, count non-zero entries instead.
        try:
            nonzero = vec.nnz
        except Exception:
            # avoid adding scikit-learn/numpy imports at module level; do
            # a local check here
            import numpy as _np
            nonzero = int(_np.count_nonzero(vec))

        if nonzero == 0:
            return jsonify({"error": "Message has no recognizable features."}), 400

        pred = model.predict(vec)[0]

        # Some models (e.g., certain estimators) may not implement
        # predict_proba; handle that gracefully.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = f"{max(proba) * 100:.2f}%"
        else:
            confidence = "N/A"

        return jsonify({
            "prediction": "Spam" if int(pred) == 1 else "Not Spam",
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
