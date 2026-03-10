import os
from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
MODEL_PATH = "app/model.pkl"

# Auto-train if model.pkl is missing
if not os.path.exists(MODEL_PATH):
    from src.pipeline import train_pipeline
    train_pipeline(data_path="data/housing.csv", save_path=MODEL_PATH)

model = joblib.load(MODEL_PATH)

# HTML template with Bootstrap styling
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Real Estate Price Prediction</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="bg-light">
<div class="container mt-5">
    <h2 class="mb-4 text-center">🏠 Real Estate Price Prediction</h2>
    <form method="POST" action="/predict" class="card p-4 shadow-sm">
        <div class="mb-3"><label>Area</label><input type="number" name="area" class="form-control" required></div>
        <div class="mb-3"><label>Bedrooms</label><input type="number" name="bedrooms" class="form-control" required></div>
        <div class="mb-3"><label>Bathrooms</label><input type="number" name="bathrooms" class="form-control" required></div>
        <div class="mb-3"><label>Stories</label><input type="number" name="stories" class="form-control" required></div>
        <div class="mb-3"><label>Main Road</label>
            <select name="mainroad" class="form-select"><option>yes</option><option>no</option></select></div>
        <div class="mb-3"><label>Guestroom</label>
            <select name="guestroom" class="form-select"><option>yes</option><option>no</option></select></div>
        <div class="mb-3"><label>Basement</label>
            <select name="basement" class="form-select"><option>yes</option><option>no</option></select></div>
        <div class="mb-3"><label>Hot Water Heating</label>
            <select name="hotwaterheating" class="form-select"><option>yes</option><option>no</option></select></div>
        <div class="mb-3"><label>Air Conditioning</label>
            <select name="airconditioning" class="form-select"><option>yes</option><option>no</option></select></div>
        <div class="mb-3"><label>Parking</label><input type="number" name="parking" class="form-control" required></div>
        <div class="mb-3"><label>Preferred Area</label>
            <select name="prefarea" class="form-select"><option>yes</option><option>no</option></select></div>
        <div class="mb-3"><label>Furnishing Status</label>
            <select name="furnishingstatus" class="form-select">
                <option>furnished</option><option>semi-furnished</option><option>unfurnished</option>
            </select></div>
        <button type="submit" class="btn btn-primary w-100">Predict Price</button>
    </form>
    {% if prediction %}
    <div class="alert alert-success mt-4 text-center">
        Predicted Price: ₹ {{ prediction }}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    data = {key: request.form[key] for key in request.form}
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return render_template_string(HTML_FORM, prediction=int(prediction))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)