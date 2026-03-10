#  Real Estate Price Prediction

A machine learning project that predicts housing prices based on property features such as area, bedrooms, bathrooms, stories, parking, and amenities.  
This project demonstrates a full ML pipeline — from preprocessing and model training to deployment with a Flask web app.

---


##  Project Structure

- **data/**
  - `housing.csv` → Dataset used for training and testing.

- **src/**
  - `preprocessing.py` → Handles data preprocessing (encoding categorical variables, scaling numeric features).
  - `model.py` → Defines machine learning models and evaluation functions.
  - `pipeline.py` → End‑to‑end training pipeline that builds, evaluates, and saves the best model.
  - `__pycache__/` → Auto‑generated Python cache files.

- **app/**
  - `main.py` → Flask web application serving an HTML form for predictions.
  - `model.pkl` → Serialized trained model saved by the pipeline.

- **requirements.txt** → List of Python dependencies required to run the project.
- **Dockerfile** → Configuration for containerizing the application.
- **README.md** → Documentation file describing the project.
