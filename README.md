# Sentiment Analysis Web App

An end-to-end AI web application that trains its own ML model to classify text as Positive, Neutral, or Negative.

## 🗂 Project Structure
```
Sentiment-Analysis/
├── backend/
│   ├── main.py
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── dataset/
│   └── sentiment.csv
├── training/
│   └── train_model.py
└── README.md
```

## 🚀 Setup Instructions

### 1. Train the Model
```bash
cd training
python train_model.py
```
This generates `model.pkl` and `vectorizer.pkl` inside `backend/`.

### 2. Run the Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
Backend will run at: `http://127.0.0.1:8000`

Test it directly:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this internship!"}'
```

### 3. Run the Frontend
Simply open `frontend/index.html` in your browser
(or serve it with `python -m http.server` inside the `frontend/` folder).

## 🧪 Test Cases
| Input | Expected |
|---|---|
| "I love this!" | positive |
| "This is terrible" | negative |
| "It's an average product" | neutral |
| "" (empty) | error message |
| Long paragraph | still returns a prediction |

## 📊 Dataset
`dataset/sentiment.csv` is a small sample dataset (41 rows) for demonstration.
Replace it with a larger real-world dataset for better accuracy.

## 🛠 Tech Stack
Python, Pandas, Scikit-learn, FastAPI, HTML/CSS/JavaScript
