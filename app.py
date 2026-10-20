import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Animated Series Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Animated Series Sentiment Analysis")
st.write(
    "Enter a viewer comment and the model will classify its overall sentiment "
    "as **Positive**, **Negative**, or **Neutral**."
)


# -----------------------------
# Text preprocessing
# Same preprocessing used in the notebook
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Train the model
# Same pipeline used in the final notebook:
# 80/20 stratified split -> training-only
# random oversampling -> unigram TF-IDF ->
# Logistic Regression (C=0.5)
# -----------------------------
@st.cache_resource
def train_model():
    data_path = Path(__file__).parent / "Sentiment Collection_Expanded(3).csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.name}. "
            "Place the CSV file in the same folder as app.py."
        )

    sentiment_data = pd.read_csv(data_path)

    sentiment_mapping = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }

    sentiment_data["sentiment_encoded"] = sentiment_data["sentiment"].map(
        sentiment_mapping
    )
    sentiment_data["clean_tweet"] = sentiment_data["tweet"].apply(clean_text)

    X = sentiment_data["clean_tweet"]
    y = sentiment_data["sentiment_encoded"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_data = pd.DataFrame(
        {
            "tweet": X_train.values,
            "sentiment": y_train.values,
        }
    )

    negative = train_data[train_data["sentiment"] == 0]
    neutral = train_data[train_data["sentiment"] == 1]
    positive = train_data[train_data["sentiment"] == 2]

    max_size = max(len(negative), len(neutral), len(positive))

    negative_upsampled = resample(
        negative,
        replace=True,
        n_samples=max_size,
        random_state=42,
    )

    neutral_upsampled = resample(
        neutral,
        replace=True,
        n_samples=max_size,
        random_state=42,
    )

    positive_upsampled = resample(
        positive,
        replace=True,
        n_samples=max_size,
        random_state=42,
    )

    balanced_train = pd.concat(
        [
            negative_upsampled,
            neutral_upsampled,
            positive_upsampled,
        ]
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(balanced_train["tweet"])

    y_train_balanced = balanced_train["sentiment"]

    model = LogisticRegression(C=0.5, max_iter=1000)
    model.fit(X_train_tfidf, y_train_balanced)

    return model, tfidf, sentiment_data, balanced_train


# -----------------------------
# Load model
# -----------------------------
try:
    model, tfidf, sentiment_data, balanced_train = train_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# -----------------------------
# User interface
# -----------------------------
comment = st.text_area(
    "Enter a comment:",
    placeholder="Example: I really enjoyed the animation and the characters!",
    height=140,
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    else:
        cleaned_text = clean_text(comment)
        text_tfidf = tfidf.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]

        sentiment_labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }

        sentiment = sentiment_labels[prediction]

        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: **{sentiment}**")
        elif sentiment == "Negative":
            st.error(f"Predicted Sentiment: **{sentiment}**")
        else:
            st.info(f"Predicted Sentiment: **{sentiment}**")


# -----------------------------
# Project information
# -----------------------------
with st.sidebar:
    st.header("Project Information")
    st.write("**Dataset:** 215 comments")
    st.write("**Classes:** Positive, Negative, Neutral")
    st.write("**Features:** Unigram TF-IDF")
    st.write("**Classifier:** Logistic Regression")
    st.write("**Regularization:** C = 0.5")
    st.write("**Training balance:** 70 / 70 / 70")
    st.caption(
        "The training data is balanced using random oversampling. "
        "The model follows the same preprocessing and training pipeline "
        "used in the project notebook."
    )
