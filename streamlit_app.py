import streamlit as st
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI vs Human Text Detector",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-base-openai-detector"
    )
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base-openai-detector"
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- BURSTINESS ----------------
def burstiness(text):
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 3]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return float(np.std(lengths))

# ---------------- MODEL PREDICTION ----------------
def predict_ai(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)
    ai_prob = probs[0][1].item()
    human_prob = probs[0][0].item()

    return ai_prob, human_prob

# ---------------- FINAL DECISION (ENSEMBLE) ----------------
def final_decision(ai_prob, burst, text):
    score = 0
    length = len(text.split())

    if ai_prob > 0.65:
        score += 1
    if burst < 5:       # AI text is smoother
        score += 1
    if length > 120:    # longer text is more reliable
        score += 1

    if score >= 2:
        return "AI-Generated"
    else:
        return "Human-Written"

# ---------------- UI ----------------
st.title("🤖 AI vs Human Text Detector")
st.write("This tool predicts whether text is more likely **AI-generated** or **Human-written**.")

text = st.text_area(
    "Paste your text here 👇",
    height=180
)

if st.button("Analyze"):
    if len(text.strip()) < 40:
        st.warning("Please enter at least 40 characters for better accuracy.")
    else:
        ai_prob, human_prob = predict_ai(text)
        burst = burstiness(text)
        result = final_decision(ai_prob, burst, text)

        st.subheader("🔍 Result")

        if result == "AI-Generated":
            st.error("🧠 More likely AI-Generated")
        else:
            st.success("👤 More likely Human-Written")

        st.write(f"**AI Probability:** {ai_prob:.2f}")
        st.write(f"**Human Probability:** {human_prob:.2f}")
        st.write(f"**Burstiness Score:** {burst:.2f}")

        st.info(
            "Explanation:\n"
            "- Model is trained to detect AI-style patterns\n"
            "- Burstiness measures sentence variation\n"
            "- Final result is based on multiple signals\n\n"
            "⚠️ AI detection is probabilistic, not 100% accurate."
        )
