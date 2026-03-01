import streamlit as st
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ---------------- PAGE SETUP ----------------
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
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
    lengths = [len(s.split()) for s in sentences]

    if len(lengths) < 2:
        return 0.0

    return float(np.std(lengths))

# ---------------- PREDICTION ----------------
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
    return probs[0][1].item()  # AI probability

# ---------------- UI ----------------
st.title("🤖 AI vs Human Text Detector")
st.write("Detect whether text is **Human-written or AI-generated**")

text = st.text_area("Paste your text here 👇", height=180)

if st.button("Analyze"):
    if len(text.strip()) < 40:
        st.warning("Please enter at least 40 characters.")
    else:
        ai_prob = predict_ai(text)
        burst = burstiness(text)

        st.subheader("Result")

        if ai_prob > 0.5:
            st.error("🧠 Likely AI-Generated")
        else:
            st.success("👤 Likely Human-Written")

        st.write(f"**AI Probability:** {ai_prob:.2f}")
        st.write(f"**Burstiness Score:** {burst:.2f}")

        st.info(
            "How it works:\n"
            "- Trained OpenAI detector model\n"
            "- Burstiness checks sentence variation\n"
            "- Final result uses learned patterns"
        )
