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
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- EXTRA AI TOOL ----------------
def burstiness(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    lengths = [len(word) for word in words]
    return float(np.std(lengths))

# ---------------- PREDICTION ----------------
def predict_ai(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    ai_probability = probs[0][1].item()
    return ai_probability

# ---------------- UI ----------------
st.title("🤖 AI vs Human Text Detector")
st.write("Detect whether text is **Human-written or AI-generated**")

text_input = st.text_area(
    "Paste your text here 👇",
    height=180
)

if st.button("Analyze"):
    if len(text_input.strip()) < 20:
        st.warning("Please enter at least 20 characters.")
    else:
        ai_score = predict_ai(text_input)
        burst_score = burstiness(text_input)

        st.subheader("🔍 Result")

        if ai_score > 0.6:
            st.error("🧠 Likely AI-Generated")
        else:
            st.success("👤 Likely Human-Written")

        st.write(f"**AI Probability:** {ai_score:.2f}")
        st.write(f"**Burstiness Score:** {burst_score:.2f}")

        st.info(
            "How it works:\n"
            "- RoBERTa checks language patterns\n"
            "- Burstiness checks writing variation\n"
            "- AI text is usually smoother"
        )
