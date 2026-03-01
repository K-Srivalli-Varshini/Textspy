import streamlit as st
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

st.set_page_config(page_title="AI vs Human Detector", layout="centered")

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("model")
    model = RobertaForSequenceClassification.from_pretrained("model")
    return tokenizer, model

tokenizer, model = load_model()

# -------- Extra AI Tool (Perplexity-like score) --------
def burstiness(text):
    words = text.split()
    lengths = [len(w) for w in words]
    return np.std(lengths)

# -------- Prediction --------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()  # AI probability

# -------- UI --------
st.title("🤖 AI vs Human Text Detector")
st.write("Detect whether text is **Human-written or AI-generated**")

text = st.text_area("Paste your text here:")

if st.button("Analyze"):
    if len(text.strip()) < 20:
        st.warning("Please enter more text")
    else:
        ai_score = predict(text)
        burst = burstiness(text)

        st.subheader("Results")

        if ai_score > 0.6:
            st.error("🧠 Likely AI-Generated")
        else:
            st.success("👤 Likely Human-Written")

        st.write(f"**AI Probability:** {ai_score:.2f}")
        st.write(f"**Burstiness Score:** {burst:.2f}")

        st.info("""
        🔍 **How detection works**
        - RoBERTa → Deep language patterns  
        - Burstiness → Humans write unevenly, AI is smoother
        """)
