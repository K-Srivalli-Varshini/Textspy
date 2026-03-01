import streamlit as st
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

st.set_page_config(page_title="AI vs Human Detector", layout="centered")

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

# -------- Perplexity (REAL) --------
def perplexity(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()

# -------- Burstiness --------
def burstiness(text):
    sentences = text.split(".")
    lengths = [len(s.split()) for s in sentences if len(s.strip()) > 0]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))

# -------- Prediction --------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[0][1].item()  # AI probability

# -------- UI --------
st.title("🤖 AI vs Human Text Detector")

text = st.text_area("Paste text here", height=180)

if st.button("Analyze"):
    if len(text.strip()) < 40:
        st.warning("Please enter more text (at least 40 characters)")
    else:
        ai_prob = predict(text)
        ppl = perplexity(text)
        burst = burstiness(text)

        st.subheader("Result")

        if ai_prob > 0.5:
            st.error("🧠 Likely AI-Generated")
        else:
            st.success("👤 Likely Human-Written")

        st.write(f"**AI Probability:** {ai_prob:.2f}")
        st.write(f"**Perplexity:** {ppl:.2f}")
        st.write(f"**Burstiness:** {burst:.2f}")

        st.info(
            "Explanation:\n"
            "- Low perplexity → AI-like\n"
            "- High perplexity → Human-like\n"
            "- Burstiness checks writing variation"
        )
