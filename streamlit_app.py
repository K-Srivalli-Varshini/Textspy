import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

st.title("AI vs Human Text Detector")

# Load model & tokenizer
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("ai_human_detector")
    model = RobertaForSequenceClassification.from_pretrained("ai_human_detector")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter text here")

if st.button("Check"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
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
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "AI Generated" if pred == 1 else "Human Written"

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}")
