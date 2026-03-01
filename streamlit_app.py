import streamlit as st
import pandas as pd

# Page title
st.set_page_config(page_title="My Streamlit App", layout="centered")

st.title("📊 My First Streamlit App")
st.write("This app runs on Streamlit Community Cloud")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Column Names")
    st.write(list(df.columns))

else:
    st.info("Please upload a CSV file to continue.")
