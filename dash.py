import streamlit as st
import pandas as pd
import json
import io
import pdfplumber
from docx import Document
import plotly.express as px
import plotly.io as pio

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------- PAGE CONFIG & STYLING ----------
st.set_page_config(layout="wide")
st.title("📊 Smart Data Dashboard using LangChain with Plotly")

st.markdown("""
    <style>
        body { background-color: #fdf6e3; }
        .block-container { padding: 2rem; }
        .stMarkdown h2 { color: #3a3a3a; margin-top: 2rem; }
        .stMarkdown h3 { color: #444; }
        .stDataFrame { background-color: #fff8dc; }
    </style>
""", unsafe_allow_html=True)

# ---------- FILE PARSER ----------
def parse_uploaded_files(uploaded_files):
    structured = []
    unstructured = []

    for file in uploaded_files:
        filename = file.name.lower()
        if filename.endswith(".csv"):
            structured.append(pd.read_csv(file))
        elif filename.endswith((".xlsx", ".xls")):
            structured.append(pd.read_excel(file))
        elif filename.endswith(".json"):
            content = json.load(file)
            try:
                structured.append(pd.json_normalize(content))
            except Exception:
                unstructured.append(str(content))
        elif filename.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            unstructured.append(text)
        elif filename.endswith(".txt"):
            unstructured.append(file.read().decode("utf-8"))
        elif filename.endswith(".docx"):
            doc = Document(file)
            fullText = "\n".join([para.text for para in doc.paragraphs])
            unstructured.append(fullText)

    return structured, unstructured

# ---------- VISUALIZATION TOOLS ----------
def smart_plots(df):
    figures = []
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns
    datetime_cols = df.select_dtypes(include='datetime').columns

    if len(datetime_cols) and len(numeric_cols):
        figures.append(px.line(df, x=datetime_cols[0], y=numeric_cols[0]))
    if len(categorical_cols) and len(numeric_cols):
        figures.append(px.bar(df, x=categorical_cols[0], y=numeric_cols[0]))
    if len(numeric_cols) >= 2:
        figures.append(px.scatter(df, x=numeric_cols[0], y=numeric_cols[1]))
    if len(numeric_cols) >= 1:
        figures.append(px.histogram(df, x=numeric_cols[0]))

    return [fig.to_json() for fig in figures]

def describe_dataframe(df: pd.DataFrame) -> str:
    desc = df.describe(include='all').T.fillna('').to_markdown()
    return f"### 📌 Descriptive Statistics:\n\n{desc}"

# ---------- GPT RECOMMENDATION ----------
llm = ChatOpenAI(temperature=0.3)

recommendation_prompt = PromptTemplate(
    input_variables=["stats", "columns"],
    template="""
You are a senior data analyst. Given the descriptive statistics and column names below, provide a professional, concise analysis summary and key insights or patterns.

Descriptive Stats:
{stats}

Column Names:
{columns}

Your Output:
"""
)

def generate_gpt_recommendation(df: pd.DataFrame) -> str:
    try:
        stats = df.describe(include='all').fillna('').to_string()
        columns = ", ".join(df.columns)
        chain = LLMChain(llm=llm, prompt=recommendation_prompt)
        result = chain.run(stats=stats, columns=columns)
        return f"### 🔍 GPT-Powered Insights:\n\n{result}"
    except Exception as e:
        return f"GPT analysis unavailable: {e}"

# ---------- UNSTRUCTURED SUMMARY ----------
def summarize_text_tool(text: str) -> str:
    summary = text[:300] + "..."  # Replace with LLM summarization logic
    return f"### 🧠 Summary of Unstructured Data:\n\n{summary}"

# ---------- MAIN APP ----------
uploaded_files = st.file_uploader("Upload your data files (CSV, Excel, PDF, TXT, JSON, DOCX)",
                                   type=["csv", "xlsx", "xls", "pdf", "txt", "json", "docx"],
                                   accept_multiple_files=True)

if uploaded_files:
    structured_data, unstructured_data = parse_uploaded_files(uploaded_files)

    if not structured_data and not unstructured_data:
        st.warning("No readable content found. Please upload valid files.")
    else:
        st.success("Files processed. Generating insights and visualizations...")

        for idx, df in enumerate(structured_data):
            st.header(f"📂 Structured File {idx + 1}")
            st.markdown(describe_dataframe(df))
            st.dataframe(df.head(10))
            st.markdown(generate_gpt_recommendation(df))
            plot_jsons = smart_plots(df)
            for pj in plot_jsons:
                fig = pio.from_json(pj)
                st.plotly_chart(fig, use_container_width=True)

        for i, text in enumerate(unstructured_data):
            st.header(f"📝 Unstructured File {i + 1}")
            st.markdown(summarize_text_tool(text))
