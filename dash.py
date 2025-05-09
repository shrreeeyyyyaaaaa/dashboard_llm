import streamlit as st
import pandas as pd
import json
from utils.file_parser import parse_uploaded_files
from utils.langchain_agent import run_agent_with_tools
import plotly.io as pio

st.set_page_config(layout="wide")
st.title("📊 Smart Data Dashboard using LangChain + Plotly")

uploaded_files = st.file_uploader("Upload your data files (CSV, Excel, PDF, TXT, JSON, DOCX)",
                                   type=["csv", "xlsx", "xls", "pdf", "txt", "json", "docx"],
                                   accept_multiple_files=True)

if uploaded_files:
    structured_data, unstructured_data = parse_uploaded_files(uploaded_files)

    if not structured_data and not unstructured_data:
        st.warning("No readable content found. Please upload valid files.")
    else:
        st.success("Files processed. Generating insights and visualizations...")
        agent_outputs = run_agent_with_tools(structured_data, unstructured_data)

        for block in agent_outputs:
            st.subheader(block["title"])
            if "description" in block:
                st.markdown(block["description"])
            if "plot" in block:
                fig = pio.from_json(block["plot"])
                st.plotly_chart(fig, use_container_width=True)
            if "text" in block:
                st.markdown(block["text"])

# utils/file_parser.py
import pandas as pd
import json
import io
import pdfplumber
from docx import Document

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

# utils/langchain_agent.py
import plotly.express as px
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import random

def smart_plot_tool(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns
    datetime_cols = df.select_dtypes(include='datetime').columns

    if len(datetime_cols) and len(numeric_cols):
        fig = px.line(df, x=datetime_cols[0], y=numeric_cols[0])
    elif len(categorical_cols) and len(numeric_cols):
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
    elif len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
    else:
        fig = px.histogram(df, x=numeric_cols[0])
    return fig.to_json()

def summarize_text_tool(text: str) -> str:
    summary = text[:300] + "..."  # Replace with LLM summary logic if needed
    return f"### Text Summary:\n\n{summary}"

def describe_dataframe(df: pd.DataFrame) -> str:
    desc = df.describe(include='all').T.fillna('').to_markdown()
    return f"### Descriptive Statistics:\n\n{desc}"

def run_agent_with_tools(structured_data, unstructured_data):
    tools = []
    responses = []

    for df in structured_data:
        plot = smart_plot_tool(df)
        description = describe_dataframe(df)
        responses.append({"title": "📈 Auto Plot for Structured Data",
                          "plot": plot,
                          "description": description})

    for text in unstructured_data:
        summary = summarize_text_tool(text)
        responses.append({"title": "🧠 Summary of Unstructured Data", "text": summary})

    return responses
