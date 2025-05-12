import streamlit as st
import pandas as pd
import json
import io
import pdfplumber
from docx import Document
import plotly.express as px
import plotly.io as pio
import ast

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

# ---------- DESCRIPTIVE STATS ----------
def describe_dataframe(df: pd.DataFrame) -> str:
    desc = df.describe(include='all').T.fillna('').to_markdown()
    return f"### 📌 Descriptive Statistics:\n\n{desc}"

# ---------- GPT ANALYSIS + CHART SUGGESTIONS ----------
llm = ChatOpenAI(temperature=0.3)

recommendation_prompt = PromptTemplate(
    input_variables=["stats", "columns"],
    template="""
You are a data visualization expert. Given the dataset statistics and the list of column names, do the following:
1. Write a short professional summary of key insights.
2. Recommend 3 useful visualizations. Format each as a JSON object like:
   {"type": "scatter", "x": "Age", "y": "Income"}
   {"type": "histogram", "x": "Sales"}
   {"type": "bar", "x": "Region", "y": "Profit"}

Stats:
{stats}

Columns:
{columns}

Respond in this format:
### Summary:
[your summary]

### Suggested Visualizations:
[{"type": "scatter", "x": "A", "y": "B"}, {"type": "histogram", "x": "C"}, ...]
"""
)

def generate_gpt_recommendation(df: pd.DataFrame):
    try:
        stats = df.describe(include='all').fillna('').to_string()
        columns = ", ".join(df.columns)
        chain = LLMChain(llm=llm, prompt=recommendation_prompt)
        result = chain.run(stats=stats, columns=columns)

        summary_section = result.split("### Suggested Visualizations:")[0]
        viz_section = result.split("### Suggested Visualizations:")[1]
        suggested_charts = ast.literal_eval(viz_section.strip())

        return summary_section.strip(), suggested_charts
    except Exception as e:
        return f"GPT analysis failed: {e}", []

# ---------- CHART CREATION BASED ON GPT ----------
def generate_charts_from_gpt(df, suggestions):
    charts = []

    for s in suggestions:
        chart_type = s.get("type")
        x = s.get("x")
        y = s.get("y", None)

        if chart_type == "scatter" and x in df.columns and y in df.columns:
            charts.append((f"Scatter: {x} vs {y}", px.scatter(df, x=x, y=y)))
        elif chart_type == "histogram" and x in df.columns:
            charts.append((f"Histogram of {x}", px.histogram(df, x=x)))
        elif chart_type == "bar" and x in df.columns and y in df.columns:
            charts.append((f"Bar Chart: {x} vs {y}", px.bar(df, x=x, y=y)))
        elif chart_type == "line" and x in df.columns and y in df.columns:
            charts.append((f"Line Chart: {x} over {y}", px.line(df, x=x, y=y)))
        elif chart_type == "box" and x in df.columns:
            charts.append((f"Box Plot of {x}", px.box(df, y=x)))

    return charts

# ---------- UNSTRUCTURED SUMMARY ----------
def summarize_text_tool(text: str) -> str:
    summary = text[:300] + "..."  # Placeholder
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
        st.success("Files processed. Choose a tab to explore insights and visualizations.")

        for idx, df in enumerate(structured_data):
            st.header(f"📂 Structured File {idx + 1}")
            tab1, tab2 = st.tabs(["📑 Descriptive Stats", "📈 Visualizations"])

            with tab1:
                st.markdown(describe_dataframe(df))
                st.dataframe(df.head(10))
                summary, suggestions = generate_gpt_recommendation(df)
                st.markdown("### 🔍 GPT-Powered Insights")
                st.markdown(summary)

            with tab2:
                st.markdown("### 📊 Charts Based on GPT Recommendations")
                charts = generate_charts_from_gpt(df, suggestions)
                if charts:
                    for i, (title, fig) in enumerate(charts):
                        st.subheader(title)
                        st.plotly_chart(fig, use_container_width=True)

                        # Export as PNG
                        try:
                            img_bytes = fig.to_image(format="png")
                            st.download_button(
                                label="📥 Download Chart as PNG",
                                data=img_bytes,
                                file_name=f"{title.replace(' ', '_')}.png",
                                mime="image/png",
                                key=f"download_button_{i}"
                            )
                        except Exception as e:
                            st.warning(f"Could not export chart: {e}")
                else:
                    st.warning("No valid chart suggestions found from GPT.")

        for i, text in enumerate(unstructured_data):
            st.header(f"📝 Unstructured File {i + 1}")
            st.markdown(summarize_text_tool(text))
