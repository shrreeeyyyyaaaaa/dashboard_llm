import streamlit as st
import pandas as pd
import json
import pdfplumber
from docx import Document
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------- Streamlit Config --------
st.set_page_config(layout="wide")
st.title("Dashboard")

# -------- File Parser --------
def parse_uploaded_files(uploaded_files):
    structured, unstructured = [], []
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            structured.append(pd.read_csv(file))
        elif file.name.endswith((".xlsx", ".xls")):
            structured.append(pd.read_excel(file))
        elif file.name.endswith(".json"):
            content = json.load(file)
            try:
                structured.append(pd.json_normalize(content))
            except:
                unstructured.append(str(content))
        elif file.name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            unstructured.append(text)
        elif file.name.endswith(".txt"):
            unstructured.append(file.read().decode("utf-8"))
        elif file.name.endswith(".docx"):
            doc = Document(file)
            fullText = "\n".join([para.text for para in doc.paragraphs])
            unstructured.append(fullText)
    return structured, unstructured

# -------- GPT Recommendation --------
llm = ChatOpenAI(temperature=0.3)

recommendation_prompt = PromptTemplate(
    input_variables=["stats", "columns"],
    template="""
Stats:
{stats}

Columns:
{columns}

Output:
- Insight 1
- Insight 2
- Insight 3
- Insight 4
- Insight 5
- Chart: Bar:
- Chart: Line: 
"""
)

def generate_gpt_recommendation(df: pd.DataFrame) -> str:
    try:
        stats = df.describe(include='all').fillna('').to_string()
        columns = ", ".join(df.columns)
        chain = LLMChain(llm=llm, prompt=recommendation_prompt)
        result = chain.run(stats=stats, columns=columns)
        return result
    except Exception as e:
        return f"GPT analysis failed: {e}"

# -------- Descriptive Stats --------
def describe_dataframe(df: pd.DataFrame) -> str:
    return df.describe(include='all').T.fillna('').to_markdown()

# -------- Auto Visualization Generator --------
def auto_generate_charts(df):
    charts = []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    dt_cols = df.select_dtypes(include='datetime').columns.tolist()

    # Histogram
    for col in num_cols:
        fig = px.histogram(df, x=col, title=f'Histogram: {col}')
        charts.append((f'Histogram: {col}', fig))

    # Box
    for col in num_cols:
        fig = px.box(df, y=col, title=f'Box Plot: {col}')
        charts.append((f'Box: {col}', fig))

    # Scatter
    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f'Scatter: {num_cols[0]} vs {num_cols[1]}')
        charts.append((f'Scatter: {num_cols[0]} vs {num_cols[1]}', fig))

    # Bar
    for cat in cat_cols[:1]:
        for num in num_cols[:1]:
            fig = px.bar(df, x=cat, y=num, title=f'Bar Chart: {cat} vs {num}')
            charts.append((f'Bar: {cat} vs {num}', fig))

    # Line (for time series)
    if dt_cols:
        for num in num_cols[:1]:
            fig = px.line(df.sort_values(by=dt_cols[0]), x=dt_cols[0], y=num, title=f'Line: {dt_cols[0]} vs {num}')
            charts.append((f'Line: {dt_cols[0]} vs {num}', fig))

    # Heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        charts.append(("Correlation Heatmap", fig))

    # Pie
    if cat_cols:
        value_counts = df[cat_cols[0]].value_counts().nlargest(5)
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                     title=f'Pie Chart: {cat_cols[0]} (Top 5)')
        charts.append((f'Pie: {cat_cols[0]}', fig))

    # Waterfall (sales or material flow)
    try:
        cols = [col for col in num_cols if 'sales' in col.lower() or 'material' in col.lower()]
        if len(cols) >= 1:
            base_col = cols[0]
            changes = df[base_col].dropna().diff().fillna(df[base_col])
            fig = go.Figure(go.Waterfall(
                name="Material Flow",
                orientation="v",
                x=df.index.astype(str),
                y=changes,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(title=f"Waterfall: {base_col} Movement")
            charts.append((f"Waterfall: {base_col}", fig))
    except:
        pass

    return charts

# -------- Export as PNG --------
def get_image_download(fig):
    img_bytes = fig.to_image(format=["png","pdf"])
    return BytesIO(img_bytes)

# -------- Streamlit UI --------
uploaded_files = st.file_uploader("📂 Upload sales/labor data files", type=["csv", "xlsx", "xls", "json", "pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    structured_data, unstructured_data = parse_uploaded_files(uploaded_files)

    for idx, df in enumerate(structured_data):
        st.subheader(f"📑 File {idx + 1}")

        tab1, tab2 = st.tabs(["📋 Descriptive Stats", "📊 Visualizations"])

        with tab1:
            st.markdown("### Descriptive Statistics")
            st.markdown(describe_dataframe(df))
            st.dataframe(df.head(10))

            st.markdown("### 🔍 GPT-Powered Insights")
            insights = generate_gpt_recommendation(df)
            st.markdown(insights)

        with tab2:
            st.markdown("### 📈 Auto-generated Charts")
            charts = auto_generate_charts(df)
            for title, fig in charts:
                st.subheader(title)
                st.plotly_chart(fig, use_container_width=True)
                png_data = get_image_download(fig)
                st.download_button(f"Download {title} as PNG", png_data, file_name=f"{title}.png", mime="image/png")
