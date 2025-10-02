import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# Set Streamlit page settings
st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA) Dashboard")
st.markdown("Upload your dataset and explore with **12 unique visualizations**")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset info
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # Select numeric columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ---- Visualization Section ----
    st.header("📈 Visualizations")

    # 1. Histogram
    st.subheader("1. Histogram")
    col = st.selectbox("Choose column for histogram", num_cols, key="hist")
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # 2. Boxplot
    st.subheader("2. Boxplot")
    col = st.selectbox("Choose column for boxplot", num_cols, key="box")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

    # 3. Violin Plot
    st.subheader("3. Violin Plot")
    col = st.selectbox("Choose column for violin plot", num_cols, key="violin")
    fig, ax = plt.subplots()
    sns.violinplot(x=df[col], ax=ax)
    st.pyplot(fig)

    # 4. Scatter Plot
    st.subheader("4. Scatter Plot")
    x_axis = st.selectbox("X-axis", num_cols, key="scatter_x")
    y_axis = st.selectbox("Y-axis", num_cols, key="scatter_y")
    fig = px.scatter(df, x=x_axis, y=y_axis, color=df[num_cols[0]])
    st.plotly_chart(fig)

    # 5. Line Chart
    st.subheader("5. Line Chart")
    col = st.selectbox("Choose column for line chart", num_cols, key="line")
    fig = px.line(df, y=col)
    st.plotly_chart(fig)

    # 6. Bar Chart
    st.subheader("6. Bar Chart")
    col = st.selectbox("Choose column for bar chart", df.columns, key="bar")
    fig = px.bar(df, x=df.index, y=col)
    st.plotly_chart(fig)

    # 7. Pairplot (Seaborn)
    st.subheader("7. Pairplot")
    st.write("Showing pairplot for first 4 numeric columns")
    fig = sns.pairplot(df[num_cols[:4]])
    st.pyplot(fig)

    # 8. Correlation Heatmap
    st.subheader("8. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 9. Density Plot (KDE)
    st.subheader("9. Density Plot")
    col = st.selectbox("Choose column for density plot", num_cols, key="kde")
    fig = ff.create_distplot([df[col].dropna()], [col], show_hist=False)
    st.plotly_chart(fig)

    # 10. Treemap
    st.subheader("10. Treemap")
    if len(df.columns) >= 2:
        fig = px.treemap(df, path=[df.columns[0]], values=num_cols[0])
        st.plotly_chart(fig)

    # 11. Count Plot (Categorical Feature)
    st.subheader("11. Count Plot")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols) > 0:
        col = st.selectbox("Choose categorical column", cat_cols, key="count")
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 12. Bubble Chart
    st.subheader("12. Bubble Chart")
    if len(num_cols) >= 3:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                         size=num_cols[2], color=num_cols[0],
                         hover_name=df.columns[0])
        st.plotly_chart(fig)

else:
    st.info("Please upload a dataset to get started.")

