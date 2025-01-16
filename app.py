import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Simplified AI: Automated EDA Tool")

# Function to summarize data
def summarize_data(df):
    return {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Summary Statistics": df.describe().to_dict()
    }

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

# Function to detect outliers using IQR
def detect_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Function to visualize data distribution
def visualize_distribution(df, column):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=30, color="blue")
    plt.title(f"Distribution Plot for {column}")
    st.pyplot(plt)

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    summary = summarize_data(df)
    for key, value in summary.items():
        st.write(f"**{key}:**")
        st.write(value)

    st.subheader("Correlation Heatmap")
    plot_correlation_heatmap(df)

    # Numerical column selection
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) > 0:
        column = st.selectbox("Select a column for detailed analysis", numerical_columns)

        if column:
            st.subheader(f"Outlier Detection for {column}")
            outliers = detect_outliers(df, column)
            st.write(f"Number of outliers in `{column}`: {len(outliers)}")
            st.write(outliers)

            st.subheader(f"Distribution Plot for {column}")
            visualize_distribution(df, column)
    else:
        st.warning("No numerical columns found for analysis!")
