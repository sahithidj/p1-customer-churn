import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# App layout setup
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("customer_churn_dataset-training-master.csv")
    return data

data = load_data()

# Preprocess data
data['Gender'] = data['Gender'].str.strip().str.capitalize()
data['Gender'] = data['Gender'].map({"Female": 0, "Male": 1})



# Dashboard Title
st.title("Customer Churn Dashboard")
st.markdown("Gain insights into factors that influence churn such as subscription type, support calls, payment delays, and more.")

# KPIs Section
st.markdown("### Quick Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(data)}")
col2.metric("Churn Rate", f"{data['Churn'].mean() * 100:.2f}%")
col3.metric("Avg. Support Calls", f"{data['Support Calls'].mean():.2f}")

# Data Overview Section
with st.expander("Raw Data Overview"):
    st.write(data.head())
with st.expander("Additional Information"):
    st.write(data.describe()) 
    

# Visualizations
st.markdown("### Insights")

# First row of charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Gender vs Churn")
    fig = px.histogram(data, x="Gender", color="Churn",
                       barmode="group", labels={"Gender": "Gender (0 = Female, 1 = Male)"}, title="Churn by Gender")
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Subscription Type vs Churn")
    fig = px.histogram(data, x="Subscription Type", color="Churn",
                       barmode="group", title="Churn by Subscription Type")
    st.plotly_chart(fig, use_container_width=True)

# Second row of charts
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("Payment Delay vs Avg Churn Rate")
    churn_by_delay = data.groupby("Payment Delay")["Churn"].mean().reset_index()
    fig = px.line(churn_by_delay, x="Payment Delay", y="Churn", markers=True,
                  title="Churn Rate vs Payment Delay", labels={"Churn": "Avg Churn Rate"})
    st.plotly_chart(fig, use_container_width=True)

with chart_col4:
    st.subheader("Average Churn Rate by Number of Support Calls")
    churn_by_calls = data.groupby("Support Calls")["Churn"].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(churn_by_calls.index, churn_by_calls.values, marker='o', color='blue')
    ax.set_title("Support Calls vs Churn Rate")
    ax.set_xlabel("Support Calls")
    ax.set_ylabel("Average Churn Rate")
    ax.grid(True)
    st.pyplot(fig)

















