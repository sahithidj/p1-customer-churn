import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
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
    data = pd.read_csv("notebooks/customer_churn_dataset-training-master.csv")
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


### ------ feature selection -------
st.header("Explore Churn by Selected Features")
additional_features = ["Tenure", "Total Spend", "Usage Frequency", "Last Interaction"]

selected = st.multiselect("Select feature(s) to explore churn trends:", additional_features)
st.info(" Select one or more features to visualize.")

# Show a boxplot for each selected feature
if selected:
    for feat in selected:
        if feat in data.columns:
            st.subheader(f"Churn vs {feat}")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.boxplot(data=data, x="Churn", y=feat, ax=ax)
            st.pyplot(fig)
        else:
            st.warning(f"⚠️ Feature '{feat}' not found in your dataset.")


### customer segmentation: 
st.subheader("Customer Segment Churn Prediction")

# Checkbox for different age groups
age_25 = st.checkbox("<25")
age_25_40 = st.checkbox("25-40")
age_40_60 = st.checkbox("40-60")
age_60_plus = st.checkbox("60+")

# Collect selected age groups
selected_ages = []

if age_25:
    selected_ages.append("<25")
if age_25_40:
    selected_ages.append("25-40")
if age_40_60:
    selected_ages.append("40-60")
if age_60_plus:
    selected_ages.append("60+")

# Create a new column 'age_group' based on age ranges
def categorize_age(age):
    if age < 25:
        return "<25"
    elif 25 <= age <= 40:
        return "25-40"
    elif 40 < age <= 60:
        return "40-60"
    else:
        return "60+"

# Apply the categorize_age function to the 'Age' column
data['age_group'] = data['Age'].apply(categorize_age)

# Filter the data by the selected age groups
filtered_data = data[data['age_group'].isin(selected_ages)]

# Calculate churn rates by age group
churn_by_age = filtered_data.groupby('age_group')['Churn'].mean().reset_index()

# Plot churn rates by age group
st.write("Churn Rate by Age Group")
plt.figure(figsize=(8, 6))
sns.barplot(x='age_group', y='Churn', data=churn_by_age)
plt.xlabel('Age Group')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Age Group')
st.pyplot(plt)

















