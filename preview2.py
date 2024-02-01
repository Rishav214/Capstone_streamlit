# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set page title and layout
st.set_page_config(page_title="Customer Segmentation Analysis", layout="wide")

# Header
st.header("Customer Segmentation Analysis")

# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/sgx-saksham/Predictive-Analysis-streamlit/main/Banking_Customer_Data.csv')
    return df.copy()  # Return a copy of the DataFrame to avoid mutation issues

df = load_data()

# Convert boolean columns to int
bool_cols = ['RetirementSaving', 'HouseBuying', 'EmergencyFund']
df[bool_cols] = df[bool_cols].astype(int)

# Label encode categorical columns
le = LabelEncoder()
cat_cols = ['Gender', 'Occupation', 'PreferredInvestmentType', 'InvestmentDuration', 'RiskTolerance']
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

# Standardize numerical columns
scaler = StandardScaler()
num_cols = ['Age', 'AvgTransactionAmount', 'TransactionFrequency', 'SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel', 'IncomeInvested', 'CreditScore', 'TotalIncome']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Sidebar
st.sidebar.title("Options")
trend = st.sidebar.selectbox('Select a trend', ['Marketing', 'Sales', 'Age', 'Investment', 'Spending'])
if trend == 'Marketing':
    selected_features = ['AvgTransactionAmount', 'TransactionFrequency', 'Age', 'Gender']  
elif trend == 'Sales':
    selected_features = ['SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel']
elif trend == 'Age':
    selected_features = ['Age', 'TotalIncome', 'CreditScore'] 
elif trend == 'Investment':
    selected_features = ['IncomeInvested', 'PreferredInvestmentType', 'InvestmentDuration', 'RiskTolerance'] 
elif trend == 'Spending':
    selected_features = ['SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel', 'AvgTransactionAmount']

# Display selected variables
st.sidebar.markdown(f"**Selected Variables for Trend '{trend}':** {', '.join(selected_features)}")

# Button for training the dataset
if st.sidebar.button('Train the dataset'):
    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[selected_features])
    st.session_state['trained'] = True  # Set the 'trained' state to True

# Display the visualization using Plotly
if st.sidebar.button('View visualization'):
    if 'trained' in st.session_state and st.session_state['trained']:
        # PCA
        pca = PCA(n_components=3)  # Use 3 components for 3D plot
        df[['PC1', 'PC2', 'PC3']] = pca.fit_transform(df[selected_features])

        # 3D Scatter plot with Plotly
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Cluster', title='Customer Segments',
                            labels={'PC1': selected_features[0], 'PC2': selected_features[1], 'PC3': selected_features[2]},
                            opacity=0.8, size_max=10, color_continuous_scale='viridis')
        st.plotly_chart(fig)

        # Inference
        st.markdown("**Inference:**")
        st.markdown(f"The 3D visualization displays customer segments based on the selected trend '**{trend}**'.")
        st.markdown(f"Each cluster represents a group of customers with similar characteristics in terms of {', '.join(selected_features)}.")
        st.markdown("This information can be used to tailor marketing strategies, improve sales, and understand customer behavior.")
    else:
        st.sidebar.error('Please train the dataset first.')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Rishav")


