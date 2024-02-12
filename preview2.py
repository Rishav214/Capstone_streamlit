# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Set page title and layout
st.set_page_config(page_title="Customer Segmentation Analysis", layout="wide")

# Custom CSS for styling and animation
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4f8bf9;
    font-family: 'Arial', sans-serif;
    animation: bodyFadeIn 2s ease;
}

header {
    text-align: center;
    padding: 20px;
    background-color: #4f8bf9;
    animation: headerFadeIn 2s ease;
}

#sidebar {
    padding: 20px;
    background-color: #4f8bf9;
    animation: sidebarFadeIn 2s ease;
}

main {
    padding: 20px;
    animation: mainFadeInUp 2s ease;
}

footer {
    text-align: center;
    padding: 10px;
    background-color: #4f8bf9;
    position: fixed;
    bottom: 0;
    width: 100%;
    font-size: 12px;
    animation: footerFadeIn 2s ease;
}

@keyframes bodyFadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes headerFadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes sidebarFadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes mainFadeInUp {
    from {
        transform: translateY(20px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes footerFadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<header><h1>Customer Segmentation Analysis</h1></header>", unsafe_allow_html=True)

# Load data
@st.cache_resource()  # Use @st.cache() decorator for caching
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Rishav214/K-Means-Finance/main/data.csv')
    return df.copy()

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
# st.sidebar.markdown("<div id='sidebar'><h2>Options</h2></div>", unsafe_allow_html=True)
# trend = st.sidebar.selectbox('Select a trend', ['Marketing', 'Sales', 'Age', 'Investment', 'Spending'])
# if trend == 'Marketing':
#     selected_features = ['AvgTransactionAmount', 'TransactionFrequency', 'Age', 'Gender']  
# elif trend == 'Sales':
#     selected_features = ['SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel']
# elif trend == 'Age':
#     selected_features = ['Age', 'TotalIncome', 'CreditScore'] 
# elif trend == 'Investment':
#     selected_features = ['IncomeInvested', 'PreferredInvestmentType', 'InvestmentDuration', 'RiskTolerance'] 
# elif trend == 'Spending':
#     selected_features = ['SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel', 'AvgTransactionAmount']

# # Display selected variables
# st.sidebar.markdown(f"<h3>Selected Variables for Trend '{trend}':</h3>{', '.join(selected_features)}", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<div id='sidebar'><h2>Options</h2></div>", unsafe_allow_html=True)

# Add a selectbox for the tier
tier = st.sidebar.selectbox('Select a tier', ['Diamond', 'Platinum', 'Gold', 'Silver'])

# Display different options based on the selected tier
if tier == 'Diamond':
    trend = st.sidebar.selectbox('Select a trend', ['Marketing', 'Sales', 'Age', 'Investment', 'Spending'])
elif tier == 'Platinum':
    trend = st.sidebar.selectbox('Select a trend', ['Marketing', 'Sales', 'Age'])
elif tier == 'Gold':
    trend = st.sidebar.selectbox('Select a trend', ['Marketing', 'Sales'])
elif tier == 'Silver':
    trend = st.sidebar.selectbox('Select a trend', ['Marketing'])

# Determine the selected features based on the selected trend
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
st.sidebar.markdown(f"<h3>Selected Variables for Trend '{trend}':</h3>{', '.join(selected_features)}", unsafe_allow_html=True)

# Initial data insights
st.markdown("<main><h2>Initial Data Insights:</h2></main>", unsafe_allow_html=True)
st.markdown("<main>Explore some initial insights from the dataset here, such as summary statistics or a few sample rows:</main>", unsafe_allow_html=True)
st.write(df.describe())

# Button for training the dataset
if st.sidebar.button('Train the dataset'):
    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
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
                            opacity=0.8, size_max=10, color_continuous_scale='Rainbow', animation_frame='Cluster')
        st.plotly_chart(fig)

        # Inference
        st.markdown("<main><h2>Inference:</h2></main>", unsafe_allow_html=True)
        st.markdown(f"<main>The 3D visualization displays customer segments based on the selected trend '<strong>{trend}</strong>'.</main>", unsafe_allow_html=True)
        st.markdown(f"<main>Each cluster represents a group of customers with similar characteristics in terms of {', '.join(selected_features)}.</main>", unsafe_allow_html=True)
        st.markdown("<main>This information can be used to tailor marketing strategies, improve sales, and understand customer behavior.</main>", unsafe_allow_html=True)

        # Business Insights based on Clusters
        st.markdown("<main><h2>Business Insights:</h2></main>", unsafe_allow_html=True)

        # Inference for Clusters
        st.markdown("<main><h3>Cluster Insights:</h3></main>", unsafe_allow_html=True)
        for cluster_num in range(4):
            st.markdown(f"<main><h4>Cluster {cluster_num}:</h4></main>", unsafe_allow_html=True)
            
            # Calculate the mean or mode of the features in the cluster
            cluster_data = df[df['Cluster'] == cluster_num]
            cluster_insights = cluster_data[selected_features].mean()  # or .mode()

            # Display the insights in a table
            st.table(cluster_insights.to_frame('Mean Value'))
    else:
        st.sidebar.error('Please train the dataset first.')

# Footer
st.sidebar.markdown("<footer><hr><p style='font-size: 12px;'>Developed by Team 10</p></footer>", unsafe_allow_html=True)
