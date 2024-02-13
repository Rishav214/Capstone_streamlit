# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings('ignore')

# Set page title and layout
st.set_page_config(page_title="Customer Segmentation Analysis", layout="wide")

# Custom CSS for styling and animation
st.markdown("""
<style>
body, * header, #sidebar, footer {
    color: #fff;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    font-family: 'Arial', sans-serif;
    background-size: 200% 200%;
    animation: gradient 5s ease infinite;
    border-radius: 15px;
}

* header, #sidebar {
    text-align: center;
    padding: 20px;
}

footer {
    text-align: center;
    padding: 10px;
    position: fixed;
    bottom: 0;
    width: 100%;
    font-size: 12px;
}

@keyframes gradient {
    0% {background-position: 100% 0%;}
    50% {background-position: 0% 100%;}
    100% {background-position: 100% 0%;}
}

body {
    background-image: linear-gradient(45deg, #4f8bf9, #8bf94f, #f94f8b, #f9bf4f, #4f8bf9, #8bf94f);
}

* header {
    background-image: linear-gradient(45deg, #8bf94f, #f94f8b, #f9bf4f, #4f8bf9, #8bf94f, #f94f8b);
}

#sidebar {
    background-image: linear-gradient(45deg, #f94f8b, #f9bf4f, #4f8bf9, #8bf94f, #f94f8b, #f9bf4f);
}

footer {
    background-image: linear-gradient(45deg, #f9bf4f, #4f8bf9, #8bf94f, #f94f8b, #f9bf4f, #4f8bf9);
}
</style>
""", unsafe_allow_html=True)

# Rest of the code remains the same

# Business Insights
diamond_insight = """
- Diamond Tier - High Total Income
People who earn more are very likely to invest in long term savings such as houses, bonds, retirement, emergency fund, etc. 
Long term schemes, housing loans, retirement schemes, etc. 
They also prefer to invest in stocks via mutual funds.
"""

platinum_insight = """
- Platinum Tier - High Credit Score
High transaction frequency but low value. 
Into long term investments such as small value housing loans or vehicle loans and we can offer them low-value loans. 
Relatively less risk-tolerant, can offer them long term mutual funds with lower per month payment.
"""

gold_insight = """
- Gold Tier - Low Credit Score
They are targets for long term investment but the chances of success will depend on the minimum investment required.
Income Invested is relatively higher, can be targeted for stocks and mutual funds.
"""

silver_insight = """
- Silver Tier - Low Total Income
High transaction frequency but low value
Into long term investments such as small value housing loans or vehicle loans and we can offer them low-value loans. 
Relatively less risk-tolerant, can offer them long term mutual funds with lower per month payment.
"""

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
st.sidebar.markdown("<div id='sidebar'><h2>Options</h2></div>", unsafe_allow_html=True)

# Add a selectbox for the tier
tier = st.sidebar.selectbox('Select a tier', ['Diamond', 'Platinum', 'Gold', 'Silver'])

# Display different options based on the selected tier
trend = None
selected_features = []

if tier == 'Diamond':
    trend = st.sidebar.selectbox('Select a trend', ['High Earners'])
    selected_features = ['TotalIncome', 'AvgTransactionAmount', 'TransactionFrequency']
elif tier == 'Platinum':
    trend = st.sidebar.selectbox('Select a trend', ['High Credit Score'])
    selected_features = ['CreditScore', 'Occupation', 'SpendingOnTravel']
elif tier == 'Gold':
    trend = st.sidebar.selectbox('Select a trend', ['Low Earners'])
    selected_features = ['TotalIncome', 'SpendingOnGroceries', 'RetirementSaving']
elif tier == 'Silver':
    trend = st.sidebar.selectbox('Select a trend', ['Low Credit Score'])
    selected_features = ['CreditScore', 'RetirementSaving', 'SpendingOnEntertainment']

# Display selected variables
st.sidebar.markdown(f"<h3>Selected Variables for Trend '{trend}':</h3>{', '.join(selected_features)}", unsafe_allow_html=True)

# Initial Data Insights
st.markdown("<main><h2>Initial Data Insights:</h2></main>", unsafe_allow_html=True)
st.markdown("<main>Explore some initial insights from the dataset here, such as summary statistics or a few sample rows:</main>", unsafe_allow_html=True)
st.write(df.describe())

# Display histograms for each numerical column
st.markdown("<main><h2>Data Distributions:</h2></main>", unsafe_allow_html=True)
for col in num_cols:
    fig = px.histogram(df, x=col, nbins=50, title=f'{col} Distribution', color_discrete_sequence=['indianred'])
    fig.update_layout(height=400, width=600)
    st.plotly_chart(fig)

# Display box plots for each numerical column
st.markdown("<main><h2>Box Plots for Numerical Columns:</h2></main>", unsafe_allow_html=True)
for col in num_cols:
    fig = px.box(df, y=col, title=f'{col} Box Plot', color_discrete_sequence=['indianred'])
    fig.update_layout(height=400, width=600)
    st.plotly_chart(fig)

# Display a correlation heatmap
st.markdown("<main><h2>Correlation Heatmap:</h2></main>", unsafe_allow_html=True)
corr = df[num_cols].corr()
fig = go.Figure(data=go.Heatmap(z=corr, x=num_cols, y=num_cols, colorscale='Viridis', zmin=-1, zmax=1))
fig.update_layout(height=600, width=800)
st.plotly_chart(fig)

# Button for training the dataset
if st.sidebar.button('Train the dataset'):
    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[selected_features])
    st.session_state['trained'] = True  # Set the 'trained' state to True

    # Save the cluster centroids
    st.session_state['centroids'] = kmeans.cluster_centers_

# Display the visualization using Plotly
if st.sidebar.button('View visualization'):
    if 'trained' in st.session_state and st.session_state['trained']:
        # PCA
        pca = PCA(n_components=3)  # Use 3 components for 3D plot
        df[['PC1', 'PC2', 'PC3']] = pca.fit_transform(df[selected_features])

        # Transform the cluster centroids
        centroids = pca.transform(st.session_state['centroids'])
        centroids_df = pd.DataFrame(centroids, columns=['PC1', 'PC2', 'PC3'])
        centroids_df['Cluster'] = centroids_df.index

        # 3D Scatter plot with Plotly
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Cluster', title='Customer Segments',
                            labels={'PC1': selected_features[0], 'PC2': selected_features[1], 'PC3': selected_features[2]},
                            opacity=0.8, size_max=10, color_continuous_scale='Rainbow', animation_frame='Cluster')

        # Add the cluster centroids to the plot
        fig.add_trace(go.Scatter3d(x=centroids_df['PC1'], y=centroids_df['PC2'], z=centroids_df['PC3'],
                                   mode='markers', marker=dict(size=10, color='black'), name='Centroids'))

        st.plotly_chart(fig)

        # Inference
        st.markdown("<main><h2>Inference:</h2></main>", unsafe_allow_html=True)
        st.markdown(f"<main>The 3D visualization displays customer segments based on the selected trend '<strong>{trend}</strong>'.</main>", unsafe_allow_html=True)
        st.markdown(f"<main>Each cluster represents a group of customers with similar characteristics in terms of {', '.join(selected_features)}.</main>", unsafe_allow_html=True)
        st.markdown("<main>This information can be used to tailor marketing strategies, improve sales, and understand customer behavior.</main>", unsafe_allow_html=True)

        # Business Insights based on Clusters
        # st.markdown("<main><h2>Business Insights:</h2></main>", unsafe_allow_html=True)

        # Inference for Clusters
        st.markdown("<main><h3>Cluster Insights:</h3></main>", unsafe_allow_html=True)
        for cluster_num in range(4):
            st.markdown(f"<main><h4>Cluster {cluster_num}:</h4></main>", unsafe_allow_html=True)

            # Calculate the mean or mode of the features in the cluster
            cluster_data = df[df['Cluster'] == cluster_num]
            cluster_insights = cluster_data[selected_features].mean()  # or .mode()

            # Display the insights in a table
            st.table(cluster_insights.to_frame('Mean Value'))

        # Display Business Insights based on Tiers
        st.markdown("<main><h3>Tier-Based Business Insights:</h3></main>", unsafe_allow_html=True)
        if tier == 'Diamond':
            st.markdown(diamond_insight, unsafe_allow_html=True)
        elif tier == 'Platinum':
            st.markdown(platinum_insight, unsafe_allow_html=True)
        elif tier == 'Gold':
            st.markdown(gold_insight, unsafe_allow_html=True)
        elif tier == 'Silver':
            st.markdown(silver_insight, unsafe_allow_html=True)

    else:
        st.sidebar.error('Please train the dataset first.')

# Footer
st.sidebar.markdown("<footer><hr><p style='font-size: 12px;'>Developed by Team 10</p></footer>", unsafe_allow_html=True)
