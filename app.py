import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load the trained model and label encoders
model = joblib.load('property_price_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Helper function to get the classes or provide a default
def get_encoder_classes(column_name):
    if column_name in label_encoders:
        return label_encoders[column_name].classes_
    else:
        st.warning(f"Encoder for {column_name} not found. Using default values.")
        return ['Unknown']

# Function to make predictions
def predict_price(features):
    return model.predict(np.array(features).reshape(1, -1))[0]

# Streamlit app
st.title('Property Price Prediction')

# Sidebar form for user input
st.sidebar.title('Input Features')
location = st.sidebar.selectbox('Location', get_encoder_classes('location'))
status = st.sidebar.selectbox('Status', get_encoder_classes('Status'))
transaction = st.sidebar.selectbox('Transaction', get_encoder_classes('Transaction'))
furnishing = st.sidebar.selectbox('Furnishing', get_encoder_classes('Furnishing'))
facing = st.sidebar.selectbox('Facing', get_encoder_classes('facing'))
overlooking = st.sidebar.selectbox('Overlooking', get_encoder_classes('overlooking'))
society = st.sidebar.selectbox('Society', get_encoder_classes('Society'))
ownership = st.sidebar.selectbox('Ownership', get_encoder_classes('Ownership'))
car_parking = st.sidebar.selectbox('Car Parking', get_encoder_classes('Car Parking'))
floor = st.sidebar.selectbox('Floor', get_encoder_classes('Floor'))
carpet_area = st.sidebar.number_input('Carpet Area (in sqft)', min_value=0.0, step=1.0)
bathroom = st.sidebar.number_input('Bathroom', min_value=1.0, step=1.0)
balcony = st.sidebar.number_input('Balcony', min_value=0.0, step=1.0)
price_per_sqft = st.sidebar.number_input('Price per sq ft (in rupees)', min_value=0.0, step=1.0)

# Transform input features using encoders if available, otherwise use default value
def transform_feature(column_name, value):
    if column_name in label_encoders:
        return label_encoders[column_name].transform([value])[0]
    else:
        return 0  # Default value if encoder is not available

# Prepare the input features for prediction
features = [
    transform_feature('location', location),
    transform_feature('Status', status),
    transform_feature('Transaction', transaction),
    transform_feature('Furnishing', furnishing),
    transform_feature('facing', facing),
    transform_feature('overlooking', overlooking),
    transform_feature('Society', society),
    transform_feature('Ownership', ownership),
    transform_feature('Floor', floor),
    carpet_area,
    bathroom,
    balcony,
    transform_feature('Car Parking', car_parking),
    price_per_sqft
]

# Make prediction
if st.sidebar.button('Predict Price'):
    price = predict_price(features)
    st.sidebar.write(f'The predicted price of the property is {price:.2f} INR')

# Load data for visualization
df = pd.read_csv('house_prices.csv')

# Ensure 'Carpet Area' and 'Price (in rupees)' are numeric, handle non-numeric values
if df['Carpet Area'].dtype == 'object':
    df['Carpet Area'] = pd.to_numeric(df['Carpet Area'].str.extract('(\d+)')[0], errors='coerce')
else:
    df['Carpet Area'] = pd.to_numeric(df['Carpet Area'], errors='coerce')

if df['Price (in rupees)'].dtype == 'object':
    df['Price (in rupees)'] = pd.to_numeric(df['Price (in rupees)'].str.extract('(\d+)')[0], errors='coerce')
else:
    df['Price (in rupees)'] = pd.to_numeric(df['Price (in rupees)'], errors='coerce')

# Drop rows with missing 'Carpet Area' or 'Price (in rupees)'
df.dropna(subset=['Carpet Area', 'Price (in rupees)'], inplace=True)

# Location by Price
fig_location_price = px.bar(df.groupby('location')['Price (in rupees)'].mean().reset_index(), x='location', y='Price (in rupees)', title='Average Price by Location')
st.plotly_chart(fig_location_price)

# Carpet Area by Price
fig_carpet_price = px.scatter(df, x='Carpet Area', y='Price (in rupees)', title='Carpet Area vs Price')
st.plotly_chart(fig_carpet_price)

# Location by Status
fig_location_status = px.bar(df.groupby(['location', 'Status']).size().reset_index(name='Count'), x='location', y='Count', color='Status', title='Number of Properties by Location and Status')
st.plotly_chart(fig_location_status)

# Transaction by Location by Furnishing
fig_transaction_location_furnishing = px.bar(df.groupby(['Transaction', 'location', 'Furnishing']).size().reset_index(name='Count'), x='location', y='Count', color='Transaction', facet_col='Furnishing', title='Transaction by Location and Furnishing')
st.plotly_chart(fig_transaction_location_furnishing)

# Furnishing by Location by Price
fig_furnishing_location_price = px.box(df, x='location', y='Price (in rupees)', color='Furnishing', title='Price by Location and Furnishing')
st.plotly_chart(fig_furnishing_location_price)

# Location by Carpet Area
fig_location_carpet = px.box(df, x='location', y='Carpet Area', title='Carpet Area by Location')
st.plotly_chart(fig_location_carpet)
