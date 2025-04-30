
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

# Use a nicer Matplotlib style
matplotlib.style.use('seaborn-v0_8-darkgrid')

# Load the model and encoders
model = joblib.load('house_price_model.pkl')
le_furnished = joblib.load('le_furnished.pkl')
le_property = joblib.load('le_property.pkl')
le_location = joblib.load('le_location.pkl')

# Streamlit App Config
st.set_page_config(page_title="Goregaon House Price Predictor", page_icon="🏡", layout="wide")

def format_inr_price(price):
    if price >= 1e7:
        return f"{price/1e7:.2f} Cr"
    else:
        return f"{price/1e5:.0f} Lakhs"

# --- Sidebar ---
st.sidebar.title("🏡 About")
st.sidebar.info(
    "This app predicts house prices in Goregaon based on your inputs and shows future price trends!"
)
st.sidebar.markdown("---")

# --- Main Page ---
st.title('✨ Goregaon House Price Predictor')
st.markdown("""
<style>
    .big-font {
        font-size:28px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Fill the details below to predict the house price:</p>', unsafe_allow_html=True)

# Input Form
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input('Area (in sqft)', min_value=100, max_value=10000, step=10, help="Enter total carpet area.")
        bhk = st.selectbox('BHK (Bedrooms)', [1, 2, 3, 4, 5])
        furnished = st.selectbox('Furnishing Status', le_furnished.classes_)
        location = st.selectbox('Location', le_location.classes_)

    with col2:
        price_per_sqft = st.number_input('Price per sqft (INR)', min_value=1000, max_value=1000000, step=100)
        property_type = st.selectbox('Property Type', le_property.classes_)

    submit_button = st.form_submit_button(label='Predict Price 💫')

# --- Prediction ---
if submit_button:
    input_data = np.array([
        area,
        bhk,
        price_per_sqft,
        le_furnished.transform([furnished])[0],
        le_property.transform([property_type])[0],
        le_location.transform([location])[0]
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    formatted_price = format_inr_price(prediction)

    st.balloons()
    st.success(f"💰 Estimated House Price: {formatted_price}")

    # Future Price Trend
    st.markdown("""<h3 style='text-align: center;'>📈 Future Price Trend (6% Annual Growth)</h3>""", unsafe_allow_html=True)

    years = np.arange(2025, 2030)
    future_prices = [prediction * (1.06)**(year-2024) for year in years]
    future_prices_formatted = [format_inr_price(p) for p in future_prices]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(years, future_prices, marker='o', color='#00cc99', linewidth=3)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Projected Price (INR)', fontsize=12)
    ax.set_title('Projected House Price Growth', fontsize=14)
    ax.grid(True)

    for i, price in enumerate(future_prices):
        ax.text(years[i], price, future_prices_formatted[i], fontsize=10, ha='center', va='bottom')

    st.pyplot(fig)

# --- Footer ---
st.markdown("""
---
<p style="text-align: center;">Made with ❤️ for Goregaon</p>
""", unsafe_allow_html=True)
