
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("bangalore_house_price_model.pkl")

# Get locations from the trained encoder
preprocessor = model.named_steps[model.steps[0][0]]

location_encoder = next(
    transformer
    for name, transformer, columns in preprocessor.transformers_
    if hasattr(transformer, "categories_")
    and "location" in columns
)

locations = sorted(
    [str(loc) for loc in location_encoder.categories_[0]],
    key=str.casefold
)

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Bengaluru House Price Predictor")
st.write("Estimate a house price using your trained ML model.")
st.divider()

location = st.selectbox(
    "Search and select a location",
    options=locations,
    index=locations.index("Whitefield")
    if "Whitefield" in locations else 0
)

col1, col2 = st.columns(2)

with col1:
    total_sqft = st.number_input(
        "Total Area (sq. ft.)",
        min_value=100,
        max_value=50000,
        value=1500,
        step=100
    )

    bhk = st.number_input(
        "Bedrooms (BHK)",
        min_value=1,
        max_value=20,
        value=3
    )

with col2:
    bath = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2
    )

if st.button("Predict House Price", type="primary"):
    house = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "bhk": [bhk]
    })

    predicted_price = float(model.predict(house)[0])

    if predicted_price > 0:
        st.success("Prediction completed!")
        st.metric(
            "Estimated House Price",
            f"₹{predicted_price:.2f} lakh"
        )
        st.write(f"Approximately ₹{predicted_price * 100000:,.0f}")
    else:
        st.warning("The predicted price is non-positive.")
