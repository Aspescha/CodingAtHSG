"""
app.py
------
Streamlit web application for the Luxury Watch Price Estimator.

Run with:
    streamlit run app.py

The app loads a pre-trained WatchPriceModel and allows users to input
watch specifications to get an estimated market price.

Note: The model must be trained and saved first by running the notebook
and executing model.save("models/watch_price_model.joblib").
"""

import streamlit as st
import os
from src.model import WatchPriceModel

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Luxury Watch Price Estimator",
    page_icon="⌚",
    layout="centered",
)

# ------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ------------------------------------------------------------------

MODEL_PATH = "models/watch_price_model.joblib"

@st.cache_resource
def load_model() -> WatchPriceModel:
    """Load the trained model from disk."""
    model = WatchPriceModel()
    model.load(MODEL_PATH)
    return model

# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

st.title("⌚ Luxury Watch Price Estimator")
st.markdown(
    "Enter the specifications of a watch below to get an estimated "
    "market value based on our trained XGBoost model."
)

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found. Please run the notebook first and save "
        "the model using `model.save('models/watch_price_model.joblib')`."
    )
    st.stop()

model = load_model()
options = model.get_valid_options()

st.divider()

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        options=["Unknown"] + options["brands"],
        index=0,
    )

    # Filter models by selected brand
    if brand != "Unknown":
        brand_models = model._watch_data[
            model._watch_data["brand"] == brand
        ]["model"].value_counts().index.tolist()
    else:
        brand_models = options["models"]

    model_name = st.selectbox(
        "Model",
        options=["Unknown"] + brand_models,
        index=0,
    )

    case_material = st.selectbox(
        "Case Material",
        options=["Unknown"] + options["cases"],
        index=0,
    )

    condition = st.selectbox(
        "Condition",
        options=["Unknown"] + options["conditions"],
        index=0,
    )

with col2:
    movement = st.selectbox(
        "Movement",
        options=["Unknown"] + options["movements"],
        index=0,
    )

    bracelet_material = st.selectbox(
        "Bracelet Material",
        options=["Unknown"] + options["bracelets"],
        index=0,
    )

    sex = st.selectbox(
        "Gender",
        options=["Unknown"] + options["sexes"],
        index=0,
    )

    size = st.number_input(
        f"Case Size (mm) — typical range: "
        f"{options['size_range'][0]}–{options['size_range'][1]} mm",
        min_value=float(options["size_range"][0]),
        max_value=float(options["size_range"][1]),
        value=40.0,
        step=0.5,
    )

    yop = st.number_input(
        f"Year of Production — "
        f"{options['yop_range'][0]}–{options['yop_range'][1]}",
        min_value=int(options["yop_range"][0]),
        max_value=int(options["yop_range"][1]),
        value=2015,
        step=1,
    )

st.divider()

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

if st.button("Estimate Price", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        estimated_price = model.predict(
            brand=brand,
            model_name=model_name,
            case_material=case_material,
            condition=condition,
            movement=movement,
            bracelet_material=bracelet_material,
            sex=sex,
            size=size if size else None,
            yop=yop if yop else None,
        )

    st.success(f"### Estimated Market Value: ${estimated_price:,.2f}")
    st.caption(
        "This estimate is based on a model trained on publicly listed watch "
        "prices. It is most reliable for liquid, mid-range watches and may "
        "not generalize well to rare vintage or ultra-high-end collectibles."
    )