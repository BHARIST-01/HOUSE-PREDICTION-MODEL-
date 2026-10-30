
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="EstateIQ | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------

MODEL_PATH = Path(__file__).parent / "linear_regression_model.pkl"

if not MODEL_PATH.exists():
    st.error(
        "Model file not found! Please save "
        "linear_regression_model.pkl in the same folder as app.py."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #080d20 0%,
        #101b3c 55%,
        #151033 100%
    );
    color: #ffffff;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Text */
h1, h2, h3, p, label {
    color: #ffffff !important;
}

/* Hero section */
.hero {
    background: linear-gradient(
        110deg,
        #172554,
        #312e81,
        #581c87
    );
    padding: 35px;
    border-radius: 22px;
    border: 1px solid #6366f1;
    margin-bottom: 30px;
    box-shadow: 0 10px 35px rgba(79,70,229,0.25);
}

.hero-tag {
    color: #67e8f9 !important;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 3px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe !important;
    font-size: 16px;
}

/* Input cards */
.input-card {
    background: linear-gradient(
        145deg,
        #172554,
        #1e1b4b
    );
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #3730a3;
    margin-bottom: 20px;
}

/* Input labels */
div[data-testid="stNumberInput"] label {
    color: #c4b5fd !important;
    font-weight: 600;
}

/* Input fields */
div[data-testid="stNumberInput"] input {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #6366f1 !important;
    border-radius: 10px !important;
}

/* Input focus */
div[data-testid="stNumberInput"] input:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 10px rgba(34,211,238,0.25);
}

/* Button */
div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    height: 55px;
    background: linear-gradient(
        90deg,
        #06b6d4,
        #6366f1,
        #a855f7
    );
    color: white !important;
    font-weight: 800;
    font-size: 17px;
    border: none;
    border-radius: 13px;
    transition: 0.3s;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}

/* Result card */
.result-card {
    background: linear-gradient(
        135deg,
        #064e3b,
        #065f46,
        #0f766e
    );
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #34d399;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 10px 30px rgba(16,185,129,0.18);
}

.result-label {
    color: #a7f3d0 !important;
    font-size: 15px;
    letter-spacing: 2px;
    font-weight: bold;
}

.result-price {
    color: #ffffff !important;
    font-size: 42px;
    font-weight: 900;
    margin: 12px 0;
}

/* Section headings */
.section-heading {
    color: #67e8f9 !important;
    font-size: 22px;
    font-weight: 800;
    margin: 15px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 50px;
    padding: 20px;
    border-top: 1px solid #334155;
}

/* Hide Streamlit decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------
# HERO SECTION
# ---------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">ESTATEIQ • MACHINE LEARNING</div>
    <h1>🏡 House Price Predictor</h1>
    <p>
        Discover estimated housing values using our
        Linear Regression Machine Learning Model.
        Enter your housing details and generate a prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------
# INPUT SECTION
# ---------------------------------------

st.markdown(
    '<div class="section-heading">✦ Enter Housing Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Provide the following eight district-level housing features."
)

with st.form("house_prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("### 🏠 Property Characteristics")

        med_inc = st.number_input(
            "Median Income (dataset units)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        house_age = st.number_input(
            "Median House Age (years)",
            min_value=1.0,
            max_value=52.0,
            value=25.0,
            step=1.0
        )

        ave_rooms = st.number_input(
            "Average Rooms",
            min_value=0.1,
            value=5.0,
            step=0.1
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms",
            min_value=0.1,
            value=1.0,
            step=0.1
        )

    with col2:

        st.markdown("### 📍 Location & Population")

        population = st.number_input(
            "Population",
            min_value=1.0,
            value=1000.0,
            step=100.0
        )

        ave_occup = st.number_input(
            "Average Occupancy",
            min_value=0.1,
            value=3.0,
            step=0.1
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=34.05,
            step=0.01,
            format="%.4f"
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.25,
            step=0.01,
            format="%.4f"
        )

    st.write("")

    submitted = st.form_submit_button(
        "✨  PREDICT HOUSE VALUE"
    )


# ---------------------------------------
# PREDICTION SECTION
# ---------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(input_data)[0]

    predicted_dollars = prediction * 100000

    st.divider()

    st.markdown(
        '<div class="section-heading">✦ Prediction Results</div>',
        unsafe_allow_html=True
    )

    if predicted_dollars < 0:
        st.warning(
            "The model returned a negative estimate. "
            "This input may be outside the model's useful range."
        )
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                ESTIMATED MEDIAN HOUSE VALUE
            </div>
            <div class="result-price">
                ${predicted_dollars:,.0f}
            </div>
            <p style="color:#a7f3d0 !important;">
                Prediction generated successfully
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Model Prediction",
                f"${predicted_dollars:,.0f}"
            )

        with metric2:
            st.metric(
                "Prediction in Dataset Units",
                f"{prediction:.4f}"
            )

    with st.expander("View submitted input details"):
        st.dataframe(input_data, use_container_width=True)


# ---------------------------------------
# INFORMATION SECTION
# ---------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">🧠 About This Project</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    **📊 Dataset**

    California Housing Dataset
    """)

with info2:
    st.markdown("""
    **🤖 Algorithm**

    Linear Regression
    """)

with info3:
    st.markdown("""
    **⚙️ Framework**

    Python + Scikit-learn + Streamlit
    """)

st.info(
    "Important: This model predicts median house values for "
    "California districts, not individual property selling prices. "
    "It is an educational machine learning project, not a "
    "professional real-estate valuation tool."
)

# ---------------------------------------
# FOOTER
# ---------------------------------------

st.markdown("""
<div class="footer">
    ESTATEIQ | HOUSE PRICE PREDICTION PROJECT<br>
    Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
