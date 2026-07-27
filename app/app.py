import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../model/model.pkl")

le_division = joblib.load("../model/le_division.pkl")
le_region = joblib.load("../model/le_region.pkl")
le_ship = joblib.load("../model/le_ship.pkl")
le_factory = joblib.load("../model/le_factory.pkl")

st.set_page_config(
    page_title="Factory Recommendation System",
    page_icon="🏭",
    layout="wide"
)
st.markdown("""
<style>

/* Reduce the size of metric values (100%, Random Forest, 7, 4) */
[data-testid="stMetricValue"] > div {
    font-size: 28px !important;
    font-weight: 600 !important;
}

/* Reduce the label size */
[data-testid="stMetricLabel"] {
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* Optional: reduce spacing */
[data-testid="metric-container"] {
    padding: 8px 10px !important;
}

</style>
""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])

with col1:
    st.image("../images/logo.png", width=180)

with col2:
    st.title("Factory Reallocation Recommendation System")
    st.write("AI Powered Logistics Decision Support Dashboard")
    k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "🎯 Accuracy",
    "100%"
)

k2.metric(
    "⚙️ Algorithm",
    "Random Forest"
)

k3.metric(
    "📊 Features",
    "7"
)

k4.metric(
    "🏭 Factories",
    "4"
)
st.markdown("## 📦 Order Details")

division = st.selectbox("Division", le_division.classes_)

region = st.selectbox("Region", le_region.classes_)

ship_mode = st.selectbox("Ship Mode", le_ship.classes_)

sales = st.number_input("Sales", value=1000.0)

units = st.number_input("Units", value=10)

cost = st.number_input("Cost", value=800.0)

lead_time = st.slider(
    "Lead Time",
    800,
    1000,
    900
)
predict = st.button(
    "🚀 Recommend Best Factory",
    use_container_width=True
)
if predict:

    input_df = pd.DataFrame({
        "Division": [le_division.transform([division])[0]],
        "Region": [le_region.transform([region])[0]],
        "Ship Mode": [le_ship.transform([ship_mode])[0]],
        "Sales": [sales],
        "Units": [units],
        "Cost": [cost],
        "Lead Time": [lead_time]
    })

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df).max() * 100
    st.metric("Prediction Confidence", f"{prob:.2f}%")
    factory = le_factory.inverse_transform(prediction)[0]

    st.success(f"""
## 🏭 Recommended Factory

### {factory}
""")

    st.info("This factory is predicted to provide the best operational efficiency.")

    summary = pd.DataFrame({
        "Field": [
            "Division",
            "Region",
            "Ship Mode",
            "Sales",
            "Units",
            "Cost",
            "Lead Time",
            "Factory"
        ],
        "Value": [
            division,
            region,
            ship_mode,
            sales,
            units,
            cost,
            lead_time,
            factory
        ]
    })

    st.subheader("Prediction Summary")
    st.table(summary)

st.sidebar.title("🏭 About")

st.sidebar.markdown("""
### Factory Reallocation Recommendation System

This application predicts the most suitable factory using a trained Machine Learning model.

### Technologies Used

- 🐍 Python
- 🚀 Streamlit
- 📊 Pandas
- 🤖 Scikit-Learn
- 🌲 Random Forest
- 📈 Plotly
""")

st.markdown("---")

st.markdown("---")

st.markdown(
    "<center><b>Developed by Radhika UR</b><br>Data Analytics Internship Project</center>",
    unsafe_allow_html=True
)
