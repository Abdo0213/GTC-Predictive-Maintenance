import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"

# Required features (must match FastAPI)
REQUIRED_FEATURES = [
    "op_setting_1", "op_setting_2",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

st.title("🔧 Predictive Maintenance Dashboard")
st.write("Upload recent sensor data to forecast Remaining Useful Life (RUL).")

option = st.radio("Choose input method:", ["Upload Log File", "Manual Sensor Input"])

# 📌 File Upload Mode
if option == "Upload Log File":
    st.info(f"CSV file must contain exactly these {len(REQUIRED_FEATURES)} columns:\n\n" + ", ".join(REQUIRED_FEATURES))
    engine_id = st.number_input("Engine ID", min_value=1, step=1)
    uploaded_file = st.file_uploader("Upload sensor log (CSV or TXT)", type=["csv", "txt"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict RUL"):
            # files = {"file": uploaded_file.getvalue()}
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            data = {"engine_id": engine_id}
            # response = requests.post(f"{API_URL}/predict/file", files={"file": uploaded_file}, data=data)
            response = requests.post(f"{API_URL}/predict/file", files=files, data=data)
            if response.ok:
                result = response.json()
                st.success(f"✅ Predicted RUL: {result['predicted_RUL']:.2f} cycles")
                st.caption("⚙️ Prediction is based on the last 50 recorded sensor readings and degradation patterns.")
            else:
                st.error(f"❌ API Error: {response.json()['detail']}")

# 📌 Manual Input Mode
else:
    st.info(f"Please enter values for all {len(REQUIRED_FEATURES)} features below.")
    engine_id = st.number_input("Engine ID", min_value=1, step=1)

    values = {}
    cols = st.columns(3)  # spread inputs in 3 columns
    for idx, feature in enumerate(REQUIRED_FEATURES):
        with cols[idx % 3]:
            values[feature] = st.number_input(feature, value=0.0, format="%.3f")

    if st.button("Predict RUL"):
        df = pd.DataFrame([values])
        payload = {"engine_id": engine_id, "readings": df.to_dict(orient="records")}
        response = requests.post(f"{API_URL}/predict/sensors", json=payload)
        if response.ok:
            result = response.json()
            st.success(f"✅ Predicted RUL: {result['predicted_RUL']:.2f} cycles")
            st.caption("🔍 Lower RUL means the engine is closer to failure and may need proactive maintenance soon.")
        else:
            st.error(f"❌ API Error: {response.json()['detail']}")
