import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------------------------------
# Setup
# -------------------------------------------
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="centered")

st.title("🩺 Heart Stroke Prediction App")
st.markdown("Built by **Balasubramanya C K** using Machine Learning (KNN Model)")

# Load required files safely
try:
    model = joblib.load("KNN_Heart.pkl")
    scaler = joblib.load("Scaler.pkl")
    expected_columns = joblib.load("Columns.pkl")
except FileNotFoundError:
    st.error("❌ Model, Scaler, or Columns file not found! Please check your setup.")
    st.stop()

# -------------------------------------------
# Sidebar Inputs
# -------------------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------------------------
# Prediction Logic
# -------------------------------------------
if st.button("🔍 Predict Risk"):
    # Raw input dict
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale matching features only
    try:
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
            input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
        else:
            numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    except Exception as e:
        st.error(f"⚠️ Error during scaling: {e}")
        st.stop()

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()

    # -------------------------------------------
    # Display Result
    # -------------------------------------------
    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        risk_text = "⚠️ High Risk of Heart Disease"
        if prob is not None:
            risk_text += f" — Confidence: {prob * 100:.1f}%"
        st.error(risk_text)
        st.markdown("💡 **Tip:** Please consult a cardiologist for further diagnosis.")
    else:
        risk_text = "✅ Low Risk of Heart Disease"
        if prob is not None:
            risk_text += f" — Confidence: {(1 - prob) * 100:.1f}%"
        st.success(risk_text)
        st.markdown("👍 **Stay healthy! Maintain regular exercise and a balanced diet.**")

    # -------------------------------------------
    # Generate Report (PDF + CSV)
    # -------------------------------------------
    report_df = pd.DataFrame({
        "Parameter": list(raw_input.keys()),
        "Value": list(raw_input.values())
    })
    report_df.loc[len(report_df)] = ["Prediction", "High Risk" if prediction == 1 else "Low Risk"]
    if prob is not None:
        report_df.loc[len(report_df)] = ["Confidence", f"{prob * 100:.1f}%" if prediction == 1 else f"{(1 - prob) * 100:.1f}%"]

    # ----- CSV -----
    csv_data = report_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report as CSV", data=csv_data, file_name="Heart_Risk_Report.csv", mime="text/csv")

    # ----- PDF -----
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(200, 760, "Heart Stroke Prediction Report")
    pdf.setFont("Helvetica", 12)
    y = 720
    for index, row in report_df.iterrows():
        pdf.drawString(50, y, f"{row['Parameter']}: {row['Value']}")
        y -= 20
        if y < 50:
            pdf.showPage()
            y = 750
    pdf.save()
    buffer.seek(0)

    st.download_button("📄 Download Report as PDF", data=buffer, file_name="Heart_Risk_Report.pdf", mime="application/pdf")

# -------------------------------------------
# Footer
# -------------------------------------------
st.markdown("---")
st.caption("© 2025 Balasubramanya C K | Machine Learning Demo App")
