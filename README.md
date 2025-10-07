# ❤️ Heart Stroke Prediction App

**Predict your risk of heart disease using a Machine Learning model**

This Streamlit app allows users to input their health parameters and predicts the likelihood of heart disease using a trained **K-Nearest Neighbors (KNN) model**. It also provides a downloadable report (CSV or PDF) of the prediction and confidence score.

---

## 🛠️ Features

* **Interactive Inputs**: Age, sex, chest pain type, blood pressure, cholesterol, ECG results, and more.
* **Real-Time Prediction**: Immediate heart disease risk prediction with confidence score.
* **Downloadable Reports**: Save your inputs and prediction as **CSV** or **PDF**.
* **User-Friendly Interface**: Built with Streamlit, making it easy to use for anyone.

---

## 📊 How It Works

1. User enters health details via sliders, dropdowns, and number inputs.
2. The app pre-processes the input to match the features used during model training.
3. Numeric inputs are scaled using the same scaler as the training dataset.
4. The trained KNN model predicts the risk of heart disease.
5. The app displays:

   * ✅ Low Risk of Heart Disease
   * ⚠️ High Risk of Heart Disease
   * Confidence percentage
6. Users can download a detailed report containing all inputs and results.

---

## 🧠 Technology Stack

* **Programming Language**: Python
* **Web Framework**: Streamlit
* **Machine Learning**: scikit-learn (KNN model)
* **Data Processing**: Pandas, NumPy
* **File Handling**: joblib (to load model and scaler), ReportLab (for PDF reports)

---

## 📁 Files in This Repository

* `app.py` — Main Streamlit app
* `KNN_Heart.pkl` — Trained KNN model
* `Scaler.pkl` — StandardScaler used to scale input data
* `Columns.pkl` — Expected feature columns for the model
* `requirements.txt` — Python dependencies

---

## 🚀 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

4. Open the local URL (usually `http://localhost:8501`) in your browser.

---

## 🌐 Live Demo

Access the app online:
[https://heartriskproject-wkp4fdas8bz2dbpnrkpnfp.streamlit.app/](https://heartriskproject-wkp4fdas8bz2dbpnrkpnfp.streamlit.app/)

---

## 📢 Future Enhancements

* Add user authentication for personalized experience.
* Include historical data logging for repeated predictions.
* Improve the UI with graphs, charts, and better visual design.
* Periodic model retraining with new datasets to improve accuracy.

---

## ⚖️ Disclaimer

This app is for **educational purposes only** and is **not a substitute for professional medical advice**. Please consult a doctor or healthcare professional for medical concerns.
