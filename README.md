# 🏙️ EstateIQ — Real Estate Intelligence Platform

A professional Streamlit portfolio website showcasing real estate price prediction and analytics using machine learning.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Make sure `Real_estate.csv` is in the **same folder** as `real_estate_app.py`.

### 3. Run the app
```bash
streamlit run real_estate_app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
your_project/
├── real_estate_app.py       ← Main Streamlit application
├── requirements.txt         ← Python dependencies
├── Real_estate.csv          ← Dataset (place here)
└── README.md                ← This file
```

---

## 🎨 Features

| Section | Description |
|---|---|
| **Overview** | Hero section, KPIs, dataset guide, interactive map |
| **Data Explorer** | Filter, view & download data as Excel |
| **Insights & Charts** | 6 interactive chart tabs (Location, Age, MRT, Amenities, Correlations, Trends) |
| **Price Predictor** | Random Forest model with sliders, gauge chart, feature importance |
| **Contact** | Your contact info and tech stack |

### Themes
Switch between 3 dark themes from the sidebar:
- 🌑 Midnight Navy
- 🟣 Deep Purple  
- 🩵 Slate Teal

---

## 🛠️ Customisation

Before deploying, update these in `real_estate_app.py`:

```python
# In page_contact() — replace with your real info:
("📧", "Email",     "your.email@example.com",         "mailto:your.email@example.com"),
("💼", "LinkedIn",  "linkedin.com/in/yourprofile",    "https://linkedin.com/in/yourprofile"),
("🐙", "GitHub",    "github.com/yourusername",        "https://github.com/yourusername"),
("🌐", "Portfolio", "yourportfolio.com",               "https://yourportfolio.com"),
```

Also update `Your Name` in the contact card and the footer.

---

## 🚢 Deploying to Streamlit Cloud (Free)

1. Push your project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `real_estate_app.py`
5. Deploy — your portfolio is live! 🎉

---

## 📊 Dataset

- **Source:** Real Estate Valuation dataset (Taiwan)
- **Records:** 414 properties
- **Features:** Transaction date, house age, MRT distance, convenience stores, lat/lon
- **Target:** Price per unit area (10,000 TWD/ping)

---

## 🤖 Model

- **Algorithm:** Random Forest Regressor (200 trees)
- **Features used:** House Age, MRT Distance, Convenience Stores, Latitude, Longitude
- **Note:** Transaction Date excluded to ensure generalisation beyond the dataset's timeframe
- **Evaluation:** R², RMSE, MAE, 5-fold Cross-Validation

---

Built with ❤️ using Python, Streamlit, Plotly & Scikit-Learn.
