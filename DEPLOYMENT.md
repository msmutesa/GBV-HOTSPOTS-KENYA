# GBV Hotspot Prediction — Kenya
## Streamlit App Deployment Guide

---

## Your folder structure MUST look exactly like this on GitHub:

```
your-repo/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── data/
    ├── femicide_2016.xlsx
    ├── kenya_population_distribution_2019census.csv
    └── crime_trends_2010_2022.xlsx
```

---

## Step-by-step deployment (15 minutes)

### Step 1 — Create a GitHub account
Go to https://github.com and sign up (free)

### Step 2 — Create a new repository
- Click the green "New" button
- Name it: `gbv-hotspot-kenya`
- Set to **Public**
- Click "Create repository"

### Step 3 — Upload your files
- Click "uploading an existing file"
- Upload ALL files maintaining the folder structure above
- Commit with message: "Initial upload"

### Step 4 — Deploy on Streamlit Cloud
- Go to https://share.streamlit.io
- Sign in with your GitHub account
- Click "New app"
- Select your repository: `gbv-hotspot-kenya`
- Main file path: `app.py`
- Click "Deploy!"

### Step 5 — Wait ~3 minutes
Streamlit Cloud installs all libraries automatically.
Your app will be live at:
`https://your-username-gbv-hotspot-kenya-app-XXXX.streamlit.app`

---

## Test locally first (recommended)

```bash
pip install streamlit streamlit-folium
streamlit run app.py
```
Opens at http://localhost:8501

---

## What the app shows

| Page | Content |
|------|---------|
| 📊 Overview | Key stats, charts, Moran's I result |
| 🗺️ Interactive Map | Folium map with 4 toggleable layers |
| 📅 Temporal Analysis | Year-by-year hotspot maps + trend lines |
| 🤖 Model Comparison | KNN vs RF table + bar chart |
| 📋 County Risk Table | Sortable, searchable, downloadable |

---

## Share with your examiner
Once deployed, send them the URL.
They can view everything without installing Python.
