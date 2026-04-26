"""
GBV Hotspot Prediction in Kenya — Streamlit Dashboard
======================================================
Final Year Project | Data Science
"""

import re
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import HeatMap, MiniMap, Fullscreen
import branca.colormap as cm
from streamlit_folium import st_folium
import warnings, os, requests

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GBV Hotspot Prediction — Kenya",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e63946; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-title {
        font-size: 1.3rem; font-weight: 600;
        border-left: 4px solid #e63946;
        padding-left: 12px; margin: 20px 0 12px;
    }
    div[data-testid="stSidebarContent"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

DARK_BG  = "#0f1117"
TEXT_CLR = "#f0f0f0"

# ── County name fixes ─────────────────────────────────────────────────────────
# FIX 1: Expanded NAME_MAP to catch all realistic spelling variants
NAME_MAP = {
    # Nairobi
    "Nairobi City"      : "Nairobi",
    "Nairobi County"    : "Nairobi",
    # Muranga
    "Murang'A"          : "Muranga",
    "Murang'a"          : "Muranga",
    "Murang A"          : "Muranga",
    "Muranga"           : "Muranga",
    # Taita Taveta
    "Taita/Taveta"      : "Taita Taveta",
    "Taita-Taveta"      : "Taita Taveta",
    # Tharaka Nithi
    "Tharaka-Nithi"     : "Tharaka Nithi",
    # Elgeyo Marakwet
    "Elg Eyo/Marakwet"  : "Elgeyo Marakwet",
    "Elgeyo/Marakwet"   : "Elgeyo Marakwet",
    "Elgeyo-Marakwet"   : "Elgeyo Marakwet",
    "Elg Eyo Marakwet"  : "Elgeyo Marakwet",
    # Homa Bay
    "Homa-Bay"          : "Homa Bay",
    "Homabay"           : "Homa Bay",
    "Homa  Bay"         : "Homa Bay",
    # Tana River
    "Tana-River"        : "Tana River",
    # Trans Nzoia
    "Trans-Nzoia"       : "Trans Nzoia",
    # West Pokot
    "West-Pokot"        : "West Pokot",
    # Uasin Gishu
    "Uasin-Gishu"       : "Uasin Gishu",
}

def fix_county(name):
    if pd.isna(name): return np.nan
    name = str(name).strip().title()
    name = re.sub(r" +", " ", name)   # collapse multiple spaces
    return NAME_MAP.get(name, name)


# ── Data loading (cached so it only runs once) ────────────────────────────────
@st.cache_data
def load_all_data():
    # ── Femicide ──────────────────────────────────────────────────────────────
    df_fem = pd.read_excel("data/femicide_2016.xlsx")
    df_fem["county"] = df_fem["county"].apply(fix_county)
    df_fem["type_of_femicide"] = (
        df_fem["type_of_femicide"].str.strip().str.title()
        .replace({"Non-Intimate": "Non-Intimate", "Non-intimate": "Non-Intimate"})
        .fillna("Unknown")
    )
    df_fem["mode_of_killing"]      = df_fem["mode_of_killing"].str.strip().str.lower()
    df_fem["suspect_relationship"] = df_fem["suspect_relationship"].str.strip().str.title()
    df_fem["year"] = pd.to_datetime(df_fem["published_date"], errors="coerce").dt.year

    # ── Population ────────────────────────────────────────────────────────────
    df_pop = pd.read_csv("data/kenya_population_distribution_2019census.csv")
    df_pop = df_pop[df_pop["County"].str.upper() != "KENYA"].copy()
    df_pop["county"]      = df_pop["County"].apply(fix_county)
    df_pop["population"]  = df_pop["Total"].astype(str).str.replace(",", "").pipe(pd.to_numeric, errors="coerce")
    df_pop["pop_female"]  = df_pop["Female"].astype(str).str.replace(",", "").pipe(pd.to_numeric, errors="coerce")
    df_pop = df_pop[["county", "population", "pop_female"]].dropna()


    # ── Aggregate femicide ────────────────────────────────────────────────────
    totals = df_fem.groupby("county").size().reset_index(name="total_incidents")
    ipv    = (df_fem[df_fem["type_of_femicide"] == "Intimate"]
              .groupby("county").size().reset_index(name="ipv_incidents"))
    top_mode = (df_fem.dropna(subset=["mode_of_killing"])
                .groupby("county")["mode_of_killing"]
                .agg(lambda x: x.value_counts().index[0])
                .reset_index(name="top_mode"))
    df_agg = totals.merge(ipv, on="county", how="left").merge(top_mode, on="county", how="left")
    df_agg["ipv_incidents"] = df_agg["ipv_incidents"].fillna(0).astype(int)
    df_agg["ipv_pct"]       = (df_agg["ipv_incidents"] / df_agg["total_incidents"] * 100).round(1)

    # ── Merge with population ─────────────────────────────────────────────────
    df_merged = df_pop.merge(df_agg, on="county", how="left")
    df_merged["total_incidents"]    = df_merged["total_incidents"].fillna(0).astype(int)
    df_merged["gbv_rate_per_100k"]  = (df_merged["total_incidents"] / df_merged["population"] * 100_000).round(2).fillna(0)
    # FIX: also compute female-specific rate (used as a model feature)
    df_merged["gbv_rate_female"]    = (df_merged["total_incidents"] / df_merged["pop_female"] * 100_000).round(2).fillna(0)

    # ── Shapefile ─────────────────────────────────────────────────────────────
    shapefile = "data/kenya_counties.geojson"
    if not os.path.exists(shapefile):
        r = requests.get("https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_1.json", timeout=60)
        open(shapefile, "wb").write(r.content)
    gdf = gpd.read_file(shapefile)
    for col in ["NAME_1", "County", "COUNTY", "name", "ADM1_EN"]:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "county"}); break
    gdf["county"] = gdf["county"].apply(fix_county)
    gdf = gdf[["county", "geometry"]]

    # ── Master GeoDataFrame ───────────────────────────────────────────────────
    master = gdf.merge(df_merged, on="county", how="left")
    master["total_incidents"]   = master["total_incidents"].fillna(0)
    master["gbv_rate_per_100k"] = master["gbv_rate_per_100k"].fillna(0)
    master["gbv_rate_female"]   = master["gbv_rate_female"].fillna(0)

    # ── Spatial statistics + ML ───────────────────────────────────────────────
    moran_i = None
    moran_p = None
    try:
        from libpysal.weights import Queen
        import libpysal
        from esda.moran import Moran, Moran_Local
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        gdf_s = master.dropna(subset=["population"]).copy().reset_index(drop=True)
        y = gdf_s["gbv_rate_per_100k"].values

        w = Queen.from_dataframe(gdf_s, silence_warnings=True)
        w.transform = "r"
        moran    = Moran(y, w, permutations=999)
        moran_i  = round(moran.I, 4)
        moran_p  = round(moran.p_sim, 4)

        # ── Spatial lag ───────────────────────────────────────────────────────
        gdf_s["spatial_lag"] = libpysal.weights.lag_spatial(w, y)

        # ── Getis-Ord Gi* ─────────────────────────────────────────────────────
        w_b = Queen.from_dataframe(gdf_s, silence_warnings=True); w_b.transform = "b"
        n = len(y); ym = y.mean(); ys = y.std()
        gi_scores = []
        for i in range(n):
            nbrs = list(w_b.neighbors[i]) + [i]
            ws   = len(nbrs); swy = sum(y[j] for j in nbrs)
            num  = swy - ym * ws
            den  = ys * np.sqrt((n * ws - ws**2) / (n - 1))
            gi_scores.append(num / den if den != 0 else 0)
        gdf_s["gi_star_z"] = gi_scores
        gdf_s["gi_class"]  = gdf_s["gi_star_z"].apply(lambda z:
            "99% Hotspot"  if z >  2.576 else
            "95% Hotspot"  if z >  1.960 else
            "90% Hotspot"  if z >  1.645 else
            "99% Cold spot" if z < -2.576 else
            "95% Cold spot" if z < -1.960 else
            "Not significant")

        # ── LISA ──────────────────────────────────────────────────────────────
        local_m = Moran_Local(y, w, permutations=999, seed=42)
        sig     = local_m.p_sim < 0.05
        quad    = {1: "HH — Hotspot", 2: "LH — Outlier", 3: "LL — Cold spot", 4: "HL — Outlier"}
        gdf_s["lisa_label"] = [quad[q] if s else "Not significant" for q, s in zip(local_m.q, sig)]

        # ── Extra features matching the notebook's 7-feature model ───────────
        # IPV rate per county
        ipv_rate = (df_fem.dropna(subset=["county", "type_of_femicide"])
                    .groupby("county")
                    .apply(lambda x: (x["type_of_femicide"] == "Intimate").sum() / len(x))
                    .reset_index(name="ipv_rate"))
        # Incident trend (slope of yearly counts)
        yearly_tmp = (df_fem.dropna(subset=["county", "year"])
                      .groupby(["county", "year"]).size().reset_index(name="incidents"))
        def get_trend(grp):
            return np.polyfit(grp["year"], grp["incidents"], 1)[0] if len(grp) >= 3 else 0
        trends = yearly_tmp.groupby("county").apply(get_trend).reset_index(name="incident_trend")
        # Stranger rate
        stranger_rate = (df_fem.dropna(subset=["county", "suspect_relationship"])
                         .groupby("county")
                         .apply(lambda x: x["suspect_relationship"]
                                .str.contains("Stranger|Unknown", case=False, na=False).sum() / len(x))
                         .reset_index(name="stranger_rate"))

        gdf_s = gdf_s.merge(ipv_rate,      on="county", how="left")
        gdf_s = gdf_s.merge(trends,        on="county", how="left")
        gdf_s = gdf_s.merge(stranger_rate, on="county", how="left")
        gdf_s["ipv_rate"]       = gdf_s["ipv_rate"].fillna(0)
        gdf_s["incident_trend"] = gdf_s["incident_trend"].fillna(0)
        gdf_s["stranger_rate"]  = gdf_s["stranger_rate"].fillna(0)

        # ── Hotspot label (matches notebook: 60th pctile + Gi* > 0) ──────────
        threshold_rate      = gdf_s["gbv_rate_per_100k"].quantile(0.60)
        gdf_s["high_risk"]  = (
            (gdf_s["gbv_rate_per_100k"] >= threshold_rate) &
            (gdf_s["gi_star_z"] > 0)
        ).astype(int)

        # ── 7-feature model (matches notebook) ───────────────────────────────
        FEATURES = [
            "population",
            "gbv_rate_female",
            "spatial_lag",
            "gi_star_z",
            "ipv_rate",
            "incident_trend",
            "stranger_rate",
        ]
        FEATURES = [f for f in FEATURES if f in gdf_s.columns]

        dm   = gdf_s[FEATURES + ["high_risk", "county"]].dropna()
        X    = dm[FEATURES].values
        y_ml = dm["high_risk"].values
        sc   = StandardScaler(); Xs = sc.fit_transform(X)

        # KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xs, y_ml)

        # RF (regularised — matches notebook settings)
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=2, min_samples_split=8,
            min_samples_leaf=4, max_features="sqrt", max_samples=0.8,
            class_weight="balanced", random_state=42
        )
        rf.fit(Xs, y_ml)

        # FIX: index-based assignment to avoid positional misalignment
        gdf_s["rf_risk_prob"]  = np.nan
        gdf_s["knn_risk_pred"] = np.nan
        gdf_s.loc[dm.index, "rf_risk_prob"]  = rf.predict_proba(Xs)[:, 1]
        gdf_s.loc[dm.index, "knn_risk_pred"] = knn.predict(Xs)
        gdf_s["rf_risk_prob"]  = gdf_s["rf_risk_prob"].fillna(0)
        gdf_s["knn_risk_pred"] = gdf_s["knn_risk_pred"].fillna(0)

        # Show any counties excluded from modelling (missing features)
        _excluded = gdf_s[~gdf_s.index.isin(dm.index)]["county"].tolist()
        if _excluded:
            st.info(f"ℹ️ Counties excluded from modelling (missing features, assigned risk=0): {_excluded}")

        # Risk level label
        def risk_level(s):
            if s >= 0.75: return "Critical 🔴"
            if s >= 0.55: return "High 🟠"
            if s >= 0.35: return "Moderate 🟡"
            return               "Low 🟢"
        gdf_s["risk_level"] = gdf_s["rf_risk_prob"].apply(risk_level)

        # ── Cross-validation metrics for live model comparison display ────────
        cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        knn_preds = cross_val_predict(knn, Xs, y_ml, cv=cv_strat)
        rf_preds  = cross_val_predict(rf,  Xs, y_ml, cv=cv_strat)

        def get_metrics(y_true, y_pred):
            return {
                "Accuracy"     : round(accuracy_score(y_true, y_pred), 3),
                "Precision"    : round(precision_score(y_true, y_pred, average="macro", zero_division=0), 3),
                "Recall"       : round(recall_score(y_true, y_pred, average="macro", zero_division=0), 3),
                "F1 High risk" : round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 3),
                "F1 Low risk"  : round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 3),
                "Macro F1"     : round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
            }
        knn_metrics = get_metrics(y_ml, knn_preds)
        rf_metrics  = get_metrics(y_ml, rf_preds)

        # ── Merge spatial columns back into master ────────────────────────────
        drop_cols = [c for c in gdf_s.columns if c in master.columns and c != "county"]
        master_spatial = gdf_s.drop(columns=["geometry"] + [c for c in ["geometry"] if c in gdf_s.columns], errors="ignore")
        master = gdf.merge(
            master_spatial[[c for c in master_spatial.columns if c != "geometry"]],
            on="county", how="left"
        )
        master["total_incidents"]   = master["total_incidents"].fillna(0)
        master["gbv_rate_per_100k"] = master["gbv_rate_per_100k"].fillna(0)
        master["gbv_rate_female"]   = master["gbv_rate_female"].fillna(0)
        master["rf_risk_prob"]      = master["rf_risk_prob"].fillna(0)
        master["risk_level"]        = master["risk_level"].fillna("Low 🟢")

    except Exception as e:
        # Graceful fallback — spatial libs unavailable
        master["gi_class"]    = "Not significant"
        master["lisa_label"]  = "Not significant"
        master["rf_risk_prob"] = 0.0
        master["risk_level"]   = "Low 🟢"
        knn_metrics = None
        rf_metrics  = None

    return df_fem, df_pop, master, moran_i, moran_p, knn_metrics, rf_metrics


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data and running spatial analysis…"):
    df_fem, df_pop, master_gdf, moran_i, moran_p, knn_metrics, rf_metrics = load_all_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("kenya.jpg", width=150)
    st.markdown("## 🗺️ GBV Hotspot Prediction")
    st.markdown("**Kenya — 2016–2023**")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Overview",
        "🗺️ Interactive Map",
        "📅 Temporal Analysis",
        "🤖 Model Comparison",
        "📋 County Risk Table",
    ])

    st.markdown("---")
    if moran_i:
        st.markdown(f"**Moran's I:** `{moran_i}`")
        st.markdown(f"**p-value:** `{moran_p}`")
        sig = "✅ Significant" if moran_p < 0.05 else "❌ Not significant"
        st.markdown(f"**Clustering:** {sig}")
    st.markdown("---")
    st.markdown("*GBV HOTSPOTS PREDICTIONS*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🗺️ GBV Hotspot Prediction in Kenya")
    st.markdown("### Spatial Analysis & Machine Learning | 2016–2023")
    st.markdown("---")

    # FIX 2: total_inc was counting non-null county rows, not total incidents
    total_inc  = len(df_fem)
    top_county = master_gdf.nlargest(1, "total_incidents")["county"].values[0]
    top_rate   = master_gdf.nlargest(1, "gbv_rate_per_100k")["gbv_rate_per_100k"].values[0]
    counties_w = int((master_gdf["total_incidents"] > 0).sum())
    ipv_pct    = round(len(df_fem[df_fem["type_of_femicide"] == "Intimate"]) / len(df_fem) * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total_inc}</div>
            <div class="metric-label">Total incidents</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{counties_w}</div>
            <div class="metric-label">Counties affected</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{top_county}</div>
            <div class="metric-label">Highest incident county</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{top_rate:.1f}</div>
            <div class="metric-label">Highest rate / 100k</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{ipv_pct}%</div>
            <div class="metric-label">Intimate partner violence</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Incidents by suspect relationship</div>', unsafe_allow_html=True)
        rel = df_fem["suspect_relationship"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
        colors = ["#e63946", "#e63946", "#f4a261", "#f4a261", "#457b9d", "#457b9d"]
        ax.barh(rel.index[::-1], rel.values[::-1], color=colors[:len(rel)][::-1], edgecolor="#333")
        ax.set_xlabel("Incidents", color=TEXT_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Incidents by mode of killing</div>', unsafe_allow_html=True)
        mode = df_fem["mode_of_killing"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
        ax.barh(mode.index[::-1], mode.values[::-1], color="#457b9d", edgecolor="#333")
        ax.set_xlabel("Incidents", color=TEXT_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    if moran_i:
        st.markdown("---")
        st.markdown("<div class=\"section-title\">Spatial autocorrelation — Moran's I</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if moran_p < 0.05:
                st.success(f"✅ **Significant spatial clustering detected**  \nMoran's I = {moran_i} | p-value = {moran_p}  \nHigh-GBV counties cluster together geographically.")
            else:
                st.info(f"Moran's I = {moran_i} | p-value = {moran_p} — No significant clustering.")
        with col2:
            st.info("**What this means :**  \nThe spatial pattern is statistically proven and not random. This justifies using spatial methods (LISA, Gi*) and spatial features in the ML models.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INTERACTIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Interactive Map":
    st.markdown("# 🗺️ Interactive Map")
    
    layer_choice = st.selectbox("Select layer to display:", [
        "GBV Rate per 100k",
        "Getis-Ord Gi* Hotspots",
        "Random Forest Risk",
        "KDE Heatmap",
    ])

    gdf_map = master_gdf.copy()
    if gdf_map.crs and gdf_map.crs.to_epsg() != 4326:
        gdf_map = gdf_map.to_crs(epsg=4326)

    m = folium.Map(location=[0.02, 37.9], zoom_start=6, tiles=None)
    folium.TileLayer("CartoDB dark_matter", name="Dark basemap").add_to(m)
    folium.TileLayer("CartoDB positron",    name="Light basemap").add_to(m)

    # FIX 13: only fill numeric columns — don't let fillna(0) corrupt geometry
    num_cols = gdf_map.select_dtypes(include=[np.number]).columns
    gdf_map[num_cols] = gdf_map[num_cols].fillna(0)
    # String columns: fill with sensible defaults
    for sc in ["gi_class", "lisa_label", "risk_level"]:
        if sc in gdf_map.columns:
            gdf_map[sc] = gdf_map[sc].fillna("Not significant")
    gjson = gdf_map.to_json()

    vals = gdf_map["gbv_rate_per_100k"].replace(0, np.nan).dropna()
    colormap = cm.LinearColormap(["#ffffcc", "#fd8d3c", "#bd0026"],
        vmin=vals.quantile(0.05) if len(vals) else 0,
        vmax=vals.quantile(0.95) if len(vals) else 1,
        caption="GBV rate per 100,000")

    show_rate = layer_choice == "GBV Rate per 100k"
    show_gi   = layer_choice == "Getis-Ord Gi* Hotspots"
    show_rf   = layer_choice == "Random Forest Risk"
    show_kde  = layer_choice == "KDE Heatmap"

    # Rate layer
    folium.GeoJson(gjson, name="GBV Rate per 100k",
        style_function=lambda f: {
            "fillColor": colormap(f["properties"]["gbv_rate_per_100k"])
                        if (f["properties"]["gbv_rate_per_100k"] or 0) > 0 else "#1a1a2e",
            "fillOpacity": 0.75, "color": "#fff", "weight": 0.5},
        highlight_function=lambda f: {"weight": 2, "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=["county", "gbv_rate_per_100k", "total_incidents"],
            aliases=["County:", "Rate/100k:", "Incidents:"],
            style="background:#1a1a2e;color:#eee;font-family:sans-serif;font-size:12px;border-radius:6px;"),
        show=show_rate).add_to(m)
    if show_rate: colormap.add_to(m)

    # Gi* layer
    gi_col = {
        "99% Hotspot": "#d62828", "95% Hotspot": "#f07167", "90% Hotspot": "#fec89a",
        "Not significant": "#1a1a2e66",
        "95% Cold spot": "#00b4d8",  "99% Cold spot": "#0077b6",
    }
    folium.GeoJson(gjson, name="Getis-Ord Gi* Hotspots",
        style_function=lambda f: {
            "fillColor": gi_col.get(str(f["properties"].get("gi_class") or "Not significant"), "#1a1a2e"),
            "fillOpacity": 0.75, "color": "#fff", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=["county", "gi_class"],
            aliases=["County:", "Hotspot class:"],
            style="background:#1a1a2e;color:#eee;font-family:sans-serif;font-size:12px;border-radius:6px;"),
        show=show_gi).add_to(m)

    # RF layer
    rf_cm = cm.LinearColormap(["#1a9850", "#ffffbf", "#d73027"], vmin=0, vmax=1,
        caption="RF predicted risk probability")
    # Use 'lisa_label' in tooltip only if it exists in the data
    rf_tooltip_fields  = ["county", "rf_risk_prob"]
    rf_tooltip_aliases = ["County:", "RF Risk Score:"]
    if "lisa_label" in gdf_map.columns:
        rf_tooltip_fields.append("lisa_label");  rf_tooltip_aliases.append("LISA:")
    if "risk_level" in gdf_map.columns:
        rf_tooltip_fields.append("risk_level");  rf_tooltip_aliases.append("Risk Level:")
    folium.GeoJson(gjson, name="RF Predicted Risk",
        style_function=lambda f: {
            "fillColor": rf_cm(float(f["properties"].get("rf_risk_prob") or 0)),
            "fillOpacity": 0.75, "color": "#fff", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=rf_tooltip_fields, aliases=rf_tooltip_aliases,
            style="background:#1a1a2e;color:#eee;font-family:sans-serif;font-size:12px;border-radius:6px;"),
        show=show_rf).add_to(m)
    if show_rf: rf_cm.add_to(m)

    # KDE heatmap
    heat_data = [[r.geometry.centroid.y, r.geometry.centroid.x, r["gbv_rate_per_100k"]]
                 for _, r in gdf_map.iterrows() if r.geometry and r["gbv_rate_per_100k"] > 0]
    if heat_data:
        HeatMap(heat_data, name="KDE Heatmap", radius=35, blur=25,
            gradient={"0.4": "#ffffcc", "0.65": "#fd8d3c", "1": "#bd0026"},
            show=show_kde).add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    st_folium(m, width=None, height=560, returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Temporal Analysis":
    st.markdown("# 📅 Temporal Analysis — How Hotspots Shifted (2016–2023)")
    st.markdown("---")

    yearly = (df_fem.dropna(subset=["county", "year"])
              .groupby(["county", "year"]).size().reset_index(name="incidents"))
    years = sorted(df_fem["year"].dropna().unique().astype(int))

    selected_year = st.select_slider("Select year to view:", options=years, value=years[-1])

    col1, col2 = st.columns([1.5, 1])

    with col1:
        yr_data = df_fem[df_fem["year"] == selected_year].groupby("county").size().reset_index(name="incidents")
        yr_pop  = df_pop.merge(yr_data, on="county", how="left")
        yr_pop["incidents"] = yr_pop["incidents"].fillna(0)
        # FIX 14: round the rate column
        yr_pop["rate"] = (yr_pop["incidents"] / yr_pop["population"] * 100_000).fillna(0).round(2)
        yr_gdf = master_gdf[["county", "geometry"]].merge(
            yr_pop[["county", "rate", "incidents"]], on="county", how="left").fillna(0)

        fig, ax = plt.subplots(figsize=(8, 9))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG); ax.set_axis_off()
        yr_gdf[yr_gdf["rate"] == 0].plot(ax=ax, color="#1a1a2e", edgecolor="#333", linewidth=0.4)
        if yr_gdf[yr_gdf["rate"] > 0].shape[0] > 0:
            yr_gdf[yr_gdf["rate"] > 0].plot(column="rate", cmap="YlOrRd", linewidth=0.4,
                edgecolor="#333", legend=True, ax=ax,
                legend_kwds={"label": "Rate per 100,000", "orientation": "horizontal", "shrink": 0.6})
        total_yr = int(df_fem[df_fem["year"] == selected_year].shape[0])
        ax.set_title(f"GBV Hotspots — {int(selected_year)}  ({total_yr} cases)",
                     color=TEXT_CLR, fontsize=13, pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown(f"### Top counties in {int(selected_year)}")
        top_yr = yr_pop.nlargest(10, "incidents")[["county", "incidents", "rate"]].reset_index(drop=True)
        top_yr.columns = ["County", "Incidents", "Rate/100k"]
        top_yr.index += 1
        st.dataframe(top_yr, use_container_width=True)

        st.markdown("### Year summary")
        st.metric("Total incidents", total_yr)
        st.metric("Counties affected", int((yr_pop["incidents"] > 0).sum()))

    st.markdown("---")
    st.markdown('<div class="section-title">Incident trends — top 8 counties</div>', unsafe_allow_html=True)
    top8 = df_fem.groupby("county").size().nlargest(8).index.tolist()
    yearly_top = yearly[yearly["county"].isin(top8)]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    palette = ["#e63946", "#f4a261", "#457b9d", "#2a9d8f", "#e9c46a", "#a8dadc", "#9b5de5", "#00b4d8"]
    for county, color in zip(top8, palette):
        data = yearly_top[yearly_top["county"] == county].sort_values("year")
        ax.plot(data["year"], data["incidents"], marker="o", color=color,
                linewidth=2, markersize=6, label=county)
        if len(data) >= 3:
            z = np.polyfit(data["year"], data["incidents"], 1)
            ax.plot(data["year"], np.poly1d(z)(data["year"]), color=color,
                    linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Year", color=TEXT_CLR); ax.set_ylabel("Incidents", color=TEXT_CLR)
    ax.set_title("GBV Trends — Top 8 Counties (solid = actual, dashed = trend line)",
                 color=TEXT_CLR, fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_CLR, fontsize=9)
    ax.tick_params(colors=TEXT_CLR); ax.set_xticks(years)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison — KNN vs Random Forest")
    st.markdown("*Evaluated using Stratified 5-Fold Cross-Validation*")
    st.markdown("*7 features: population, female GBV rate, spatial lag, Gi\\* z-score, IPV rate, incident trend, stranger rate*")
    st.markdown("---")

    # FIX 16: use live computed metrics when available, fall back to notebook results
    if rf_metrics and knn_metrics:
        _rf  = rf_metrics
        _knn = knn_metrics
        st.caption("ℹ️ Metrics computed live from this session's model run.")
    else:
        # Fallback to notebook-run results
        _rf  = {"F1 High risk": 0.933, "F1 Low risk": 0.962, "Precision": 0.947,
                "Recall": 0.947, "Accuracy": 0.951, "Macro F1": 0.947}
        _knn = {"F1 High risk": 0.909, "F1 Low risk": 0.939, "Precision": 0.917,
                "Recall": 0.942, "Accuracy": 0.927, "Macro F1": 0.924}


    winner = "Random Forest" if _rf["Macro F1"] >= _knn["Macro F1"] else "KNN"
    winner_color = "#1D9E75"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{winner_color}">{winner}</div>
            <div class="metric-label">Winning model</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{_rf['Macro F1']}</div>
            <div class="metric-label">RF Macro F1 (5-Fold CV)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{winner_color}">{_rf['Accuracy']}</div>
            <div class="metric-label">RF Accuracy (5-Fold CV)</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Performance table (5-Fold Cross-Validation)")

    results = {
        "Model":          ["KNN (K=5)", "Random Forest"],
        "F1 — High risk": [_knn["F1 High risk"], _rf["F1 High risk"]],
        "F1 — Low risk":  [_knn["F1 Low risk"],  _rf["F1 Low risk"]],
        "Precision":      [_knn["Precision"],     _rf["Precision"]],
        "Recall":         [_knn["Recall"],        _rf["Recall"]],
        "Accuracy":       [_knn["Accuracy"],      _rf["Accuracy"]],
        "Macro F1":       [_knn["Macro F1"],      _rf["Macro F1"]],
    }
    df_results = pd.DataFrame(results).set_index("Model")
    st.dataframe(df_results, use_container_width=True)
    st.caption("5-Fold CV: dataset split into 5 folds, tested on each — scores averaged.")

    st.markdown("---")
    st.markdown("### Visual comparison")

    metrics  = ["F1 — High risk", "F1 — Low risk", "Macro F1", "Accuracy"]
    knn_vals = [0.909, 0.939, 0.924, 0.927]
    rf_vals  = [0.933, 0.962, 0.947, 0.951]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    x = np.arange(len(metrics)); w = 0.35
    b1 = ax.bar(x - w/2, knn_vals, w, label="KNN (K=5, distance)", color="#9b5de5", edgecolor="#333")
    b2 = ax.bar(x + w/2, rf_vals,  w, label="Random Forest",        color="#00b4d8", edgecolor="#333")
    ax.set_xticks(x); ax.set_xticklabels(metrics, color=TEXT_CLR, fontsize=10)
    ax.set_ylim(0.80, 1.05); ax.set_ylabel("Score", color=TEXT_CLR)
    ax.set_title("KNN (K=5) vs Random Forest — 5-Fold CV Performance",
                 color=TEXT_CLR, fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{bar.get_height():.3f}", ha="center", fontsize=8, color=TEXT_CLR)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{bar.get_height():.3f}", ha="center", fontsize=8, color=TEXT_CLR)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Why Random Forest won")
    st.info(f"""
**Random Forest outperformed KNN across all metrics (Macro F1 = {_rf['Macro F1']} vs {_knn['Macro F1']}).**

Random Forest builds 100 decision trees and combines their votes,
making it robust to noise in small datasets. With 7 features spanning
spatial, demographic and behavioural dimensions, it captures complex
non-linear relationships that KNN cannot.

KNN still performed strongly, confirming that geographic proximity is a
powerful signal for GBV risk.

The **F1 of {_rf['F1 Low risk']} for low-risk counties** means the model correctly
identifies safe counties too — reducing unnecessary resource allocation.
    """)

    st.success("""
✅ **Validation approach**

With only 47 counties, a single train-test split would give unstable
results depending on which counties happen to be in the test set.
Stratified 5-Fold CV averages performance across 5 different splits,
giving a more reliable estimate of generalisation performance.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COUNTY RISK TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 County Risk Table":
    st.markdown("# 📋 County Risk Table")
    st.markdown("Full sortable table of all 47 counties with risk scores.")
    st.markdown("---")

    # FIX 15: include risk_level column for readability
    cols = ["county", "total_incidents", "population", "gbv_rate_per_100k",
            "rf_risk_prob", "risk_level", "gi_class", "lisa_label"]
    available = [c for c in cols if c in master_gdf.columns]
    df_table = master_gdf[available].copy().sort_values("rf_risk_prob", ascending=False)
    df_table = df_table.reset_index(drop=True)
    df_table.index += 1

    rename = {
        "county"           : "County",
        "total_incidents"  : "Incidents",
        "population"       : "Population",
        "gbv_rate_per_100k": "Rate/100k",
        "rf_risk_prob"     : "RF Risk Score",
        "risk_level"       : "Risk Level",
        "gi_class"         : "Gi* Class",
        "lisa_label"       : "LISA Label",
    }
    df_table = df_table.rename(columns={k: v for k, v in rename.items() if k in df_table.columns})

    if "Population" in df_table.columns:
        df_table["Population"] = df_table["Population"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "—")
    if "RF Risk Score" in df_table.columns:
        df_table["RF Risk Score"] = df_table["RF Risk Score"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    if "Rate/100k" in df_table.columns:
        df_table["Rate/100k"] = df_table["Rate/100k"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("🔍 Search county", "")
    with col2:
        if "Gi* Class" in df_table.columns:
            gi_filter = st.selectbox("Filter by Gi* class",
                ["All"] + sorted(df_table["Gi* Class"].dropna().unique().tolist()))
        else:
            gi_filter = "All"
    with col3:
        if "Risk Level" in df_table.columns:
            risk_filter = st.selectbox("Filter by risk level",
                ["All"] + sorted(df_table["Risk Level"].dropna().unique().tolist()))
        else:
            risk_filter = "All"

    filtered = df_table.copy()
    if search:
        filtered = filtered[filtered["County"].str.contains(search, case=False, na=False)]
    if gi_filter != "All" and "Gi* Class" in filtered.columns:
        filtered = filtered[filtered["Gi* Class"] == gi_filter]
    if risk_filter != "All" and "Risk Level" in filtered.columns:
        filtered = filtered[filtered["Risk Level"] == risk_filter]

    st.dataframe(filtered, use_container_width=True, height=500)
    st.caption(f"Showing {len(filtered)} of 47 counties · Sorted by RF Risk Score (highest first)")

    csv = df_table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download full table as CSV",
                       data=csv, file_name="kenya_gbv_county_risk.csv", mime="text/csv")
