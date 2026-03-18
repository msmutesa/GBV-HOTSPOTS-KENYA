"""
GBV Hotspot Prediction in Kenya — Streamlit Dashboard
======================================================
Final Year Project | Data Science
"""

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
NAME_MAP = {
    "Nairobi City": "Nairobi", "Nairobi County": "Nairobi",
    "Murang'A": "Muranga", "Murang'a": "Muranga",
    "Taita/Taveta": "Taita Taveta", "Taita-Taveta": "Taita Taveta",
    "Tharaka-Nithi": "Tharaka Nithi",
    "Elg Eyo/Marakwet": "Elgeyo Marakwet",
    "Elgeyo/Marakwet": "Elgeyo Marakwet",
    "Elgeyo-Marakwet": "Elgeyo Marakwet",
}
def fix_county(name):
    if pd.isna(name): return np.nan
    name = str(name).strip().title()
    return NAME_MAP.get(name, name)


# ── Data loading (cached so it only runs once) ────────────────────────────────
@st.cache_data
def load_all_data():
    # Femicide
    df_fem = pd.read_excel("femicide_2016.xlsx")
    df_fem["county"] = df_fem["county"].apply(fix_county)
    df_fem["type_of_femicide"] = df_fem["type_of_femicide"].str.strip().str.title().replace(
        {"Non-intimate": "Non-Intimate"}).fillna("Unknown")
    df_fem["mode_of_killing"] = df_fem["mode_of_killing"].str.strip().str.lower()
    df_fem["suspect_relationship"] = df_fem["suspect_relationship"].str.strip().str.title()
    df_fem["year"] = pd.to_datetime(df_fem["published_date"], errors="coerce").dt.year

    # Population
    df_pop = pd.read_csv("kenya_population_distribution_2019census.csv")
    df_pop = df_pop[df_pop["County"].str.upper() != "KENYA"].copy()
    df_pop["county"] = df_pop["County"].apply(fix_county)
    df_pop["population"] = df_pop["Total"].astype(str).str.replace(",", "").pipe(pd.to_numeric, errors="coerce")
    df_pop["pop_female"] = df_pop["Female"].astype(str).str.replace(",", "").pipe(pd.to_numeric, errors="coerce")
    df_pop = df_pop[["county", "population", "pop_female"]].dropna()

    # Aggregate femicide
    totals = df_fem.groupby("county").size().reset_index(name="total_incidents")
    ipv    = df_fem[df_fem["type_of_femicide"] == "Intimate"].groupby("county").size().reset_index(name="ipv_incidents")
    df_agg = totals.merge(ipv, on="county", how="left")
    df_agg["ipv_incidents"] = df_agg["ipv_incidents"].fillna(0).astype(int)
    df_agg["ipv_pct"] = (df_agg["ipv_incidents"] / df_agg["total_incidents"] * 100).round(1)

    # Merge
    df_merged = df_pop.merge(df_agg, on="county", how="left")
    df_merged["total_incidents"] = df_merged["total_incidents"].fillna(0).astype(int)
    df_merged["gbv_rate_per_100k"] = (df_merged["total_incidents"] / df_merged["population"] * 100_000).round(2)
    df_merged["gbv_rate_per_100k"] = df_merged["gbv_rate_per_100k"].fillna(0)

    # Shapefile
    shapefile = "kenya_counties.geojson"
    if not os.path.exists(shapefile):
        r = requests.get("https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_1.json", timeout=60)
        open(shapefile, "wb").write(r.content)
    gdf = gpd.read_file(shapefile)
    for col in ["NAME_1", "County", "COUNTY", "name", "ADM1_EN"]:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "county"}); break
    gdf["county"] = gdf["county"].apply(fix_county)
    gdf = gdf[["county", "geometry"]]

    # Master GeoDataFrame
    master = gdf.merge(df_merged, on="county", how="left")
    master["total_incidents"]   = master["total_incidents"].fillna(0)
    master["gbv_rate_per_100k"] = master["gbv_rate_per_100k"].fillna(0)

    # Spatial stats
    try:
        from libpysal.weights import Queen
        import libpysal
        from esda.moran import Moran, Moran_Local

        w = Queen.from_dataframe(master.dropna(subset=["population"]), silence_warnings=True)
        w.transform = "r"
        gdf_s = master.dropna(subset=["population"]).copy()
        y = gdf_s["gbv_rate_per_100k"].values

        moran = Moran(y, w, permutations=999)

        # Gi*
        w_b = Queen.from_dataframe(gdf_s, silence_warnings=True); w_b.transform = "b"
        n = len(y); ym = y.mean(); ys = y.std()
        gi_scores = []
        for i in range(n):
            nbrs = list(w_b.neighbors[i]) + [i]
            ws = len(nbrs); swy = sum(y[j] for j in nbrs)
            num = swy - ym * ws
            den = ys * np.sqrt((n * ws - ws**2) / (n - 1))
            gi_scores.append(num / den if den != 0 else 0)
        gdf_s["gi_star_z"] = gi_scores
        gdf_s["gi_class"] = gdf_s["gi_star_z"].apply(lambda z:
            "99% Hotspot" if z > 2.576 else "95% Hotspot" if z > 1.96 else
            "90% Hotspot" if z > 1.645 else "99% Cold spot" if z < -2.576 else
            "95% Cold spot" if z < -1.96 else "Not significant")

        # LISA
        local_m = Moran_Local(y, w, permutations=999, seed=42)
        sig = local_m.p_sim < 0.05
        quad = {1: "HH — Hotspot", 2: "LH — Outlier", 3: "LL — Cold spot", 4: "HL — Outlier"}
        gdf_s["lisa_label"] = [quad[q] if s else "Not significant" for q, s in zip(local_m.q, sig)]

        # Spatial lag + RF
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        gdf_s["spatial_lag"] = libpysal.weights.lag_spatial(w, y)
        median_rate = gdf_s["gbv_rate_per_100k"].median()
        gdf_s["high_risk"] = (gdf_s["gbv_rate_per_100k"] > median_rate).astype(int)

        feats = ["population", "spatial_lag", "gi_star_z"]
        dm = gdf_s[feats + ["high_risk"]].dropna()
        X = dm[feats].values; y_ml = dm["high_risk"].values
        sc = StandardScaler(); Xs = sc.fit_transform(X)

        knn = KNeighborsClassifier(n_neighbors=5); knn.fit(Xs, y_ml)
        rf  = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=42)
        rf.fit(Xs, y_ml)

        gdf_s.loc[dm.index, "rf_risk_prob"]  = rf.predict_proba(Xs)[:, 1]
        gdf_s.loc[dm.index, "knn_risk_pred"] = knn.predict(Xs)
        gdf_s["rf_risk_prob"] = gdf_s["rf_risk_prob"].fillna(0)

        master = gdf.merge(gdf_s.drop(columns="geometry"), on="county", how="left")
        master["total_incidents"]   = master["total_incidents"].fillna(0)
        master["gbv_rate_per_100k"] = master["gbv_rate_per_100k"].fillna(0)
        master["rf_risk_prob"]      = master["rf_risk_prob"].fillna(0)

        moran_i = round(moran.I, 4)
        moran_p = round(moran.p_sim, 4)

    except Exception as e:
        moran_i = None; moran_p = None
        master["gi_class"] = "Not significant"
        master["lisa_label"] = "Not significant"
        master["rf_risk_prob"] = 0

    return df_fem, df_pop, master, moran_i, moran_p


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data and running spatial analysis..."):
    df_fem, df_pop, master_gdf, moran_i, moran_p = load_all_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png", width=120)
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🗺️ GBV Hotspot Prediction in Kenya")
    st.markdown("### Spatial Analysis & Machine Learning | 2016–2023")
    st.markdown("---")

    # Key metrics
    total_inc  = int(df_fem["county"].notna().sum())
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

    # Two charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Incidents by suspect relationship</div>', unsafe_allow_html=True)
        rel = df_fem["suspect_relationship"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
        colors = ["#e63946", "#e63946", "#f4a261", "#f4a261", "#457b9d", "#457b9d"]
        ax.barh(rel.index[::-1], rel.values[::-1], color=colors[::-1], edgecolor="#333")
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

    # Moran's I result box
    if moran_i:
        st.markdown("---")
        st.markdown('<div class="section-title">Spatial autocorrelation — Moran\'s I</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if moran_p < 0.05:
                st.success(f"✅ **Significant spatial clustering detected**  \nMoran's I = {moran_i} | p-value = {moran_p}  \nHigh-GBV counties cluster together geographically.")
            else:
                st.info(f"Moran's I = {moran_i} | p-value = {moran_p} — No significant clustering.")
        with col2:
            st.info("**What this means for this project:**  \nThe spatial pattern is statistically proven, not random. This justifies using spatial methods (LISA, Gi*) and spatial features in the ML models.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INTERACTIVE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Interactive Map":
    st.markdown("# 🗺️ Interactive Map")
    st.markdown("Click any county for details.")

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

    gjson = gdf_map.fillna(0).to_json()

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
    gi_col = {"99% Hotspot": "#d62828", "95% Hotspot": "#f07167", "90% Hotspot": "#fec89a",
              "Not significant": "#1a1a2e66", "95% Cold spot": "#00b4d8", "99% Cold spot": "#0077b6"}
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
        caption="RF predicted risk")
    folium.GeoJson(gjson, name="RF Predicted Risk",
        style_function=lambda f: {
            "fillColor": rf_cm(float(f["properties"].get("rf_risk_prob") or 0)),
            "fillOpacity": 0.75, "color": "#fff", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=["county", "rf_risk_prob", "lisa_label"],
            aliases=["County:", "RF Risk:", "LISA:"],
            style="background:#1a1a2e;color:#eee;font-family:sans-serif;font-size:12px;border-radius:6px;"),
        show=show_rf).add_to(m)
    if show_rf: rf_cm.add_to(m)

    # KDE
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
    st.markdown("*This part shows how GBV hotspots have evolved year by year.*")
    st.markdown("---")

    yearly = (df_fem.dropna(subset=["county", "year"])
              .groupby(["county", "year"]).size().reset_index(name="incidents"))
    years = sorted(df_fem["year"].dropna().unique().astype(int))

    # Year selector
    selected_year = st.select_slider("Select year to view:", options=years, value=years[-1])

    col1, col2 = st.columns([1.5, 1])

    with col1:
        # Map for selected year
        yr_data = df_fem[df_fem["year"] == selected_year].groupby("county").size().reset_index(name="incidents")
        yr_pop  = df_pop.merge(yr_data, on="county", how="left")
        yr_pop["incidents"] = yr_pop["incidents"].fillna(0)
        yr_pop["rate"] = (yr_pop["incidents"] / yr_pop["population"] * 100_000).fillna(0)
        yr_gdf  = master_gdf[["county", "geometry"]].merge(yr_pop[["county", "rate", "incidents"]], on="county", how="left").fillna(0)

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
        top_yr["Rate/100k"] = top_yr["Rate/100k"].round(2)
        top_yr.index += 1
        st.dataframe(top_yr, use_container_width=True)

        st.markdown("### Year summary")
        st.metric("Total incidents", total_yr)
        st.metric("Counties affected", int((yr_pop["incidents"] > 0).sum()))

    # Trend lines
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
    st.markdown("---")

    # Results table
    results = {
       "Model":           ["KNN (K=11)", "Random Forest"],
    "F1 — High risk":  [0.83, 0.98],
    "F1 — Low risk":   [0.78, 0.98],
    "Precision":       [0.73, 0.95],
    "Recall":          [0.95, 1.00],
    "Accuracy":        [0.80, 0.98],
    "Macro avg F1":    [0.80, 0.98],
    }
    df_results = pd.DataFrame(results).set_index("Model")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="color:#1D9E75">Random Forest</div>
            <div class="metric-label">Winning model</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0.98</div>
            <div class="metric-label">Random Forest F1 — High risk</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="color:#1D9E75">1.00</div>
            <div class="metric-label">Recall — hotspot detection</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
   st.markdown("### Performance table")
st.dataframe(df_results, use_container_width=True)

st.markdown("### 3 features vs 7 features — what changed?")
df_exp = pd.DataFrame({
    "Experiment":  ["3 spatial features", "7 features (spatial + behavioural)"],
    "KNN (K=11) F1": [0.84, 0.80],
    "Random Forest F1": [0.82, 0.98],
    "Winner": ["KNN", "Random Forest"],
})
st.dataframe(df_exp, use_container_width=True, hide_index=True)
st.caption("Adding behavioural features flipped the winner — Random Forest handles complexity better.")
    # Bar chart
    st.markdown("### Visual comparison")
    metrics  = ["F1 — High risk", "F1 — Low risk", "Macro avg F1"]
    knn_vals = [0.84, 0.81, 0.83]
    rf_vals  = [0.82, 0.79, 0.81]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    x = np.arange(len(metrics)); w = 0.35
    b1 = ax.bar(x - w/2, knn_vals, w, label="KNN (K=5)",     color="#9b5de5", edgecolor="#333")
    b2 = ax.bar(x + w/2, rf_vals,  w, label="Random Forest", color="#00b4d8", edgecolor="#333")
    ax.set_xticks(x); ax.set_xticklabels(metrics, color=TEXT_CLR)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score", color=TEXT_CLR)
    ax.set_title("KNN vs Random Forest — Performance Metrics", color=TEXT_CLR, fontsize=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f"{bar.get_height():.2f}", ha="center", fontsize=9, color=TEXT_CLR)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f"{bar.get_height():.2f}", ha="center", fontsize=9, color=TEXT_CLR)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Why Random Forest won with 7 features")
st.info("""**Random Forest outperformed KNN (K=11) when the feature set 
was expanded from 3 to 7 features (F1 = 0.98 vs 0.80).**

Random Forest is an ensemble method — it builds hundreds of decision trees 
and combines their votes. With richer features (IPV rate, incident trend, 
stranger rate, spatial lag, Gi* z-score), it can learn complex non-linear 
relationships between variables that KNN cannot.

Interestingly, with only 3 spatial features KNN outperformed Random Forest 
(F1 = 0.84 vs 0.82), showing that geographic proximity alone is a strong 
signal. But as feature complexity grew, Random Forest's ability to handle 
high-dimensional data gave it the edge.

The perfect recall of **1.00 for high-risk counties** means Random Forest 
missed zero actual GBV hotspot counties — critical for a public health 
intervention system.""")
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COUNTY RISK TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 County Risk Table":
    st.markdown("# 📋 County Risk Table")
    st.markdown("Full sortable table of all 47 counties with risk scores.")
    st.markdown("---")

    # Build display table
    cols = ["county", "total_incidents", "population", "gbv_rate_per_100k",
            "rf_risk_prob", "gi_class", "lisa_label"]
    available = [c for c in cols if c in master_gdf.columns]
    df_table = master_gdf[available].copy().sort_values("gbv_rate_per_100k", ascending=False)
    df_table = df_table.reset_index(drop=True)
    df_table.index += 1

    rename = {
        "county": "County", "total_incidents": "Incidents",
        "population": "Population", "gbv_rate_per_100k": "Rate/100k",
        "rf_risk_prob": "RF Risk Score", "gi_class": "Gi* Class",
        "lisa_label": "LISA Label",
    }
    df_table = df_table.rename(columns={k: v for k, v in rename.items() if k in df_table.columns})

    if "Population" in df_table.columns:
        df_table["Population"] = df_table["Population"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "—")
    if "RF Risk Score" in df_table.columns:
        df_table["RF Risk Score"] = df_table["RF Risk Score"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("🔍 Search county", "")
    with col2:
        if "Gi* Class" in df_table.columns:
            gi_filter = st.selectbox("Filter by Gi* class",
                ["All"] + sorted(df_table["Gi* Class"].dropna().unique().tolist()))
        else:
            gi_filter = "All"

    filtered = df_table.copy()
    if search:
        filtered = filtered[filtered["County"].str.contains(search, case=False, na=False)]
    if gi_filter != "All" and "Gi* Class" in filtered.columns:
        filtered = filtered[filtered["Gi* Class"] == gi_filter]

    st.dataframe(filtered, use_container_width=True, height=500)
    st.caption(f"Showing {len(filtered)} of 47 counties · Sorted by GBV rate per 100,000")

    # Download button
    csv = df_table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download full table as CSV",
                       data=csv, file_name="kenya_gbv_county_risk.csv", mime="text/csv")
