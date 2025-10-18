import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Global Tsunami Risk AI",
    page_icon="🌊",
    layout="wide",
)

# ---------- LOAD DATA & MODELS ----------
@st.cache_resource
def load_all():
    data = pd.read_csv("data/earthquake_data_tsunami.csv")
    model_old = joblib.load("tsunami_rf_model.pkl")           # model lama (pakai Year)
    feat_old = joblib.load("tsunami_features.pkl")
    model_new = joblib.load("tsunami_rf_noyear.pkl")          # model baru (tanpa Year)
    feat_new = joblib.load("tsunami_features_noyear.pkl")
    return data, model_old, feat_old, model_new, feat_new

data, model_old, feat_old, model_new, feat_new = load_all()

# ---------- SIDEBAR ----------
st.sidebar.title("🌊 Tsunami Risk AI Dashboard")
mode = st.sidebar.radio(
    "Pilih Mode:",
    ["Analisis Historis", "Prediksi Tsunami"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    'Developed by Ahmad Mihdan Advani • '
    '<a href="https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/" target="_blank">Global Earthquake–Tsunami Dataset (2001–2022)</a>',
    unsafe_allow_html=True
)

# ======================================================================
# 📊 MODE 1: ANALISIS HISTORIS
# ======================================================================
if mode == "Analisis Historis":
    st.title("Global Earthquake–Tsunami Analysis (2001–2022)")
    st.markdown(
        "Menampilkan persebaran global dan pola tsunami menggunakan model historis."
    )

    # --- Peta Persebaran ---
    fig = px.scatter_geo(
        data,
        lat="latitude",
        lon="longitude",
        color="tsunami",
        color_continuous_scale=["#1f77b4", "#d62728"],
        hover_data=["magnitude", "depth", "Year", "sig"],
        projection="natural earth",
        title="Persebaran Gempa Global (Merah = Tsunami, Biru = Non-Tsunami)",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # --- Scatter Magnitude vs Depth ---
    st.subheader("Distribusi Magnitude vs Depth")
    fig2 = px.scatter(
        data,
        x="depth",
        y="magnitude",
        color="tsunami",
        color_discrete_map={0: "blue", 1: "red"},
        hover_data=["Year", "latitude", "longitude"],
        labels={"depth": "Depth (km)", "magnitude": "Magnitude (Richter)"},
        title="Hubungan Kedalaman dan Magnitudo terhadap Potensi Tsunami",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info("""
    Sebagian besar gempa yang berpotensi tsunami terjadi pada kedalaman kurang dari 100 km 
    dengan magnitudo di atas 7.5. Semakin dalam pusat gempa, kemungkinan menimbulkan tsunami
    semakin kecil.
    """)

    trend = (
        data.groupby("Year")["tsunami"]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "tsunami_events"})
    )
    trend["tsunami_rate"] = (trend["tsunami_events"] / trend["total"]) * 100

    fig_trend = px.line(
        trend,
        x=trend.index,
        y="tsunami_events",
        title="Jumlah Kejadian Tsunami per Tahun",
        markers=True,
    )
    fig_trend.update_traces(line_color="#d62728")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.info("""
    Jumlah kejadian tsunami mulai meningkat tajam setelah tahun 2011, yang bertepatan dengan 
    terjadinya gempa besar di beberapa wilayah seperti Jepang dan Pasifik. 
    Setelah itu, tren cenderung fluktuatif namun tetap berada pada level tinggi, 
    menunjukkan peningkatan aktivitas gempa yang berpotensi tsunami dalam dekade terakhir.
    """)


    fig_rate = px.line(
        trend,
        x=trend.index,
        y="tsunami_rate",
        title="Persentase Tsunami dari Total Gempa (%)",
        markers=True,
    )
    fig_rate.update_traces(line_color="#1f77b4")
    st.plotly_chart(fig_rate, use_container_width=True)

    st.info("""
    Persentase kejadian tsunami terhadap total gempa meningkat signifikan setelah 2011, 
    bahkan mencapai lebih dari 60% di beberapa tahun. Hal ini mengindikasikan bahwa 
    proporsi gempa yang memicu tsunami menjadi semakin besar pada periode modern, 
    mungkin karena pengaruh lokasi gempa yang lebih dekat dengan lautan dangkal.
    """)


    st.subheader("Distribusi Magnitude — Tsunami vs Non-Tsunami")
    st.caption("Menunjukkan distribusi kekuatan gempa untuk masing-masing kategori.")
    fig_hist = px.histogram(
        data,
        x="magnitude",
        color="tsunami",
        nbins=25,
        barmode="overlay",
        marginal="box",
        opacity=0.7,
        color_discrete_map={0: "#1f77b4", 1: "#d62728"},
    )
    fig_hist.update_layout(
        title="Distribusi Magnitude Berdasarkan Kategori Tsunami",
        xaxis_title="Magnitude (Richter)",
        yaxis_title="Jumlah Kejadian",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.info("""
    Distribusi menunjukkan bahwa gempa yang menyebabkan tsunami umumnya memiliki 
    magnitudo lebih tinggi (rata-rata di atas 7.0 Richter). 
    Sebaliknya, gempa non-tsunami cenderung terjadi pada rentang magnitudo yang lebih kecil. 
    Artinya, semakin besar magnitudo, semakin besar pula potensi tsunami yang dihasilkan.
    """)


    st.subheader("Korelasi antar Fitur")
    st.caption("Analisis hubungan antar fitur numerik untuk melihat faktor paling relevan terhadap tsunami.")

    num_cols = ["magnitude", "depth", "sig", "mmi", "cdi", "nst", "dmin", "gap", "latitude", "longitude", "tsunami"]
    corr = data[num_cols].corr()

    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (Spearman)",
        aspect="auto",
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info("""
    Hasil korelasi menunjukkan bahwa magnitudo dan kedalaman (depth) memiliki hubungan 
    paling signifikan dengan potensi tsunami. Gempa dengan magnitudo besar dan kedalaman 
    rendah (shallow) umumnya berpotensi tinggi memicu tsunami. 
    Fitur lokasi (latitude, longitude) juga turut berperan, menandakan pengaruh 
    zona geografis terhadap risiko tsunami.
    """)


    

    # --- Statistik Ringkas ---
    st.subheader("Statistik Ringkas Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{len(data)} kejadian")
    with col2:
        st.metric("Rasio Tsunami", f"{data['tsunami'].mean()*100:.1f}%")
    with col3:
        st.metric("Rentang Tahun", f"{data['Year'].min()} – {data['Year'].max()}")


    st.info("""
    Dataset berisi total 782 data kejadian gempa bumi signifikan yang tercatat secara global 
    antara tahun 2001 hingga 2022. Dari seluruh peristiwa tersebut, sekitar 38.9% di antaranya 
    terkategorikan sebagai gempa yang berpotensi menimbulkan tsunami. 
    Rentang waktu yang luas ini memberikan representasi yang baik untuk analisis pola 
    dan prediksi risiko tsunami di berbagai wilayah dunia.
    """)


# ======================================================================
# 🔮 MODE 2: PREDIKSI MASA DEPAN
# ======================================================================
elif mode == "Prediksi Tsunami":
    st.title("Tsunami Risk Predictor")
    st.caption("Model tanpa fitur waktu, dirancang untuk prediksi gempa baru di masa depan.")

    # --- Input Parameter ---
    with st.expander("🔧 Input Parameter Gempa"):
        col1, col2 = st.columns(2)
        with col1:
            magnitude = st.number_input("Magnitude (6.5–10)", 6.5, 10.0, 7.2, step=0.1)
            depth = st.number_input("Depth (km)", 0.0, 700.0, 30.0, step=1.0)
            sig = st.number_input("Significance (sig)", 0, 4000, 800, step=10)
            mmi = st.number_input("MMI (1–9)", 1, 9, 5, step=1)
            cdi = st.number_input("CDI (0–9)", 0, 9, 3, step=1)
        with col2:
            nst = st.number_input("Stations (nst)", 0, 1000, 50)
            dmin = st.number_input("dmin (deg)", 0.0, 20.0, 2.0)
            gap = st.number_input("gap (deg)", 0.0, 360.0, 120.0)
            latitude = st.number_input("Latitude", -90.0, 90.0, 0.0)
            longitude = st.number_input("Longitude", -180.0, 180.0, 120.0)

    # --- Fitur turunan ---
    mag_depth_ratio = magnitude / (depth if depth != 0 else 1)
    cat = "shallow" if depth < 70 else "intermediate" if depth < 300 else "deep"
    st.caption(f"Depth Category: **{cat}** | Magnitude/Depth Ratio: **{mag_depth_ratio:.3f}**")

    # --- Build DataFrame Input ---
    row = {col: 0.0 for col in feat_new}
    base = {
        "magnitude": magnitude,
        "depth": depth,
        "sig": sig,
        "mmi": mmi,
        "cdi": cdi,
        "nst": nst,
        "dmin": dmin,
        "gap": gap,
        "latitude": latitude,
        "longitude": longitude,
        "mag_depth_ratio": mag_depth_ratio,
    }
    row.update(base)
    row["depth_category_intermediate"] = 1.0 if cat == "intermediate" else 0.0
    row["depth_category_shallow"] = 1.0 if cat == "shallow" else 0.0
    X_input = pd.DataFrame([[row[c] for c in feat_new]], columns=feat_new)

    # --- Predict Button ---
    if st.button("🚀 Prediksi Tsunami"):
        proba = float(model_new.predict_proba(X_input)[0, 1])
        pred = int(proba >= 0.5)

        st.metric("Tsunami Probability", f"{proba*100:.2f}%")
        st.success("🟥 Likely Tsunami" if pred == 1 else "🟦 Not Likely")

        # --- Peta Lokasi ---
        df_map = pd.DataFrame({"lat": [latitude], "lon": [longitude], "mag": [magnitude]})
        fig3 = px.scatter_geo(
            df_map,
            lat="lat",
            lon="lon",
            size="mag",
            projection="natural earth",
            title="Epicenter Location",
        )
        fig3.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)



# ======================================================================
# END
# ======================================================================
st.markdown("---")
st.caption(
    '🌍 Developed with Streamlit — Dataset: '
    '<a href="https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/" target="_blank">Global Earthquake–Tsunami Dataset (2001–2022)</a>',
    unsafe_allow_html=True
)