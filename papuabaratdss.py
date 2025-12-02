import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================== PAGE CONFIG =======================
st.set_page_config(page_title="🌤️ Prediksi Iklim Kepulauan Riau", layout="wide")

# ================== SOFT BLUE AESTHETIC CSS =======================
st.markdown("""
    <style>

    .main {
        background-color: #E8F4FF !important;
    }

    [data-testid="stSidebar"] {
        background-color: #D6ECFF !important;
    }

    .main-title {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #2A6F97 !important;
        text-align: center;
        padding-bottom: 6px;
    }

    .subtitle {
        font-size: 19px;
        color: #468FAF;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Card Statistik */
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
        border-left: 6px solid #61A5C2;
        text-align: center;
        margin-bottom: 15px;
    }

    h4 {
        color: #1B4965;
        margin-bottom: -5px;
    }

    </style>
""", unsafe_allow_html=True)


# ================== TITLE =======================
st.markdown("""
<h1 class="main-title">🌤️ Dashboard Analisis & Prediksi Iklim — Kepulauan Riau</h1>
<p class="subtitle">Visualisasi historis, eksplorasi variabel iklim, dan prediksi jangka panjang dengan tema soft blue aesthetic.</p>
""", unsafe_allow_html=True)


# ================== LOAD DATA =======================
@st.cache_data
def load_data():
    df = pd.read_excel("KEPRI.xlsx", sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin": "FF_X"})
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    return df

df = load_data()


# ================== SIDEBAR =======================
st.sidebar.header("🔍 Filter Data")

selected_year = st.sidebar.multiselect(
    "Pilih Tahun", sorted(df["Tahun"].unique()), default=df["Tahun"].unique()
)

selected_month = st.sidebar.multiselect(
    "Pilih Bulan", range(1, 13), default=range(1, 13)
)

df = df[df["Tahun"].isin(selected_year)]
df = df[df["Bulan"].isin(selected_month)]


# ================== VARIABLES =======================
possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Matahari (jam)",
    "FF_X": "Kecepatan Angin (m/s)",
    "DDD_X": "Arah Angin (°)"
}


# ================== AGGREGASI BULANAN =======================
agg_dict = {v: "mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly = df.groupby(["Tahun","Bulan"]).agg(agg_dict).reset_index()


# ================== TRAIN MODEL =======================
models = {}
metrics = {}

for v in available_vars:
    X = monthly[["Tahun","Bulan"]]
    y = monthly[v]
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=180, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xts)

    models[v] = model
    metrics[v] = (mean_squared_error(yts, pred)**0.5, r2_score(yts, pred))


# ================== CARD STATISTIK =======================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📏 Data Historis</h4>
        <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📅 Rentang Tahun</h4>
        <h2>{df['Tahun'].min()} - {df['Tahun'].max()}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📦 Variabel Iklim</h4>
        <h2>{len(available_vars)}</h2>
    </div>
    """, unsafe_allow_html=True)


# ================== GRAFIK HISTORIS =======================
st.subheader("📈 Tren Data Historis")

var_plot = st.selectbox("Pilih Variabel", [label[v] for v in available_vars])
key = [k for k,v in label.items() if v == var_plot][0]

monthly["Tanggal"] = pd.to_datetime(
    monthly["Tahun"].astype(str) + "-" + monthly["Bulan"].astype(str) + "-01"
)

fig1 = px.line(
    monthly,
    x="Tanggal",
    y=key,
    markers=True,
    title=var_plot,
    template="plotly_white",
    color_discrete_sequence=["#2A6F97"]
)
st.plotly_chart(fig1, use_container_width=True)


# ================== PREDIKSI 2025–2075 =======================
future = pd.DataFrame(
    [(y,m) for y in range(2025,2076) for m in range(1,13)],
    columns=["Tahun","Bulan"]
)

for v in available_vars:
    future[f"Pred_{v}"] = models[v].predict(future[["Tahun","Bulan"]])

st.subheader("🔮 Prediksi 2025–2075")

var_pred = st.selectbox("Pilih Variabel Prediksi", [label[v] for v in available_vars])
key2 = [k for k,v in label.items() if v == var_pred][0]

future["Tanggal"] = pd.to_datetime(
    future["Tahun"].astype(str) + "-" + future["Bulan"].astype(str) + "-01"
)

fig2 = px.line(
    future,
    x="Tanggal",
    y=f"Pred_{key2}",
    title=f"Prediksi {var_pred}",
    template="plotly_white",
    color_discrete_sequence=["#61A5C2"]
)

st.plotly_chart(fig2, use_container_width=True)


# ================== DOWNLOAD =======================
csv = future.to_csv(index=False).encode("utf8")
st.download_button(
    "📥 Download Dataset Prediksi",
    data=csv,
    file_name="prediksi_KEPRI.csv",
    mime="text/csv"
)

