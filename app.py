"""
DASHBOARD REGRESI PANEL FISKAL - 97 KOTA DI INDONESIA (2014-2024)
Tema: biru muda lautan.  Jalankan: streamlit run app.py
Tab: Ringkasan | Tren | Korelasi | Regresi | Machine Learning | Regresi 3D
"""
# ---------------------------------------------------------------
# 1. IMPORT LIBRARY & KONFIGURASI
# ---------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Dashboard Regresi Panel Fiskal", page_icon="🌊", layout="wide")

# Palet warna laut
LAUT, TOSCA, LANGIT, DALAM = "#0ea5e9", "#14b8a6", "#bae6fd", "#0c4a6e"

# Gaya tampilan (CSS): latar gradasi laut, sidebar, banner, kartu metrik, tab
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: linear-gradient(160deg, #e0f7ff 0%, #f1fbff 45%, #d6f1f5 100%); }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #b6e4f8 0%, #e3f6fd 60%, #cdeee9 100%); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp p, .stApp li {{ color: {DALAM}; }}
.hero {{ background: linear-gradient(120deg, #0369a1, {LAUT} 55%, #22d3ee); padding: 1.4rem 1.8rem;
        border-radius: 18px; box-shadow: 0 8px 24px rgba(14,165,233,.30); margin-bottom: 1rem; }}
.hero h1, .hero p {{ color: #fff !important; margin: 0; }}
.hero h1 {{ font-size: 1.9rem; font-weight: 700; }}
.hero p {{ opacity: .92; margin-top: .3rem; }}
[data-testid="stMetric"] {{ background: #fff; border-left: 6px solid {LAUT}; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 4px 14px rgba(14,165,233,.15); }}
[data-testid="stMetricValue"] {{ color: #0369a1; font-weight: 700; }}
button[data-baseweb="tab"] {{ font-weight: 500; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background: {LAUT}; color: #fff !important; border-radius: 10px 10px 0 0; }}
.catatan {{ background: #e0f2fe; border-left: 5px solid {LAUT}; border-radius: 10px; padding: .8rem 1rem; color: {DALAM}; }}
</style>
""", unsafe_allow_html=True)

# Variabel-variabel dataset
KANDIDAT_Y = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]
FISKAL = ["DBH_PAJAK", "DBH_SDA", "DBH_Lainnya", "DAU_Grant", "DAU_Earmark", "DAK_Fisik",
          "DAK_Nonfisik", "PAD", "ICT", "bm_pendidikan", "bm_kesehatan", "bm_perumahan",
          "bm_lingkungan", "bm_ekonomi", "bm_infrastruktur", "bm_pariwisata",
          "bm_perlindungan_sosial"]


def tema(fig, tinggi=430):
    """Seragamkan tampilan grafik Plotly dengan tema laut."""
    fig.update_layout(template="plotly_white", height=tinggi, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,.65)", margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Poppins, sans-serif", color=DALAM))
    return fig


# ---------------------------------------------------------------
# 2. PEMUATAN DATASET (otomatis cari CSV berawalan "dataset_fiskal")
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Memuat dataset...")
def muat_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Kota"] = df["Kota"].astype(str).str.strip()
    df["Tahun"] = df["Tahun"].astype(int)
    return df.sort_values(["Kota", "Tahun"]).reset_index(drop=True)


csv_ada = sorted(Path(__file__).parent.glob("dataset_fiskal*.csv"))
if not csv_ada:
    st.error("File dataset (dataset_fiskal*.csv) tidak ditemukan di folder yang sama dengan app.py.")
    st.stop()
data = muat_data(str(csv_ada[0]))
num = [c for c in data.select_dtypes("number").columns if c != "Tahun"]
fiskal = [c for c in FISKAL if c in num]
opsi_y = [c for c in KANDIDAT_Y if c in num]


# ---------------------------------------------------------------
# 3. FUNGSI ANALISIS
# ---------------------------------------------------------------
def estimasi(df, y, xs, jenis, efek_waktu, jenis_se):
    """Regresi panel (linearmodels). Indeks: Kota, Tahun."""
    d = df[["Kota", "Tahun", y] + xs].dropna().set_index(["Kota", "Tahun"])
    Y, X = d[y], sm.add_constant(d[xs], has_constant="add")
    if jenis_se == "Clustered (per kota)":
        opsi = dict(cov_type="clustered", cluster_entity=True)
    elif jenis_se == "Robust":
        opsi = dict(cov_type="robust")
    else:
        opsi = dict(cov_type="unadjusted")
    if jenis == "Pooled OLS":
        m = PooledOLS(Y, X)
    elif jenis == "Fixed Effects":
        m = PanelOLS(Y, X, entity_effects=True, time_effects=efek_waktu)
    else:
        m = RandomEffects(Y, X)
    return m.fit(**opsi), X


def tabel_koef(h):
    ci = h.conf_int()
    t = pd.DataFrame({"Koefisien": h.params, "Std. Error": h.std_errors, "t-stat": h.tstats,
                      "p-value": h.pvalues, "CI bawah": ci.iloc[:, 0], "CI atas": ci.iloc[:, 1]})
    t["Sig."] = t["p-value"].apply(lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "")
    return t


def uji_hausman(df, y, xs):
    fe, _ = estimasi(df, y, xs, "Fixed Effects", False, "Biasa")
    re, _ = estimasi(df, y, xs, "Random Effects", False, "Biasa")
    k = [c for c in fe.params.index if c != "const" and c in re.params.index]
    b = (fe.params[k] - re.params[k]).values
    v = (fe.cov.loc[k, k] - re.cov.loc[k, k]).values
    stat = float(b @ np.linalg.pinv(v) @ b)
    return stat, len(k), float(stats.chi2.sf(stat, len(k)))


def narasi_tren(s, nama, wilayah):
    """Kalimat interpretasi otomatis untuk grafik tren."""
    selisih = s.iloc[-1] - s.iloc[0]
    arah = "naik" if selisih > 0 else "turun"
    return (f"Secara keseluruhan, <b>{nama}</b> di {wilayah} <b>{arah}</b> dari {s.iloc[0]:.2f} "
            f"({s.index[0]}) menjadi {s.iloc[-1]:.2f} ({s.index[-1]}), selisih {selisih:+.2f}. "
            f"Nilai tertinggi terjadi pada {s.idxmax()} dan terendah pada {s.idxmin()}. "
            "Perhatikan juga pola grafik: perubahan antartahun bisa naik-turun dan tidak selalu "
            "searah dengan perubahan total.")


def kekuatan(r):
    r = abs(r)
    return "sangat kuat" if r >= .8 else "kuat" if r >= .6 else "sedang" if r >= .4 else "lemah"


# ---------------------------------------------------------------
# 4. SIDEBAR: FILTER
# ---------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Filter")
cakupan = st.sidebar.radio("Cakupan data", ["Satu Kota", "Semua Kota"])
t_min, t_max = int(data["Tahun"].min()), int(data["Tahun"].max())
rentang = st.sidebar.slider("Rentang tahun", t_min, t_max, (t_min, t_max))
df = data[data["Tahun"].between(*rentang)]
if cakupan == "Satu Kota":
    kota = st.sidebar.selectbox("Pilih kota", sorted(data["Kota"].unique()))
    df = df[df["Kota"] == kota]
    wilayah = kota
else:
    wilayah = f"{data['Kota'].nunique()} kota"
satu = cakupan == "Satu Kota"

# ---------------------------------------------------------------
# 5. HEADER
# ---------------------------------------------------------------
st.markdown(
    f'<div class="hero"><h1>🌊 Dashboard Regresi Panel Fiskal</h1>'
    f'<p>{data["Kota"].nunique()} kota di Indonesia ({t_min}-{t_max}) · transfer fiskal, PAD, ICT, '
    f'dan belanja modal terhadap indikator pembangunan</p></div>', unsafe_allow_html=True)
st.markdown(f"## 📍 {wilayah} ({rentang[0]}-{rentang[1]})")
if df.empty:
    st.warning("Tidak ada data untuk filter ini.")
    st.stop()

tabs = st.tabs(["🏠 Ringkasan", "📈 Tren", "🔲 Korelasi", "Σ Regresi", "🤖 Machine Learning", "🧊 Regresi 3D"])

# ---------------------------------------------------------------
# 6. TAB RINGKASAN
# ---------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah observasi", f"{len(df):,}")
    c2.metric("Jumlah kota", df["Kota"].nunique())
    c3.metric("Periode", f"{df['Tahun'].min()}-{df['Tahun'].max()}")
    c4.metric("Jumlah variabel", len(num))
    st.markdown("#### Tabel Data")
    st.dataframe(df, use_container_width=True, height=320)
    st.markdown("#### Statistik Deskriptif")
    st.dataframe(df[num].describe().T.round(3), use_container_width=True)
    st.download_button("⬇️ Unduh data terfilter (CSV)", df.to_csv(index=False).encode(), "data_terfilter.csv", "text/csv")

# ---------------------------------------------------------------
# 7. TAB TREN
# ---------------------------------------------------------------
with tabs[1]:
    v = st.selectbox("Pilih variabel", num, index=num.index(opsi_y[0]) if opsi_y else 0, key="tren")
    g = df.groupby("Tahun")[v]
    rata = g.mean()
    fig = go.Figure()
    if not satu:  # pita min-maks antar kota
        fig.add_trace(go.Scatter(x=rata.index, y=g.max(), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rata.index, y=g.min(), fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(14,165,233,.18)", name="Rentang min-maks"))
    fig.add_trace(go.Scatter(x=rata.index, y=rata.values, mode="lines+markers", name="Rata-rata" if not satu else v,
                             line=dict(color=LAUT, width=3), marker=dict(size=8, color=TOSCA)))
    fig.update_layout(title=f"Tren {v} - {wilayah}", xaxis_title="Tahun", yaxis_title=v)
    st.plotly_chart(tema(fig), use_container_width=True)
    st.markdown("### 📝 Interpretasi")
    st.markdown(f'<div class="catatan">{narasi_tren(rata, v, wilayah)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# 8. TAB KORELASI
# ---------------------------------------------------------------
with tabs[2]:
    pilih = st.multiselect("Variabel untuk korelasi", num, default=num)
    if len(pilih) < 2:
        st.info("Pilih minimal dua variabel.")
    else:
        c = df[pilih].corr()
        fig = px.imshow(c, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(tema(fig, 640), use_container_width=True)
        iu = np.triu_indices_from(c, k=1)
        pasang = pd.Series(c.values[iu], index=[(c.index[i], c.columns[j]) for i, j in zip(*iu)])
        if not pasang.empty:
            a, b = pasang.abs().idxmax()
            r = pasang[(a, b)]
            n_kuat = int((pasang.abs() >= .5).sum())
            st.markdown("### 📝 Interpretasi")
            st.markdown(
                f'<div class="catatan">Korelasi terkuat ada pada <b>{a}</b> dan <b>{b}</b> dengan r = <b>{r:.2f}</b> '
                f'(hubungan {"positif" if r > 0 else "negatif"}, kekuatan {kekuatan(r)}). Dari {len(pasang)} pasangan '
                f'variabel unik, {n_kuat} pasangan memiliki |r| ≥ 0,5. Korelasi tidak berarti sebab-akibat.'
                f'{" Catatan: data satu kota hanya berisi beberapa tahun, sehingga korelasinya mudah berubah." if satu else ""}'
                '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# 9. TAB REGRESI (panel)
# ---------------------------------------------------------------
with tabs[3]:
    cA, cB = st.columns([1, 2])
    y = cA.selectbox("Variabel dependen (Y)", opsi_y, key="y_reg")
    xs = cB.multiselect("Variabel independen (X)", [c for c in fiskal if c != y], default=[c for c in fiskal if c != y])
    cC, cD, cE = st.columns(3)
    if satu:
        jenis, efek_waktu, se = "Pooled OLS", False, "Biasa"
        cC.info("Satu kota: hanya Pooled OLS yang bisa dipakai (data panel butuh banyak kota).")
    else:
        jenis = cC.radio("Jenis model", ["Pooled OLS", "Fixed Effects", "Random Effects"], index=1, horizontal=True)
        efek_waktu = cD.checkbox("Efek tetap tahun (two-way)", disabled=jenis != "Fixed Effects")
        se = cE.selectbox("Standard error", ["Clustered (per kota)", "Robust", "Biasa"])

    if not xs or len(df) <= len(xs) + 2:
        st.warning("Pilih variabel X yang lebih sedikit atau perluas filter. Jumlah observasi harus lebih besar dari jumlah variabel.")
    else:
        try:
            h, Xm = estimasi(df, y, xs, jenis, bool(efek_waktu), se)
        except Exception as e:
            st.error(f"Estimasi gagal: {e}")
            st.stop()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² (within)", f"{getattr(h, 'rsquared_within', np.nan):.4f}")
        m2.metric("R² (overall)", f"{getattr(h, 'rsquared_overall', np.nan):.4f}")
        m3.metric("F-statistik", f"{h.f_statistic.stat:.2f}")
        m4.metric("Prob (F)", f"{h.f_statistic.pval:.4f}")

        tk = tabel_koef(h)
        st.markdown("#### Tabel Koefisien")
        st.dataframe(tk.style.format({k: "{:.4f}" for k in tk.columns if k != "Sig."}), use_container_width=True)
        st.caption("Signifikansi: *** p<0,01 | ** p<0,05 | * p<0,10")

        kt = tk.drop(index="const", errors="ignore").sort_values("Koefisien")
        fig = go.Figure(go.Scatter(
            x=kt["Koefisien"], y=kt.index, mode="markers",
            marker=dict(size=9, color=np.where(kt["p-value"] < .05, "#e11d48", LAUT)),
            error_x=dict(type="data", symmetric=False, array=kt["CI atas"] - kt["Koefisien"],
                         arrayminus=kt["Koefisien"] - kt["CI bawah"], color=TOSCA)))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Koefisien dan CI 95% (merah = signifikan 5%)")
        st.plotly_chart(tema(fig, max(320, 28 * len(kt) + 120)), use_container_width=True)

        st.markdown("### 📝 Interpretasi")
        sig = kt[kt["p-value"] < .05]
        if sig.empty:
            teks = (f"Tidak ada variabel yang signifikan pada tingkat 5% (Prob F = {h.f_statistic.pval:.4f}). "
                    f"Pada spesifikasi ini variabel fiskal belum terbukti memengaruhi {y}.")
        else:
            teks = "<br>".join(f"• <b>{n}</b>: kenaikan 1 satuan diasosiasikan dengan {y} "
                               f"{'naik' if r_['Koefisien'] > 0 else 'turun'} {abs(r_['Koefisien']):.4f} satuan "
                               f"(p = {r_['p-value']:.4f}), ceteris paribus." for n, r_ in sig.iterrows())
        st.markdown(f'<div class="catatan">{teks}</div>', unsafe_allow_html=True)

        with st.expander("Diagnostik residual"):
            res, fit = h.resids.values, h.fitted_values.iloc[:, 0].values
            d1, d2 = st.columns(2)
            d1.plotly_chart(tema(px.scatter(x=fit, y=res, labels={"x": "Prediksi", "y": "Residual"},
                                            title="Residual vs Prediksi", color_discrete_sequence=[LAUT]), 340), use_container_width=True)
            d2.plotly_chart(tema(px.histogram(x=res, nbins=30, title="Distribusi Residual",
                                              color_discrete_sequence=[TOSCA]), 340), use_container_width=True)
            jb = stats.jarque_bera(res)
            st.write(f"Jarque-Bera: statistik = {jb.statistic:.3f}, p-value = {jb.pvalue:.4f}")
        with st.expander("Multikolinearitas (VIF)"):
            vif = pd.DataFrame({"Variabel": Xm.columns[1:],
                                "VIF": [variance_inflation_factor(Xm.values, i) for i in range(1, Xm.shape[1])]})
            st.dataframe(vif.sort_values("VIF", ascending=False).style.format({"VIF": "{:.2f}"}), use_container_width=True)
            st.caption("VIF > 10 menandakan multikolinearitas serius.")
        if not satu:
            with st.expander("Uji Hausman (FE vs RE)"):
                try:
                    s_h, dof, p_h = uji_hausman(df, y, xs)
                    st.write(f"Chi-square = {s_h:.3f}, df = {dof}, p-value = {p_h:.4f}")
                    st.write("→ **Fixed Effects** lebih sesuai (p < 0,05)." if p_h < .05 else "→ **Random Effects** dapat dipakai (p ≥ 0,05).")
                except Exception as e:
                    st.warning(f"Uji Hausman tidak dapat dihitung: {e}")
            with st.expander("Perbandingan tiga model"):
                bd = {m: estimasi(df, y, xs, m, False, se)[0].params for m in ["Pooled OLS", "Fixed Effects", "Random Effects"]}
                st.dataframe(pd.DataFrame(bd).round(4), use_container_width=True)
        st.download_button("⬇️ Unduh tabel koefisien (CSV)", tk.to_csv().encode(), "koefisien_regresi.csv", "text/csv")

# ---------------------------------------------------------------
# 10. TAB MACHINE LEARNING
# ---------------------------------------------------------------
with tabs[4]:
    st.markdown("Bandingkan **Regresi Linear** dan **Random Forest** untuk memprediksi variabel Y.")
    m_a, m_b, m_c = st.columns([1, 2, 1])
    y_ml = m_a.selectbox("Variabel target (Y)", opsi_y, key="y_ml")
    x_ml = m_b.multiselect("Fitur (X)", [c for c in fiskal if c != y_ml], default=[c for c in fiskal if c != y_ml], key="x_ml")
    uji = m_c.slider("Porsi data uji", 0.1, 0.4, 0.2, 0.05)
    if satu or len(df) < 30:
        st.warning("Data terlalu sedikit untuk machine learning. Pilih **Semua Kota** agar hasilnya lebih bermakna.")
    if len(x_ml) >= 1 and len(df) >= 10:
        d = df[[y_ml] + x_ml].dropna()
        Xtr, Xte, ytr, yte = train_test_split(d[x_ml], d[y_ml], test_size=uji, random_state=42)
        model = {"Regresi Linear": LinearRegression(),
                 "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)}
        hasil, prediksi = [], {}
        for nama, mdl in model.items():
            mdl.fit(Xtr, ytr)
            p = mdl.predict(Xte)
            prediksi[nama] = p
            hasil.append({"Model": nama, "R² (uji)": r2_score(yte, p),
                          "RMSE": np.sqrt(mean_squared_error(yte, p)), "MAE": mean_absolute_error(yte, p)})
        hasil = pd.DataFrame(hasil).set_index("Model")
        st.dataframe(hasil.style.format("{:.4f}"), use_container_width=True)
        e1, e2 = st.columns(2)
        fi = pd.Series(model["Random Forest"].feature_importances_, index=x_ml).sort_values()
        e1.plotly_chart(tema(px.bar(fi, orientation="h", title="Pentingnya Fitur (Random Forest)",
                                    color_discrete_sequence=[LAUT]).update_layout(showlegend=False), 460), use_container_width=True)
        fa = px.scatter(x=yte, y=prediksi["Random Forest"], labels={"x": "Aktual", "y": "Prediksi"},
                        title="Aktual vs Prediksi (Random Forest)", color_discrete_sequence=[TOSCA])
        fa.add_shape(type="line", x0=yte.min(), y0=yte.min(), x1=yte.max(), y1=yte.max(), line=dict(dash="dash", color="gray"))
        e2.plotly_chart(tema(fa, 460), use_container_width=True)
        terbaik = hasil["R² (uji)"].idxmax()
        st.markdown("### 📝 Interpretasi")
        st.markdown(f'<div class="catatan">Model terbaik pada data uji: <b>{terbaik}</b> '
                    f'(R² = {hasil.loc[terbaik, "R² (uji)"]:.3f}). R² mendekati 0 atau negatif berarti fitur '
                    'belum mampu memprediksi Y lebih baik daripada rata-rata sederhana.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# 11. TAB REGRESI 3D
# ---------------------------------------------------------------
with tabs[5]:
    st.markdown("**Y = b0 + b1·X1 + b2·X2.** Titik = data aktual, bidang = hasil regresi.")
    r1, r2, r3 = st.columns(3)
    x1 = r1.selectbox("Variabel X1", fiskal, index=fiskal.index("PAD") if "PAD" in fiskal else 0)
    x2 = r2.selectbox("Variabel X2", [c for c in fiskal if c != x1], index=2)
    y3 = r3.selectbox("Variabel Y", opsi_y, index=opsi_y.index("IPM") if "IPM" in opsi_y else 0, key="y3d")
    d3 = df[[x1, x2, y3]].dropna()
    if len(d3) < 4:
        st.warning("Data kurang dari 4 observasi.")
    else:
        ols = sm.OLS(d3[y3], sm.add_constant(d3[[x1, x2]])).fit()
        b0, b1, b2 = ols.params
        k1, k2, k3 = st.columns(3)
        k1.metric("R-squared", f"{ols.rsquared:.3f}")
        k2.metric("Adj. R-squared", f"{ols.rsquared_adj:.3f}")
        k3.metric("Prob (F-stat)", f"{ols.f_pvalue:.4f}")
        st.markdown(f'<div class="catatan"><b>Persamaan:</b> {y3} = {b0:.3f} + ({b1:.4f} × {x1}) + ({b2:.4f} × {x2})</div>',
                    unsafe_allow_html=True)
        g1 = np.linspace(d3[x1].min(), d3[x1].max(), 20)
        g2 = np.linspace(d3[x2].min(), d3[x2].max(), 20)
        G1, G2 = np.meshgrid(g1, g2)
        fig = go.Figure([
            go.Surface(x=G1, y=G2, z=b0 + b1 * G1 + b2 * G2, opacity=.65, showscale=False,
                       colorscale=[[0, LANGIT], [1, "#0369a1"]]),
            go.Scatter3d(x=d3[x1], y=d3[x2], z=d3[y3], mode="markers", marker=dict(size=4, color="#0e7490"), name="Data")])
        fig.update_layout(scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=y3))
        st.plotly_chart(tema(fig, 620), use_container_width=True)
        st.markdown("### 📝 Interpretasi")
        sig_f = "signifikan" if ols.f_pvalue < .05 else "tidak signifikan"
        st.markdown(f'<div class="catatan">Secara bersama-sama, {x1} dan {x2} menjelaskan {ols.rsquared * 100:.1f}% variasi {y3} '
                    f'(model {sig_f} pada α = 5%, Prob F = {ols.f_pvalue:.4f}).</div>', unsafe_allow_html=True)
