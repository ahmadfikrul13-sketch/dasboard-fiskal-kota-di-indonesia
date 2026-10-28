"""
=====================================================================
DASHBOARD ANALISIS REGRESI DATA PANEL - DATA FISKAL 97 KOTA (2014-2024)
=====================================================================
Cara menjalankan secara lokal:
    pip install -r requirements.txt
    streamlit run app.py

Fitur utama:
  1. Pemuatan dataset otomatis (atau unggah CSV sendiri)
  2. Filter kota & tahun lewat sidebar
  3. Eksplorasi data: statistik deskriptif, tren, distribusi, korelasi
  4. Regresi data panel: Pooled OLS, Fixed Effects, Random Effects
  5. Uji pendukung: VIF, uji Hausman, diagnostik residual
"""

# ---------------------------------------------------------------
# 1. IMPORT LIBRARY & KONFIGURASI AWAL
# ---------------------------------------------------------------
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # backend non-interaktif (aman untuk server)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Konfigurasi halaman (harus dipanggil pertama kali sebelum perintah st lain)
st.set_page_config(
    page_title="Dashboard Regresi Panel Fiskal",
    page_icon="📊",
    layout="wide",
)

# Tema grafik seaborn agar tampilan konsisten
sns.set_theme(style="whitegrid", context="notebook")

# Nama file dataset default (diletakkan satu folder dengan app.py)
DATASET_DEFAULT = Path(__file__).parent / "dataset_fiskal_97_hipoFull.csv"

# Variabel hasil (dependen) yang lazim dipakai pada dataset ini
KANDIDAT_Y = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]

# Pengelompokan variabel fiskal untuk memudahkan pemilihan di sidebar
GRUP_VARIABEL = {
    "Dana Bagi Hasil (DBH)": ["DBH_PAJAK", "DBH_SDA", "DBH_Lainnya"],
    "Dana Alokasi Umum (DAU)": ["DAU_Grant", "DAU_Earmark"],
    "Dana Alokasi Khusus (DAK)": ["DAK_Fisik", "DAK_Nonfisik"],
    "PAD & ICT": ["PAD", "ICT"],
    "Belanja Modal (bm_)": [
        "bm_pendidikan", "bm_kesehatan", "bm_perumahan", "bm_lingkungan",
        "bm_ekonomi", "bm_infrastruktur", "bm_pariwisata",
        "bm_perlindungan_sosial",
    ],
}


# ---------------------------------------------------------------
# 2. PEMUATAN DATASET (di-cache agar tidak dibaca ulang tiap interaksi)
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Memuat dataset...")
def muat_data(sumber) -> pd.DataFrame:
    """Baca CSV dari path atau file unggahan, lalu rapikan tipe datanya."""
    df = pd.read_csv(sumber)
    df.columns = df.columns.str.strip()          # buang spasi tersembunyi di nama kolom
    df["Kota"] = df["Kota"].astype(str).str.strip()
    df["Tahun"] = df["Tahun"].astype(int)
    return df.sort_values(["Kota", "Tahun"]).reset_index(drop=True)


# ---------------------------------------------------------------
# 3. FUNGSI-FUNGSI REGRESI
# ---------------------------------------------------------------
def siapkan_panel(df, y, xs):
    """Ubah data menjadi format panel (indeks bertingkat: Kota, Tahun)."""
    data = df[["Kota", "Tahun", y] + xs].dropna().set_index(["Kota", "Tahun"])
    Y = data[y]
    X = sm.add_constant(data[xs], has_constant="add")
    return Y, X


def estimasi_model(df, y, xs, jenis, efek_waktu, jenis_se):
    """
    Estimasi model panel.
      jenis      : 'Pooled OLS' | 'Fixed Effects' | 'Random Effects'
      efek_waktu : True -> tambahkan efek tetap tahun (hanya untuk Fixed Effects)
      jenis_se   : 'Clustered (per kota)' | 'Robust' | 'Biasa'
    """
    Y, X = siapkan_panel(df, y, xs)

    # Pilihan cara menghitung standard error
    if jenis_se == "Clustered (per kota)":
        opsi = dict(cov_type="clustered", cluster_entity=True)
    elif jenis_se == "Robust":
        opsi = dict(cov_type="robust")
    else:
        opsi = dict(cov_type="unadjusted")

    if jenis == "Pooled OLS":
        model = PooledOLS(Y, X)
    elif jenis == "Fixed Effects":
        model = PanelOLS(Y, X, entity_effects=True, time_effects=efek_waktu)
    else:
        model = RandomEffects(Y, X)
    return model.fit(**opsi)


def tabel_koefisien(hasil) -> pd.DataFrame:
    """Susun koefisien, SE, t-stat, p-value, dan interval kepercayaan 95%."""
    ci = hasil.conf_int()
    tabel = pd.DataFrame({
        "Koefisien": hasil.params,
        "Std. Error": hasil.std_errors,
        "t-stat": hasil.tstats,
        "p-value": hasil.pvalues,
        "CI 95% bawah": ci.iloc[:, 0],
        "CI 95% atas": ci.iloc[:, 1],
    })

    def bintang(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    tabel["Sig."] = tabel["p-value"].apply(bintang)
    return tabel


def hitung_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor untuk mendeteksi multikolinearitas."""
    nilai = X.values
    vif = [variance_inflation_factor(nilai, i) for i in range(1, nilai.shape[1])]
    return pd.DataFrame({"Variabel": X.columns[1:], "VIF": vif}).sort_values(
        "VIF", ascending=False
    )


def uji_hausman(df, y, xs):
    """
    Uji Hausman: H0 = Random Effects konsisten (pilih RE),
                 H1 = RE tidak konsisten (pilih Fixed Effects).
    Memakai standard error biasa (syarat uji klasik).
    """
    fe = estimasi_model(df, y, xs, "Fixed Effects", False, "Biasa")
    re = estimasi_model(df, y, xs, "Random Effects", False, "Biasa")
    umum = [k for k in fe.params.index if k != "const" and k in re.params.index]
    selisih_b = (fe.params[umum] - re.params[umum]).values
    selisih_v = (fe.cov.loc[umum, umum] - re.cov.loc[umum, umum]).values
    stat = float(selisih_b @ np.linalg.pinv(selisih_v) @ selisih_b)
    p = float(stats.chi2.sf(stat, len(umum)))
    return stat, len(umum), p


def ambil(hasil, nama, default=np.nan):
    """Ambil atribut hasil model dengan aman (tidak semua model punya semua atribut)."""
    try:
        return getattr(hasil, nama)
    except Exception:
        return default


# ---------------------------------------------------------------
# 4. SIDEBAR - SUMBER DATA, FILTER, DAN PENGATURAN MODEL
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")

# 4a. Sumber data
st.sidebar.subheader("1. Sumber Data")
file_unggah = st.sidebar.file_uploader("Unggah CSV lain (opsional)", type="csv")

if file_unggah is not None:
    df_asli = muat_data(file_unggah)
elif DATASET_DEFAULT.exists():
    df_asli = muat_data(str(DATASET_DEFAULT))
else:
    st.error(
        f"File `{DATASET_DEFAULT.name}` tidak ditemukan. Letakkan di folder yang sama "
        "dengan app.py atau unggah lewat sidebar."
    )
    st.stop()

# Deteksi kolom numerik (selain Tahun) secara otomatis
kolom_numerik = [
    c for c in df_asli.select_dtypes(include="number").columns if c != "Tahun"
]
semua_fiskal = [v for g in GRUP_VARIABEL.values() for v in g if v in kolom_numerik]

# 4b. Filter data
st.sidebar.subheader("2. Filter Data")
tahun_min, tahun_maks = int(df_asli["Tahun"].min()), int(df_asli["Tahun"].max())
rentang_tahun = st.sidebar.slider(
    "Rentang tahun", tahun_min, tahun_maks, (tahun_min, tahun_maks)
)
daftar_kota = sorted(df_asli["Kota"].unique())
kota_dipilih = st.sidebar.multiselect(
    "Kota (kosongkan = semua)", daftar_kota, default=[]
)

df = df_asli[df_asli["Tahun"].between(*rentang_tahun)]
if kota_dipilih:
    df = df[df["Kota"].isin(kota_dipilih)]

# 4c. Spesifikasi model
st.sidebar.subheader("3. Spesifikasi Model")
opsi_y = [v for v in KANDIDAT_Y if v in kolom_numerik] or kolom_numerik
var_y = st.sidebar.selectbox("Variabel dependen (Y)", opsi_y)

# Pilih variabel independen per kelompok agar rapi
var_x = []
for nama_grup, daftar in GRUP_VARIABEL.items():
    tersedia = [v for v in daftar if v in kolom_numerik and v != var_y]
    if tersedia:
        var_x += st.sidebar.multiselect(nama_grup, tersedia, default=tersedia)

jenis_model = st.sidebar.radio(
    "Jenis model", ["Pooled OLS", "Fixed Effects", "Random Effects"], index=1
)
efek_waktu = False
if jenis_model == "Fixed Effects":
    efek_waktu = st.sidebar.checkbox("Tambahkan efek tetap tahun (two-way)", value=False)
jenis_se = st.sidebar.selectbox(
    "Standard error", ["Clustered (per kota)", "Robust", "Biasa"]
)

# ---------------------------------------------------------------
# 5. HEADER & RINGKASAN DATA
# ---------------------------------------------------------------
st.title("📊 Dashboard Regresi Data Panel Fiskal")
st.caption(
    "Analisis pengaruh transfer fiskal, PAD, ICT, dan belanja modal terhadap "
    "indikator pembangunan kota."
)

if df.empty:
    st.warning("Tidak ada data dengan filter ini. Ubah filter di sidebar.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah observasi", f"{len(df):,}")
k2.metric("Jumlah kota", df["Kota"].nunique())
k3.metric("Periode", f"{df['Tahun'].min()} - {df['Tahun'].max()}")
k4.metric("Jumlah variabel", len(kolom_numerik))

tab_data, tab_eksplor, tab_regresi, tab_unduh = st.tabs(
    ["🗂️ Data", "🔍 Eksplorasi", "📈 Regresi", "⬇️ Unduh"]
)

# ---------------------------------------------------------------
# 6. TAB DATA
# ---------------------------------------------------------------
with tab_data:
    st.subheader("Tabel Data")
    st.dataframe(df, use_container_width=True, height=350)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df[kolom_numerik].describe().T.round(3), use_container_width=True)

    # Cek kelengkapan panel (apakah setiap kota punya jumlah tahun yang sama)
    obs_per_kota = df.groupby("Kota")["Tahun"].count()
    status = "seimbang (balanced)" if obs_per_kota.nunique() == 1 else "tidak seimbang (unbalanced)"
    st.info(f"Struktur panel: **{status}**, rata-rata {obs_per_kota.mean():.1f} tahun per kota.")

# ---------------------------------------------------------------
# 7. TAB EKSPLORASI (grafik interaktif lewat widget)
# ---------------------------------------------------------------
with tab_eksplor:
    kiri, kanan = st.columns(2)

    # 7a. Tren rata-rata per tahun
    with kiri:
        st.subheader("Tren per Tahun")
        var_tren = st.selectbox("Variabel", kolom_numerik, index=kolom_numerik.index(var_y))
        kota_tren = st.multiselect(
            "Bandingkan dengan kota tertentu", sorted(df["Kota"].unique()), max_selections=5
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        rata = df.groupby("Tahun")[var_tren].mean()
        ax.plot(rata.index, rata.values, lw=3, color="black", label="Rata-rata semua kota")
        for k in kota_tren:
            sub = df[df["Kota"] == k]
            ax.plot(sub["Tahun"], sub[var_tren], marker="o", alpha=0.8, label=k)
        ax.set_xlabel("Tahun")
        ax.set_ylabel(var_tren)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    # 7b. Distribusi
    with kanan:
        st.subheader("Distribusi Variabel")
        var_dist = st.selectbox("Variabel ", kolom_numerik, index=kolom_numerik.index(var_y))
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[var_dist], kde=True, ax=ax, color="steelblue")
        ax.set_xlabel(var_dist)
        st.pyplot(fig)
        plt.close(fig)

    # 7c. Scatter plot X vs Y dengan garis tren
    st.subheader("Hubungan Dua Variabel")
    c1, c2 = st.columns(2)
    sumbu_x = c1.selectbox("Sumbu X", [v for v in kolom_numerik if v != var_y])
    sumbu_y = c2.selectbox("Sumbu Y", kolom_numerik, index=kolom_numerik.index(var_y))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.regplot(
        data=df, x=sumbu_x, y=sumbu_y, ax=ax,
        scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "crimson"},
    )
    st.pyplot(fig)
    plt.close(fig)

    # 7d. Heatmap korelasi
    st.subheader("Matriks Korelasi")
    var_korelasi = st.multiselect(
        "Variabel yang dikorelasikan", kolom_numerik, default=[var_y] + var_x[:8]
    )
    if len(var_korelasi) >= 2:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            df[var_korelasi].corr(), annot=len(var_korelasi) <= 12, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("Pilih minimal dua variabel.")

# ---------------------------------------------------------------
# 8. TAB REGRESI
# ---------------------------------------------------------------
with tab_regresi:
    st.subheader(f"Model {jenis_model}: {var_y} ~ variabel fiskal")

    # Validasi sebelum estimasi
    if not var_x:
        st.warning("Pilih minimal satu variabel independen di sidebar.")
        st.stop()
    if df["Kota"].nunique() < 2 or len(df) <= len(var_x) + 5:
        st.warning("Data terlalu sedikit untuk estimasi. Perluas filter kota/tahun.")
        st.stop()

    try:
        hasil = estimasi_model(df, var_y, var_x, jenis_model, efek_waktu, jenis_se)
    except Exception as e:
        st.error(f"Estimasi gagal: {e}")
        st.stop()

    # 8a. Ringkasan kecocokan model
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (within)", f"{ambil(hasil, 'rsquared_within'):.4f}")
    m2.metric("R² (overall)", f"{ambil(hasil, 'rsquared_overall'):.4f}")
    try:
        m3.metric("F-statistik", f"{hasil.f_statistic.stat:.2f}")
        m4.metric("Prob (F)", f"{hasil.f_statistic.pval:.4f}")
    except Exception:
        m3.metric("F-statistik", "-")
        m4.metric("Prob (F)", "-")

    # 8b. Tabel koefisien
    st.markdown("#### Tabel Koefisien")
    tab_koef = tabel_koefisien(hasil)
    st.dataframe(
        tab_koef.style.format(
            {"Koefisien": "{:.4f}", "Std. Error": "{:.4f}", "t-stat": "{:.3f}",
             "p-value": "{:.4f}", "CI 95% bawah": "{:.4f}", "CI 95% atas": "{:.4f}"}
        ),
        use_container_width=True,
    )
    st.caption("Signifikansi: *** p<0,01 | ** p<0,05 | * p<0,10")

    # 8c. Plot koefisien dengan interval kepercayaan
    st.markdown("#### Plot Koefisien (CI 95%)")
    kt = tab_koef.drop(index="const", errors="ignore").sort_values("Koefisien")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(kt))))
    warna = np.where(kt["p-value"] < 0.05, "crimson", "gray")
    ax.hlines(kt.index, kt["CI 95% bawah"], kt["CI 95% atas"], color=warna, lw=2)
    ax.scatter(kt["Koefisien"], kt.index, color=warna, zorder=3)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Koefisien (merah = signifikan 5%)")
    st.pyplot(fig)
    plt.close(fig)

    # 8d. Interpretasi otomatis
    st.markdown("#### Interpretasi Singkat")
    signif = kt[kt["p-value"] < 0.05]
    if signif.empty:
        st.write("Tidak ada variabel yang signifikan pada tingkat 5% pada spesifikasi ini.")
    else:
        for nama, baris in signif.iterrows():
            arah = "naik" if baris["Koefisien"] > 0 else "turun"
            st.write(
                f"- **{nama}**: kenaikan 1 satuan diasosiasikan dengan {var_y} yang "
                f"**{arah} {abs(baris['Koefisien']):.4f}** satuan (ceteris paribus, p = {baris['p-value']:.4f})."
            )

    # 8e. Diagnostik: residual, VIF, Hausman
    st.markdown("#### Diagnostik Model")
    d1, d2 = st.columns(2)
    resid = hasil.resids.values
    fitted = hasil.fitted_values.iloc[:, 0].values

    with d1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(fitted, resid, alpha=0.4, s=15)
        ax.axhline(0, color="crimson", ls="--")
        ax.set_xlabel("Nilai prediksi")
        ax.set_ylabel("Residual")
        ax.set_title("Residual vs Prediksi")
        st.pyplot(fig)
        plt.close(fig)

    with d2:
        fig, ax = plt.subplots(figsize=(6, 4))
        stats.probplot(resid, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot Residual")
        st.pyplot(fig)
        plt.close(fig)

    uji_jb = stats.jarque_bera(resid)
    st.write(f"Uji normalitas Jarque-Bera: statistik = {uji_jb.statistic:.3f}, p-value = {uji_jb.pvalue:.4f}")

    with st.expander("Multikolinearitas (VIF)"):
        _, X_vif = siapkan_panel(df, var_y, var_x)
        tabel_vif = hitung_vif(X_vif)
        st.dataframe(tabel_vif.style.format({"VIF": "{:.2f}"}), use_container_width=True)
        st.caption("Aturan praktis: VIF > 10 mengindikasikan multikolinearitas serius.")

    with st.expander("Uji Hausman (Fixed Effects vs Random Effects)"):
        try:
            stat_h, dof, p_h = uji_hausman(df, var_y, var_x)
            st.write(f"Chi-square = {stat_h:.3f}, df = {dof}, p-value = {p_h:.4f}")
            if p_h < 0.05:
                st.success("p < 0,05 → tolak H0: **Fixed Effects** lebih sesuai.")
            else:
                st.info("p ≥ 0,05 → gagal tolak H0: **Random Effects** dapat dipakai.")
        except Exception as e:
            st.warning(f"Uji Hausman tidak dapat dihitung: {e}")

    with st.expander("Perbandingan tiga model"):
        kolom_banding = {}
        for nama_model in ["Pooled OLS", "Fixed Effects", "Random Effects"]:
            h = estimasi_model(df, var_y, var_x, nama_model, False, jenis_se)
            kolom_banding[nama_model] = h.params
        st.dataframe(pd.DataFrame(kolom_banding).round(4), use_container_width=True)

    with st.expander("Output lengkap (teks)"):
        st.code(str(hasil.summary))

# ---------------------------------------------------------------
# 9. TAB UNDUH
# ---------------------------------------------------------------
with tab_unduh:
    st.subheader("Unduh Hasil")
    st.download_button(
        "Unduh data terfilter (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="data_terfilter.csv",
        mime="text/csv",
    )
    if "tab_koef" in globals():
        st.download_button(
            "Unduh tabel koefisien (CSV)",
            tab_koef.to_csv().encode("utf-8"),
            file_name="koefisien_regresi.csv",
            mime="text/csv",
        )
