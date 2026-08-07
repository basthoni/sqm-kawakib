import streamlit as st
import sys
import os
import re
import zipfile
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# Pastikan ephem terinstal
try:
    import ephem
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ephem"])
    import ephem

# Modul untuk Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# =====================================================================
# KONFIGURASI HALAMAN WEB & LINK PERMANEN
# =====================================================================
st.set_page_config(page_title="Kawakib SQM Analyzer", page_icon="🌌", layout="wide")

# Masukkan link Google Sheets permanen Anda di sini
GSHEETS_PERMANEN_URL = "https://docs.google.com/spreadsheets/d/1E4RpTfcPeQorW3r9cjpZ5cp31dpa7N_oXRZksRWdxG4/edit?gid=0#gid=0"

# Masukkan link sample data Google Drive Anda di sini
SAMPLE_DATA_DRIVE_URL = "https://drive.google.com/drive/folders/1KHg8dRtkt9KrdDFZ8esbiuHQtKJvP2AN?usp=drive_link"

if 'history_plot' not in st.session_state:
    st.session_state.history_plot = []

st.title("🌌 KAWAKIB INSTITUTE: Otonom SQM & Fajar Analyzer")
st.markdown("""
Aplikasi web ini menggunakan algoritma **SIGMAG-STAB** atau pemodelan **SIGMOID** dinamis untuk mengekstrak titik belok fajar sadiq, menyaring gangguan bulan, dan menganalisis tutupan awan. 
Data hasil analisis tersinkronisasi secara otomatis ke basis data cloud Google Sheets.
""")

# =====================================================================
# FUNGSI KONEKSI & OPERASI GOOGLE SHEETS
# =====================================================================
def get_gsheets_client():
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        return None

def save_to_google_sheets(data_dict):
    client = get_gsheets_client()
    if client is None:
        return False, "Kredensial GSheets belum diatur di Streamlit Secrets."
    try:
        sheet = client.open_by_url(GSHEETS_PERMANEN_URL).sheet1
        existing_data = sheet.get_all_values()
        
        # Jika sheet kosong, buat header terlebih dahulu
        if not existing_data:
            header = list(data_dict.keys())
            sheet.append_row(header)
            
        row_data = [str(data_dict.get(key, "")) for key in data_dict.keys()]
        sheet.append_row(row_data)
        return True, "Berhasil"
    except Exception as e:
        return False, str(e)

def load_data_from_google_sheets():
    client = get_gsheets_client()
    if client is None:
        return pd.DataFrame()
    try:
        sheet = client.open_by_url(GSHEETS_PERMANEN_URL).sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

# =====================================================================
# FUNGSI MATEMATIKA & ASTRONOMI
# =====================================================================
def read_header_and_find_data_start(path, max_header_lines=80):
    header, data_start = list(), None
    site, lat, lon, utc_offset = "Unknown Site", None, None, 7
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i in range(max_header_lines):
            line = f.readline()
            if not line: break
            s = line.strip()
            header.append(s)
            if data_start is None and not s.startswith("#") and ";" in s:
                data_start = i
                break
    for line in header:
        if "Location name:" in line: site = line.split("Location name:")[-1].strip()
        if "Position:" in line or "Position" in line:
            nums = re.findall(r"[+-]?\d+\.?\d*", line)
            if len(nums) >= 2:
                lat, lon = float(nums[0]), float(nums[1])
        if "Local timezone:" in line:
            m = re.search(r"UTC([+-]\d+)", line)
            if m: utc_offset = int(m.group(1))
    if data_start is None: data_start = 35
    return site, lat, lon, utc_offset, data_start

def solar_alt(local_dt, lat, lon, utc_offset):
    dt_utc = local_dt - pd.Timedelta(hours=utc_offset)
    jd = dt_utc.values.astype("datetime64[ns]").astype("int64") / 86400000000000 + 2440587.5
    n = jd - 2451545.0
    g = np.deg2rad((357.529 + 0.98560028 * n) % 360)
    L = (280.459 + 0.98564736 * n) % 360
    lam = np.deg2rad(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.deg2rad(23.439 - 0.00000036 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    E = (L - np.rad2deg(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)))) / 15.0
    hr = dt_utc.dt.hour + dt_utc.dt.minute / 60 + dt_utc.dt.second / 3600
    LST = hr + lon / 15.0 + E
    H = np.deg2rad((LST * 15.0) - 180.0)
    latr = np.deg2rad(lat)
    alt = np.arcsin(np.sin(latr) * np.sin(dec) + np.cos(latr) * np.cos(dec) * np.cos(H))
    return np.rad2deg(alt)

def load_sqm_data(file_path):
    site, lat, lon, utc_offset, data_start = read_header_and_find_data_start(file_path)
    if lat is None or lon is None: lat, lon, utc_offset = -7.972, 114.425, 7
    df = pd.read_csv(file_path, skiprows=data_start, sep=";", header=None,
                   names=["utc","local","temp","cnt","hz","mpsas"],
                   engine="python", on_bad_lines="skip")
    df["local_dt"] = pd.to_datetime(df["local"], errors="coerce")
    df = df.dropna(subset=["local_dt","mpsas"])
    df["sun_alt"] = solar_alt(df["local_dt"], lat, lon, utc_offset)
    am = df[(df["local_dt"].dt.hour < 12) & (df["mpsas"] > 0)].copy()
    am = am.sort_values("sun_alt").reset_index(drop=True)
    date_str = am["local_dt"].iloc[0].strftime("%Y-%m-%d") if not am.empty else "Unknown"
    return am, site, lat, lon, utc_offset, date_str

def apply_moonlight_correction(am, lat, lon, utc_offset):
    obs = ephem.Observer()
    obs.lat, obs.lon = str(lat), str(lon)
    moon = ephem.Moon()
    corrected_mpsas = list()
    is_corrected = False
    for _, row in am.iterrows():
        utc_time = row["local_dt"] - pd.Timedelta(hours=utc_offset)
        obs.date = utc_time.strftime('%Y/%m/%d %H:%M:%S')
        moon.compute(obs)
        if np.rad2deg(moon.alt) > 0 and (moon.phase / 100.0) > 0.05:
            is_corrected = True
            I_tot = 10 ** (-0.4 * row["mpsas"])
            I_moon = ((moon.phase / 100.0) * np.sin(moon.alt)) * (10 ** (-0.4 * 21.5))
            corrected_mpsas.append(-2.5 * np.log10(max(I_tot - I_moon, 10 ** (-0.4 * 22.0))))
        else:
            corrected_mpsas.append(row["mpsas"])
    am["mpsas_corrected"] = corrected_mpsas
    return am, is_corrected

def analyze_cloud_cover(am, onset_alt, window_minutes=60):
    if onset_alt is None: return 0.0, pd.DataFrame()
    onset_idx = (np.abs(am["sun_alt"] - onset_alt)).argmin()
    onset_dt = am["local_dt"].iloc[onset_idx]
    mask = (am["local_dt"] >= onset_dt - pd.Timedelta(minutes=window_minutes)) & \
           (am["local_dt"] <= onset_dt + pd.Timedelta(minutes=window_minutes))
    df_win = am[mask].copy()
    if len(df_win) < 21: return 0.0, df_win

    df_win['rolling_std'] = df_win['mpsas_corrected'].rolling(21, center=True).std()
    dyn_thresh = max((-0.04545 * df_win['mpsas_corrected'].mean()) + 1.0500, 0.05)
    awan_cepat = df_win['rolling_std'] > dyn_thresh

    base_series = am[am["sun_alt"] < -20]["mpsas_corrected"]
    garis_dasar = base_series.median() if not base_series.empty else am["mpsas_corrected"].max()
    margin_deviasi = 0.2
    awan_lambat = (np.abs(df_win['mpsas_corrected'] - garis_dasar) > margin_deviasi) & (df_win['sun_alt'] < onset_alt)

    df_win['is_cloudy'] = awan_cepat | awan_lambat
    return (df_win['is_cloudy'].mean() * 100), df_win

def categorize_light_pollution(baseline_mpsas):
    if baseline_mpsas >= 21.3: cat = "Gelap / Dark (Tipe 1)"
    elif 20.2 <= baseline_mpsas < 21.3: cat = "Agak Gelap / Slightly Dark (Tipe 2)"
    elif 19.1 <= baseline_mpsas < 20.2: cat = "Agak Terang / Slightly Bright (Tipe 3)"
    else: cat = "Terang / Urban Skyglow (Tipe 4)"
    return cat, (baseline_mpsas - 11.12) / 0.55

def get_dynamic_params(am):
    if len(am) < 2: return 0.25, 5
    delta_s = am["local_dt"].diff().median().total_seconds()
    if delta_s <= 65: return 0.25, 5
    elif delta_s <= 125: return 0.5, 4
    elif delta_s <= 305: return 0.75, 3
    else: return 1.0, 3

def bin_alt(x, y, bin_deg):
    bins = np.floor((x + 90.0) / bin_deg) * bin_deg - 90.0
    dfb = pd.DataFrame({"bin": bins, "x": x, "y": y})
    g = dfb.groupby("bin", sort=True).agg(x=("x", "median"), y=("y", "median")).reset_index(drop=True)
    return g["x"].values, g["y"].values

def mad_sigma(arr):
    arr = np.asarray(arr)
    return 1.4826 * np.median(np.abs(arr - np.median(arr)))

def analyze_sigmag(am, bin_deg, n_consec):
    am["mpsas_smooth"] = savgol_filter(am["mpsas_corrected"], window_length=min(31, len(am)|1), polyorder=2)
    xb, yb = bin_alt(am["sun_alt"].values, am["mpsas_smooth"].values, bin_deg)
    grad_binned = np.gradient(yb, xb)
    grad_mean = pd.Series(pd.Series(grad_binned).rolling(5, center=True, min_periods=1).median().values)\
                .rolling(9, center=True, min_periods=1).mean().values

    g_base = grad_mean[xb < -20.0]
    if len(g_base) < 5: return None, None
    mu_grad, sigma_used = float(np.mean(g_base)), max(float(mad_sigma(g_base)), 0.01)

    k_factor = 1.0 if sigma_used < 0.02 else (1.2 if sigma_used < 0.05 else 1.5)
    threshold = mu_grad - k_factor * sigma_used

    search_mask = xb >= -20.0
    xs, gs = xb[search_mask], grad_mean[search_mask]

    onset_idx, consec_count = None, 0
    for i, flag in enumerate(gs < threshold):
        if flag:
            consec_count += 1
            if consec_count >= n_consec:
                onset_idx = i - n_consec + 1
                break
        else: consec_count = 0

    if onset_idx is not None and len(xs) > 0:
        alt = float(xs[onset_idx])
        return alt, float(np.interp(alt, am["sun_alt"], am["mpsas_corrected"]))
    return None, None

def analyze_sigmoid(am):
    def sigmoid(x, L, x0, k, b): return L / (1 + np.exp(-k * (x - x0))) + b
    am["mpsas_smooth"] = savgol_filter(am["mpsas_corrected"], window_length=min(31, len(am)|1), polyorder=2)
    x_data, y_data = am["sun_alt"].values, am["mpsas_smooth"].values
    try:
        p0 = [y_data.min() - y_data.max(), -15.0, 1.0, y_data.max()]
        popt, _ = curve_fit(sigmoid, x_data, y_data, p0=p0, maxfev=5000)
        x_eval = np.linspace(-30, -5, 500)
        y_eval = sigmoid(x_eval, *popt)
        garis_dasar_teoretis = popt[3]
        ambang_batas = garis_dasar_teoretis - 0.15
        onset_idx = np.argmax(y_eval < ambang_batas)
        if onset_idx == 0: return None, None
        alt = x_eval[onset_idx]
        return alt, float(np.interp(alt, am["sun_alt"], am["mpsas_corrected"]))
    except:
        return None, None

# =====================================================================
# UI KONTROL & SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    method = st.selectbox("Metode Analisis", ["SIGMAG-STAB", "SIGMOID"])
    st.info("Unggah file mentah SQM (.dat) atau arsip .zip untuk analisis batch.")
    
    st.divider()
    st.markdown("### 📂 Sample Data Uji Coba")
    st.markdown(f"[🔗 Unduh Sample Data]({SAMPLE_DATA_DRIVE_URL})")

# Tab Antarmuka Utama
tab_analisis, tab_histori, tab_algoritma = st.tabs(["🚀 Analisis Baru", "📜 Histori Cloud (Google Sheets)", "📖 Penjelasan Algoritma"])

# =====================================================================
# TAB 1: PROSES ANALISIS & SIMPAN KE CLOUD
# =====================================================================
with tab_analisis:
    uploaded_files = st.file_uploader("Unggah File Data SQM", accept_multiple_files=True, type=['dat', 'txt', 'zip'])

    if uploaded_files:
        if st.button("Jalankan Analisis 🚀"):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = list()
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if uploaded_file.name.endswith('.zip'):
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                            for root, dirs, files_in_dir in os.walk(temp_dir):
                                for file in files_in_dir:
                                    if file.endswith(('.dat', '.txt')):
                                        file_paths.append(os.path.join(root, file))
                    else:
                        file_paths.append(file_path)

                if not file_paths:
                    st.error("❌ Tidak ada file .dat yang ditemukan.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_files = len(file_paths)
                    
                    for idx, path in enumerate(file_paths):
                        status_text.text(f"Memproses file {idx+1} dari {total_files}: {os.path.basename(path)}")
                        try:
                            am, site, lat, lon, utc_offset, date_str = load_sqm_data(path)
                            if am.empty: continue
                            
                            am, is_corrected = apply_moonlight_correction(am, lat, lon, utc_offset)
                            bin_deg, n_consec = get_dynamic_params(am)

                            if method == "SIGMAG-STAB":
                                onset_alt, onset_msas = analyze_sigmag(am, bin_deg, n_consec)
                            else:
                                onset_alt, onset_msas = analyze_sigmoid(am)
                            
                            cloud_pct, df_win = analyze_cloud_cover(am, onset_alt)
                            base_series = am[am["sun_alt"] < -20]["mpsas_corrected"]
                            baseline_mpsas = base_series.median() if not base_series.empty else am["mpsas_corrected"].max()
                            lp_category, expected_alt = categorize_light_pollution(baseline_mpsas)
                            
                            rekap_data = {
                                "Tanggal": date_str, "Lokasi": site, "Metode": method,
                                "Bortle": lp_category.split("(")[-1].replace(")",""),
                                "Awan_%": round(cloud_pct, 1), "Koreksi_Bulan": "Aktif" if is_corrected else "Pasif",
                                "Garis_Dasar": round(baseline_mpsas, 2),
                                "Fajar_Alt": round(onset_alt, 2) if onset_alt is not None else "",
                                "Fajar_MSAS": round(onset_msas, 2) if onset_msas is not None else ""
                            }

                            # Sinkronisasi otomatis ke Google Sheets Permanen
                            save_to_google_sheets(rekap_data)
                                    
                            # Visualisasi Plot
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(am["sun_alt"], am["mpsas_corrected"], color="black", alpha=0.7, label="SQM Terkoreksi")
                            if is_corrected:
                                ax.plot(am["sun_alt"], am["mpsas"], color="grey", alpha=0.3, linestyle=":", label="SQM Mentah")
                            
                            if onset_alt is not None:
                                ax.axvline(onset_alt, color="red", linestyle="--", label=f"Titik Belok ({onset_alt:.2f}°)")
                                ax.scatter([onset_alt], [onset_msas], color="red", zorder=5)
                            
                            cloudy_points = df_win[df_win['is_cloudy'] == True] if 'is_cloudy' in df_win.columns else pd.DataFrame()
                            if not cloudy_points.empty:
                                ax.scatter(cloudy_points["sun_alt"], cloudy_points["mpsas_corrected"], color="blue", s=15, label="Awan", zorder=4)
                                
                            ax.invert_yaxis()
                            ax.set_xlim(-30, -5)
                            ax.set_xlabel("Ketinggian Matahari (Derajat)")
                            ax.set_ylabel("Kecerlangan (Mpsas)")
                            ax.set_title(f"{site} | {date_str} [{method}]")
                            ax.grid(True, linestyle=":", alpha=0.7)
                            
                            onset_str = f"{onset_alt:.2f}°" if onset_alt is not None else "Gagal"
                            info_text = (
                                f"Garis Dasar : {baseline_mpsas:.2f} Mpsas\n"
                                f"Awan / Bulan: {cloud_pct:.1f}% / {'Aktif' if is_corrected else 'Pasif'}\n"
                                f"Titik Fajar : {onset_str}"
                            )
                            props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
                            ax.text(0.02, baseline_mpsas - 1.2, info_text, transform=ax.get_yaxis_transform(), 
                                    fontsize=9, verticalalignment='bottom', bbox=props, family='monospace')
                            ax.legend(loc="upper right")
                            
                            st.pyplot(fig)
                            st.session_state.history_plot.append((f"{site} - {date_str} [{method}]", fig))
                            
                        except Exception as e:
                            st.error(f"Gagal memproses {os.path.basename(path)}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / total_files)
                
                    status_text.text("")
                    st.success("🎉 Analisis selesai! Data telah otomatis tersimpan permanen di Google Sheets.")

# =====================================================================
# TAB 2: HISTORI CLOUD (MENGAMBIL LANGSUNG DARI GOOGLE SHEETS)
# =====================================================================
with tab_histori:
    st.header("📜 Histori & Basis Data Cloud (Google Sheets)")
    st.markdown("Tabel di bawah ini ditarik secara *real-time* langsung dari file Google Sheets pusat Anda.")
    
    if st.button("🔄 Muat Ulang Data dari Cloud"):
        st.rerun()
        
    df_cloud = load_data_from_google_sheets()
    
    if df_cloud.empty:
        st.info("Belum ada data di Google Sheets atau koneksi belum terkonfigurasi dengan benar.")
    else:
        st.dataframe(df_cloud, use_container_width=True)
        
        csv_cloud = df_cloud.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Unduh Data Cloud (CSV)",
            data=csv_cloud,
            file_name='Rekap_Kawakib_Cloud.csv',
            mime='text/csv',
        )
        
        st.divider()
        st.subheader("📈 Arsip Grafik Sesi Ini")
        if not st.session_state.history_plot:
            st.info("Belum ada grafik yang di-generate pada sesi penjelajahan ini.")
        else:
            for title, figure in st.session_state.history_plot:
                with st.expander(f"Lihat Grafik: {title}"):
                    st.pyplot(figure)

# =====================================================================
# TAB 3: PENJELASAN ALGORITMA
# =====================================================================
with tab_algoritma:
    st.header("📖 Landasan Algoritma Ekstraksi Fajar")
    st.markdown("Aplikasi ini beroperasi menggunakan dua pendekatan matematis utama untuk menentukan titik belok (onset) fajar sadiq dari kurva penurunan kecerlangan langit malam.")
    
    st.subheader("1. Metode SIGMAG-STAB")
    st.markdown("""
    Metode ini bekerja dengan menganalisis laju perubahan (gradien) dari kurva kecerlangan langit terhadap perubahan ketinggian matahari. 
    
    *   **Proses Smoothing**: Data disaring menggunakan filter Savitzky-Golay untuk mereduksi *noise*.
    *   **Ambang Batas Dinamis**: Titik fajar diekstrak saat gradien jatuh melebihi ambang batas statistik dinamis:
    """)
    st.latex(r"T = \mu - (k \cdot \sigma)")
    st.markdown("""
    Di mana:
    *   $\mu$ adalah rata-rata gradien saat malam gelap total (Matahari $<-20^\circ$).
    *   $\sigma$ adalah penyimpangan absolut median (*Median Absolute Deviation*) gradien.
    *   $k$ adalah faktor pengali adaptif ($1.0 \leq k \leq 1.5$).
    """)

    st.divider()

    st.subheader("2. Metode SIGMOID")
    st.markdown("Alih-alih mencari turunan, metode ini memodelkan keseluruhan kurva penurunan kecerlangan menggunakan **Fungsi Logistik (Sigmoid)** melalui teknik *non-linear least squares fitting*.")
    st.latex(r"y = \frac{L}{1 + e^{-k(x - x_0)}} + b")
    st.markdown("""
    Di mana:
    *   $y$ = Kecerlangan langit (Mpsas)
    *   $x$ = Ketinggian matahari (Derajat)
    *   $L$ = Amplitudo kurva
    *   $k$ = Kelandaian kurva saat fajar
    *   $x_0$ = Titik infleksi (tengah transisi fajar)
    *   $b$ = Garis dasar kecerlangan malam (*baseline*)
    
    Awal fajar (onset) dikunci pada titik di mana model kurva pertama kali menyimpang sebesar 0.15 Mpsas dari asimtot garis dasar teoritis ($b$).
    """)
