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

# =====================================================================
# KONFIGURASI HALAMAN WEB
# =====================================================================
st.set_page_config(page_title="Kawakib SQM Analyzer", page_icon="🌌", layout="wide")

st.title("🌌 KAWAKIB INSTITUTE: Otonom SQM & Fajar Analyzer")
st.markdown("""
Aplikasi web ini menggunakan algoritma **SIGMAG-STAB** atau **SIGMOID** dan pemodelan awan dinamis Cavazzani untuk mengekstrak titik belok fajar sadiq, menyaring gangguan bulan, dan menganalisis tutupan awan dari data instrumen Sky Quality Meter (SQM).
""")

# =====================================================================
# SEMUA FUNGSI MATEMATIKA & ASTRONOMI
# =====================================================================
def read_header_and_find_data_start(path, max_header_lines=80):
    header = list()
    data_start = None
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
                lat_str, lon_str, *etc = nums
                lat, lon = float(lat_str), float(lon_str)
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

def bin_alt(x, y, bin_deg=0.25):
    bins = np.floor((x + 90.0) / bin_deg) * bin_deg - 90.0
    dfb = pd.DataFrame({"bin": bins, "x": x, "y": y})
    g = dfb.groupby("bin", sort=True).agg(x=("x", "median"), y=("y", "median")).reset_index(drop=True)
    return g["x"].values, g["y"].values

def mad_sigma(arr):
    arr = np.asarray(arr)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad

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
        moon_alt_deg = np.rad2deg(moon.alt)
        moon_phase = moon.phase / 100.0  
        if moon_alt_deg > 0 and moon_phase > 0.05:
            is_corrected = True
            I_total = 10 ** (-0.4 * row["mpsas"])
            I_moon_estimate = (moon_phase * np.sin(moon.alt)) * (10 ** (-0.4 * 21.5))
            I_sky_real = max(I_total - I_moon_estimate, 10 ** (-0.4 * 22.0))
            corrected_mpsas.append(-2.5 * np.log10(I_sky_real))
        else:
            corrected_mpsas.append(row["mpsas"])
    am["mpsas_corrected"] = corrected_mpsas
    return am, is_corrected

def analyze_cloud_cover(am, onset_alt, window_minutes=60):
    if onset_alt is None: return 0.0, pd.DataFrame()
    onset_idx = (np.abs(am["sun_alt"] - onset_alt)).argmin()
    onset_dt = am["local_dt"].iloc[onset_idx]
    mask_window = (am["local_dt"] >= onset_dt - pd.Timedelta(minutes=window_minutes)) & \
                  (am["local_dt"] <= onset_dt + pd.Timedelta(minutes=window_minutes))
    df_win = am[mask_window].copy()
    if len(df_win) < 21: return 0.0, df_win
    df_win['rolling_std'] = df_win['mpsas_corrected'].rolling(21, center=True).std()
    mean_mpsas = df_win['mpsas_corrected'].mean()
    dynamic_threshold = max((-0.04545 * mean_mpsas) + 1.0500, 0.05)
    df_win['is_cloudy'] = df_win['rolling_std'] > dynamic_threshold
    cloud_pct = df_win['is_cloudy'].mean() * 100
    return cloud_pct, df_win

def categorize_light_pollution(baseline_mpsas):
    if baseline_mpsas >= 21.3: category = "Gelap / Dark (Tipe 1)"
    elif 20.2 <= baseline_mpsas < 21.3: category = "Agak Gelap / Slightly Dark (Tipe 2)"
    elif 19.1 <= baseline_mpsas < 20.2: category = "Agak Terang / Slightly Bright (Tipe 3)"
    else: category = "Terang / Urban Skyglow (Tipe 4)"
    expected_alt = (baseline_mpsas - 11.12) / 0.55
    return category, expected_alt

# =====================================================================
# UI & PROSES UNGGAH FILE STREAMLIT
# =====================================================================

with st.sidebar:
    st.header("⚙️ Pengaturan")
    method = st.selectbox("Metode Analisis", ["SIGMAG-STAB", "SIGMOID"])
    st.info("Pilih file mentah SQM (.dat) atau file .zip yang berisi banyak file .dat sekaligus.")

uploaded_files = st.file_uploader("Unggah File Data", accept_multiple_files=True, type=['dat', 'txt', 'zip'])

if uploaded_files:
    if st.button("Jalankan Analisis 🚀"):
        # Membuat folder sementara untuk mengekstrak/menyimpan file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = list()
            
            # Memproses setiap file yang diunggah
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
                st.success(f"✅ Ditemukan {len(file_paths)} file untuk diproses. Memulai kalkulasi...")
                
                results_list = list()
                
                for path in file_paths:
                    try:
                        # --- PROSES INTI ---
                        am, site, lat, lon, utc_offset, date_str = load_sqm_data(path)
                        if am.empty: continue
                        
                        am, is_corrected = apply_moonlight_correction(am, lat, lon, utc_offset)
                        am["mpsas_smooth_temp"] = savgol_filter(am["mpsas_corrected"], window_length=31, polyorder=2)
                        
                        xb, yb = bin_alt(am["sun_alt"].values, am["mpsas_smooth_temp"].values, bin_deg=0.25)
                        grad_binned = np.gradient(yb, xb)
                        grad_med = pd.Series(grad_binned).rolling(5, center=True, min_periods=1).median().values
                        grad_mean = pd.Series(grad_med).rolling(9, center=True, min_periods=1).mean().values
                        
                        base_mask = xb < -20.0
                        g_base = grad_mean[base_mask]
                        mu_grad = float(np.mean(g_base))
                        sigma_used = max(float(mad_sigma(g_base)), 0.01) 
                        
                        if sigma_used < 0.02: k_factor = 1.0
                        elif sigma_used < 0.05: k_factor = 1.2
                        else: k_factor = 1.5
                        
                        threshold_grad = mu_grad - k_factor * sigma_used
                        search_mask = xb >= -20.0
                        xs = xb[search_mask]
                        gs = grad_mean[search_mask]
                        below = gs < threshold_grad
                        
                        onset_idx = None
                        consec_count = 0
                        for i, flag in enumerate(below):
                            if flag:
                                consec_count += 1
                                if consec_count >= 5:
                                    onset_idx = i - 5 + 1
                                    break
                            else:
                                consec_count = 0
                                
                        onset_alt, onset_msas = None, None
                        if onset_idx is not None and len(xs) > 0:
                            onset_alt = float(xs[onset_idx])
                            onset_msas = float(np.interp(onset_alt, am["sun_alt"], am["mpsas_corrected"]))
                            
                        cloud_pct, df_win = analyze_cloud_cover(am, onset_alt, window_minutes=60)
                        
                        base_s = am[am["sun_alt"] < -20]["mpsas_corrected"]
                        baseline_mpsas = base_s.median() if not base_s.empty else am["mpsas_corrected"].max()
                        lp_category, expected_alt = categorize_light_pollution(baseline_mpsas)
                        
                        results_list.append({
                            "Tanggal": date_str, "Lokasi": site,
                            "Bortle/ALAN": lp_category.split("(")[-1].replace(")",""),
                            "Awan_%": round(cloud_pct, 1), "Bulan": "Aktif" if is_corrected else "-",
                            "Fajar_Alt": round(onset_alt, 2) if onset_alt is not None else None,
                            "Fajar_MSAS": round(onset_msas, 2) if onset_msas is not None else None
                        })

                        # --- PLOT KE WEB ---
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
                        
                        onset_str = f"{onset_alt:.2f}°" if onset_alt is not None else "Gagal Terdeteksi"
                        info_text = (
                            f"Kategori Polusi : {lp_category}\n"
                            f"Garis Dasar     : {baseline_mpsas:.2f} Mpsas\n"
                            f"Koreksi Bulan   : {'Aktif' if is_corrected else 'Tidak Ada'}\n"
                            f"Tutupan Awan    : {cloud_pct:.1f}%\n"
                            f"Titik Aktual    : {onset_str}"
                        )
                        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
                        y_pos_dinamis = baseline_mpsas - 1.2 
                        ax.text(0.02, y_pos_dinamis, info_text, transform=ax.get_yaxis_transform(), fontsize=9,
                                verticalalignment='bottom', bbox=props, family='monospace')
                        
                        ax.legend(loc="upper right")
                        
                        # Tampilkan Grafik di Streamlit
                        st.pyplot(fig)
                        plt.close(fig)

                    except Exception as e:
                        st.error(f"Gagal memproses {os.path.basename(path)}: {str(e)}")
                
                # --- TABEL REKAPITULASI ---
                if results_list:
                    st.success("🎉 Analisis Selesai! Berikut adalah data rekapitulasinya:")
                    df_rekap = pd.DataFrame(results_list)
                    st.dataframe(df_rekap)
                    
                    # Tombol Download CSV
                    csv = df_rekap.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Unduh Rekap CSV",
                        data=csv,
                        file_name=f'Rekap_SQM_{method}.csv',
                        mime='text/csv',
                    )
