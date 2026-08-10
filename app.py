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

import cloudinary
import cloudinary.uploader
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# =====================================================================
# KONFIGURASI TEMA, LOGO, & LINK PERMANEN
# =====================================================================
st.set_page_config(page_title="Kawakib SQM Analyzer", page_icon="🌌", layout="wide")

KAWAKIB_LOGO_URL = "https://lh3.googleusercontent.com/d/1aoTDRdL-wS8EPytGGZ7dsJY3Nntnp-3U"
GSHEETS_PERMANEN_URL = "https://docs.google.com/spreadsheets/d/1E4RpTfcPeQorW3r9cjpZ5cp31dpa7N_oXRZksRWdxG4/edit?gid=0#gid=0"
SAMPLE_DATA_DRIVE_URL = "https://drive.google.com/drive/folders/1KHg8dRtkt9KrdDFZ8esbiuHQtKJvP2AN?usp=drive_link"

# --- INJEKSI CSS BERSIH & FIX DROPDOWN + IKON SIDEBAR ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,600;1,400&family=Inter:wght@300;400;600&display=swap');
    .block-container { padding-top: 2.9rem !important; padding-bottom: 2rem !important; }
    [data-testid="stSidebar"] { background-color: #1A3C40; padding-top: 1.5rem !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] li { color: #E8F1F2 !important; font-family: 'Inter', sans-serif !important; }
    [data-testid="stSidebar"] span:not([data-baseweb="select"] span) { color: #E8F1F2 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] { background-color: #FFFFFF !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #1A3C40 !important; -webkit-text-fill-color: #1A3C40 !important; }
    .stButton>button { background-color: #1D9A9C; color: #FFFFFF !important; border-radius: 50px !important; border: none; padding: 0.5rem 1.5rem !important; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #25B8BA; }
    [data-testid="stSidebar"] a { color: #79E0E2 !important; font-weight: 600; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# FUNGSI BACKEND (CLOUD & MATEMATIKA)
# =====================================================================
try:
    cloudinary.config(cloud_name=st.secrets["cloudinary"]["cloud_name"], api_key=st.secrets["cloudinary"]["api_key"], api_secret=st.secrets["cloudinary"]["api_secret"], secure=True)
except: pass

def upload_plot_to_cloudinary(fig, filename):
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=100)
            tmp_path = tmp.name
        response = cloudinary.uploader.upload(tmp_path, folder="kawakib_arsip", public_id=filename.replace(".png", ""))
        os.remove(tmp_path)
        return response.get("secure_url")
    except: return ""

def upload_raw_to_cloudinary(file_path, filename):
    try:
        response = cloudinary.uploader.upload(file_path, resource_type="raw", folder="kawakib_raw_data", public_id=filename)
        return response.get("secure_url")
    except: return ""

def get_gsheets_client():
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        return gspread.authorize(creds)
    except: return None

def save_to_google_sheets(data_dict):
    client = get_gsheets_client()
    if client is None: return False
    try:
        sheet = client.open_by_url(GSHEETS_PERMANEN_URL).sheet1
        existing_data = sheet.get_all_values()
        if not existing_data:
            headers = list(data_dict.keys())
            sheet.append_row(headers)
            sheet.append_row([str(data_dict.get(key, "")) for key in headers])
            return True
            
        headers = existing_data[0]
        
        new_headers_added = False
        for key in data_dict.keys():
            if key not in headers:
                headers.append(key)
                new_headers_added = True
                
        if new_headers_added:
            sheet.update(range_name='A1', values=[headers])
            
        try:
            idx_tgl = headers.index("Tanggal")
            idx_lok = headers.index("Lokasi")
            idx_met = headers.index("Metode")
        except:
            sheet.append_row([str(data_dict.get(key, "")) for key in headers])
            return True
            
        row_to_update = None
        for i, row in enumerate(existing_data[1:], start=2):
            if len(row) > max(idx_tgl, idx_lok, idx_met):
                if (row[idx_tgl] == str(data_dict["Tanggal"]) and row[idx_lok] == str(data_dict["Lokasi"]) and row[idx_met] == str(data_dict["Metode"])):
                    row_to_update = i; break
                    
        values = [str(data_dict.get(h, "")) for h in headers]
        if row_to_update:
            cell_range = f'A{row_to_update}:{chr(65 + len(headers) - 1)}{row_to_update}' if len(headers) <= 26 else f'A{row_to_update}'
            cell_list = sheet.range(cell_range)
            for cell, val in zip(cell_list, values): cell.value = val
            sheet.update_cells(cell_list)
        else: 
            sheet.append_row(values)
        return True
    except Exception as e:
        print(f"Error saving to sheets: {e}")
        return False

def load_data_from_google_sheets():
    client = get_gsheets_client()
    if client is None: return pd.DataFrame()
    try:
        sheet = client.open_by_url(GSHEETS_PERMANEN_URL).sheet1
        return pd.DataFrame(sheet.get_all_records())
    except: return pd.DataFrame()

# =====================================================================
# FUNGSI INTEGRASI DRIVE (THE HARVESTER - DIRECT FILE SCAN)
# =====================================================================
def get_drive_service():
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def sync_from_soof_drive():
    service = get_drive_service()
    downloaded_paths = []
    try:
        results = service.files().list(
            q="mimeType != 'application/vnd.google-apps.folder' and (name contains '.dat' or name contains '.DAT' or name contains '.txt') and trashed = false",
            fields="files(id, name)"
        ).execute()
        files = results.get('files', [])
        
        for file in files:
            file_path = os.path.join(tempfile.gettempdir(), file['name'])
            request = service.files().get_media(fileId=file['id'])
            with open(file_path, "wb") as f:
                f.write(request.execute())
            downloaded_paths.append(file_path)
    except Exception as e:
        st.error(f"Error accessing Google Drive: {e}")
        
    return downloaded_paths

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
                data_start = i; break
    for line in header:
        if "Location name:" in line: site = line.split("Location name:")[-1].strip()
        if "Position:" in line or "Position" in line:
            nums = re.findall(r"[+-]?\d+\.?\d*", line)
            if len(nums) >= 2: lat, lon = float(nums[0]), float(nums[1])
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
    df = pd.read_csv(file_path, skiprows=data_start, sep=";", header=None, names=["utc","local","temp","cnt","hz","mpsas"], engine="python", on_bad_lines="skip")
    df["local_dt"] = pd.to_datetime(df["local"], errors="coerce")
    df = df.dropna(subset=["local_dt","mpsas"])
    df["sun_alt"] = solar_alt(df["local_dt"], lat, lon, utc_offset)
    am = df[(df["local_dt"].dt.hour < 12) & (df["mpsas"] > 0)].copy()
    am = am.sort_values("sun_alt").reset_index(drop=True)
    date_str = am["local_dt"].iloc[0].strftime("%Y-%m-%d") if not am.empty else "Unknown"
    return am, site, lat, lon, utc_offset, date_str

def apply_moonlight_correction(am, lat, lon, utc_offset):
    obs, moon = ephem.Observer(), ephem.Moon()
    obs.lat, obs.lon = str(lat), str(lon)
    corrected_mpsas, is_corrected = list(), False
    for _, row in am.iterrows():
        obs.date = (row["local_dt"] - pd.Timedelta(hours=utc_offset)).strftime('%Y/%m/%d %H:%M:%S')
        moon.compute(obs)
        if np.rad2deg(moon.alt) > 0 and (moon.phase / 100.0) > 0.05:
            is_corrected = True
            I_tot = 10 ** (-0.4 * row["mpsas"])
            I_moon = ((moon.phase / 100.0) * np.sin(moon.alt)) * (10 ** (-0.4 * 21.5))
            corrected_mpsas.append(-2.5 * np.log10(max(I_tot - I_moon, 10 ** (-0.4 * 22.0))))
        else: corrected_mpsas.append(row["mpsas"])
    am["mpsas_corrected"] = corrected_mpsas
    return am, is_corrected

def analyze_cloud_cover(am, onset_alt, window_minutes=60):
    if onset_alt is None: return 0.0, pd.DataFrame()
    onset_idx = (np.abs(am["sun_alt"] - onset_alt)).argmin()
    onset_dt = am["local_dt"].iloc[onset_idx]
    mask = (am["local_dt"] >= onset_dt - pd.Timedelta(minutes=window_minutes)) & (am["local_dt"] <= onset_dt + pd.Timedelta(minutes=window_minutes))
    df_win = am[mask].copy()
    if len(df_win) < 21: return 0.0, df_win
    df_win['rolling_std'] = df_win['mpsas_corrected'].rolling(21, center=True).std()
    dyn_thresh = max((-0.04545 * df_win['mpsas_corrected'].mean()) + 1.0500, 0.05)
    base_series = am[am["sun_alt"] < -20]["mpsas_corrected"]
    garis_dasar = base_series.median() if not base_series.empty else am["mpsas_corrected"].max()
    df_win['is_cloudy'] = (df_win['rolling_std'] > dyn_thresh) | ((np.abs(df_win['mpsas_corrected'] - garis_dasar) > 0.2) & (df_win['sun_alt'] < onset_alt))
    return (df_win['is_cloudy'].mean() * 100), df_win

def categorize_light_pollution(baseline_mpsas):
    if baseline_mpsas >= 21.3: return "Gelap (Tipe 1)", (baseline_mpsas - 11.12) / 0.55
    elif 20.2 <= baseline_mpsas < 21.3: return "Agak Gelap (Tipe 2)", (baseline_mpsas - 11.12) / 0.55
    elif 19.1 <= baseline_mpsas < 20.2: return "Agak Terang (Tipe 3)", (baseline_mpsas - 11.12) / 0.55
    else: return "Terang / Urban (Tipe 4)", (baseline_mpsas - 11.12) / 0.55

def get_dynamic_params(am):
    if len(am) < 2: return 0.25, 5
    delta_s = am["local_dt"].diff().median().total_seconds()
    return (0.25, 5) if delta_s <= 65 else ((0.5, 4) if delta_s <= 125 else ((0.75, 3) if delta_s <= 305 else (1.0, 3)))

def bin_alt(x, y, bin_deg):
    dfb = pd.DataFrame({"bin": np.floor((x + 90.0) / bin_deg) * bin_deg - 90.0, "x": x, "y": y})
    g = dfb.groupby("bin", sort=True).agg(x=("x", "median"), y=("y", "median")).reset_index(drop=True)
    return g["x"].values, g["y"].values

def mad_sigma(arr):
    arr = np.asarray(arr)
    return 1.4826 * np.median(np.abs(arr - np.median(arr)))

def analyze_sigmag(am, bin_deg, n_consec):
    am["mpsas_smooth"] = savgol_filter(am["mpsas_corrected"], window_length=min(31, len(am)|1), polyorder=2)
    xb, yb = bin_alt(am["sun_alt"].values, am["mpsas_smooth"].values, bin_deg)
    grad_mean = pd.Series(pd.Series(np.gradient(yb, xb)).rolling(5, center=True, min_periods=1).median().values).rolling(9, center=True, min_periods=1).mean().values
    g_base = grad_mean[xb < -20.0]
    if len(g_base) < 5: return None, None
    mu_grad, sigma_used = float(np.mean(g_base)), max(float(mad_sigma(g_base)), 0.01)
    k_factor = 1.0 if sigma_used < 0.02 else (1.2 if sigma_used < 0.05 else 1.5)
    search_mask = xb >= -20.0
    xs, gs = xb[search_mask], grad_mean[search_mask]
    onset_idx, consec_count = None, 0
    for i, flag in enumerate(gs < (mu_grad - k_factor * sigma_used)):
        if flag:
            consec_count += 1
            if consec_count >= n_consec: onset_idx = i - n_consec + 1; break
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
        popt, _ = curve_fit(sigmoid, x_data, y_data, p0=[y_data.min() - y_data.max(), -15.0, 1.0, y_data.max()], maxfev=5000)
        x_eval = np.linspace(-30, -5, 500)
        y_eval = sigmoid(x_eval, *popt)
        onset_idx = np.argmax(y_eval < (popt[3] - 0.15))
        if onset_idx == 0: return None, None
        return x_eval[onset_idx], float(np.interp(x_eval[onset_idx], am["sun_alt"], am["mpsas_corrected"]))
    except: return None, None

def process_and_save_data(file_paths, method, df_existing):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for idx, path in enumerate(file_paths):
        status_text.text(f"Memproses {idx+1}/{len(file_paths)}: {os.path.basename(path)}")
        try:
            am, site, lat, lon, utc_offset, date_str = load_sqm_data(path)
            if am.empty: continue
            am, is_corrected = apply_moonlight_correction(am, lat, lon, utc_offset)
            bin_deg, n_consec = get_dynamic_params(am)
            if method == "SIGMAG-STAB": onset_alt, onset_msas = analyze_sigmag(am, bin_deg, n_consec)
            else: onset_alt, onset_msas = analyze_sigmoid(am)
            cloud_pct, df_win = analyze_cloud_cover(am, onset_alt)
            base_series = am[am["sun_alt"] < -20]["mpsas_corrected"]
            baseline_mpsas = base_series.median() if not base_series.empty else am["mpsas_corrected"].max()
            lp_category, expected_alt = categorize_light_pollution(baseline_mpsas)
            
            kota = re.split(r'[-,\|]', site)[-1].strip()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(am["sun_alt"], am["mpsas_corrected"], color="#1A3C40", alpha=0.8, linewidth=1.5, label="SQM Terkoreksi")
            if is_corrected: ax.plot(am["sun_alt"], am["mpsas"], color="#808080", alpha=0.4, linestyle=":", label="SQM Mentah")
            if onset_alt is not None:
                ax.axvline(onset_alt, color="#1D9A9C", linestyle="--", linewidth=2, label=f"Titik Belok ({onset_alt:.2f}°)")
                ax.scatter([onset_alt], [onset_msas], color="#1D9A9C", s=60, zorder=5)
            cloudy_points = df_win[df_win['is_cloudy'] == True] if 'is_cloudy' in df_win.columns else pd.DataFrame()
            if not cloudy_points.empty: ax.scatter(cloudy_points["sun_alt"], cloudy_points["mpsas_corrected"], color="#d9534f", s=15, label="Indikasi Awan", zorder=4)
            ax.invert_yaxis()
            ax.set_xlim(-30, -5)
            ax.set_xlabel("Ketinggian Matahari (Derajat)", fontweight='bold')
            ax.set_ylabel("Kecerlangan Langit (Mpsas)", fontweight='bold')
            ax.set_title(f"{site} | {date_str} [{method}]", color="#1A3C40", fontweight='bold')
            ax.grid(True, linestyle=":", alpha=0.6)
            onset_str = f"{onset_alt:.2f}°" if onset_alt is not None else "Tidak Ditemukan"
            info_text = (f"Garis Dasar : {baseline_mpsas:.2f} Mpsas\n"
                         f"Awan / Bulan : {cloud_pct:.1f}% / {'Aktif' if is_corrected else 'Pasif'}\n"
                         f"Fajar Sadiq : {onset_str}")
            props = dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9, edgecolor='#1A3C40')
            ax.text(0.02, baseline_mpsas - 1.2, info_text, transform=ax.get_yaxis_transform(), fontsize=9, verticalalignment='bottom', bbox=props, family='monospace')
            ax.legend(loc="upper right")
            
            plot_url, raw_url = "", ""
            if not df_existing.empty and {"Tanggal", "Lokasi", "Metode"}.issubset(df_existing.columns):
                match = df_existing[(df_existing["Tanggal"].astype(str).str.strip() == str(date_str)) & (df_existing["Lokasi"].astype(str).str.strip() == str(site)) & (df_existing["Metode"].astype(str).str.strip() == str(method))]
                if not match.empty:
                    plot_url = str(match.iloc[0].get("Link_Grafik", "")).strip()
                    raw_url = str(match.iloc[0].get("Link_DataMentah", "")).strip()

            if plot_url and raw_url: 
                status_text.text(f"Data duplikat terdeteksi: Menggunakan arsip lama...")
            else:
                status_text.text(f"Mengunggah arsip baru ke Cloudinary...")
                plot_url = upload_plot_to_cloudinary(fig, f"Plot_{site}_{date_str}_{method}".replace(" ", "_"))
                raw_url = upload_raw_to_cloudinary(path, f"Raw_{site}_{date_str}_{method}.dat".replace(" ", "_"))
            
            save_to_google_sheets({
                "Tanggal": date_str, "Kota": kota, "Lokasi": site, 
                "Lintang": lat, "Bujur": lon, 
                "Metode": method, "Bortle": lp_category.split("(")[-1].replace(")",""),
                "Awan_%": round(cloud_pct, 1), "Koreksi_Bulan": "Aktif" if is_corrected else "Pasif",
                "Garis_Dasar": round(baseline_mpsas, 2), "Fajar_Alt": round(onset_alt, 2) if onset_alt is not None else "",
                "Fajar_MSAS": round(onset_msas, 2) if onset_msas is not None else "",
                "Link_Grafik": plot_url, "Link_DataMentah": raw_url 
            })
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e: st.error(f"Gagal memproses {os.path.basename(path)}: {str(e)}")
        progress_bar.progress((idx + 1) / len(file_paths))
    status_text.text("")
    st.success("🎉 Seluruh pengamatan berhasil diproses.")

# =====================================================================
# UI KONTROL & SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    method = st.selectbox("Metode Ekstraksi Fajar", ["SIGMAG-STAB", "SIGMOID"])
    st.info("Upload mandiri atau tarik data otomatis dari SOOF Drive.")
    st.divider()
    if st.button("📡 Tarik Data dari SOOF (Drive)"):
        with st.spinner("Menyedot data dari kotak pos..."):
            file_paths = sync_from_soof_drive()
            if not file_paths: 
                st.warning("Tidak ada file baru di Drive atau pengaturan folder belum lengkap.")
            else: 
                st.success(f"Berhasil menarik {len(file_paths)} file!")
                df_existing = load_data_from_google_sheets()
                process_and_save_data(file_paths, method, df_existing)
    
    st.markdown("### 📂 Data Pembelajaran")
    st.markdown(f"[🔗 Unduh Sample Data SQM]({SAMPLE_DATA_DRIVE_URL})")

# =====================================================================
# HEADER UTAMA
# =====================================================================
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 4px;">
        <img src="{KAWAKIB_LOGO_URL}" style="height: 48px; width: auto; object-fit: contain;">
        <h1 style='font-family: Lora, serif; color: #1A3C40; font-size: 1.65rem; margin: 0; padding: 0; line-height: 1.2;'>
            KAWAKIB INSTITUTE: SQM Fajar Analyzer
        </h1>
    </div>
    <div style="border-bottom: 2px solid #1D9A9C; margin-top: 8px; margin-bottom: 10px;"></div>
    <p style='color: #555; font-size: 0.92rem; margin-bottom: 15px;'>
        Aplikasi web ini mengekstrak titik belok fajar sadiq secara otonom. Terintegrasi dengan sistem penyimpanan cloud untuk analisis data yang persisten.
    </p>
""", unsafe_allow_html=True)

tab_analisis, tab_histori, tab_dashboard, tab_algoritma = st.tabs(["🚀 Analisis Data", "☁️ Basis Data Cloud", "📊 Dashboard Statistik", "📖 Metodologi & Algoritma"])

with tab_analisis:
    uploaded_files = st.file_uploader("Unggah File Data Observasi Secara Manual", accept_multiple_files=True, type=['dat', 'DAT', 'txt', 'TXT', 'zip', 'ZIP'])
    if uploaded_files:
        if st.button("Mulai Kalkulasi Fotometri 🚀"):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = list()
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    if uploaded_file.name.endswith(('.zip', '.ZIP')):
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                            for root, dirs, files_in_dir in os.walk(temp_dir):
                                for file in files_in_dir:
                                    if file.endswith(('.dat', '.txt', '.DAT', '.TXT')): file_paths.append(os.path.join(root, file))
                    else: file_paths.append(file_path)

                if not file_paths: st.error("❌ Tidak ada file .dat yang valid ditemukan.")
                else:
                    df_existing = load_data_from_google_sheets()
                    process_and_save_data(file_paths, method, df_existing)

with tab_histori:
    st.header("☁️ Basis Data Fotometri Terpusat")
    if st.button("🔄 Sinkronisasi"): st.rerun()
    df_cloud = load_data_from_google_sheets()
    if df_cloud.empty: st.info("Basis data kosong.")
    else:
        # Menampilkan seluruh kolom termasuk Link Grafik & Data Mentah secara utuh
        st.dataframe(df_cloud, use_container_width=True)
        st.download_button("⬇️ Unduh CSV", df_cloud.to_csv(index=False).encode('utf-8'), 'Rekap_Kawakib_Cloud.csv', 'text/csv')

with tab_dashboard:
    st.header("📊 Ringkasan Statistik Data")
    df_stat = load_data_from_google_sheets()
    
    if not df_stat.empty:
        df_stat['Fajar_Alt'] = pd.to_numeric(df_stat['Fajar_Alt'], errors='coerce')
        df_stat['Awan_%'] = pd.to_numeric(df_stat['Awan_%'], errors='coerce')
        
        if 'Kota' not in df_stat.columns and 'Lokasi' in df_stat.columns:
            df_stat['Kota'] = df_stat['Lokasi'].apply(lambda x: re.split(r'[-,\|]', str(x))[-1].strip())
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data", len(df_stat))
        c2.metric("Rata-rata Fajar Global", f"{df_stat['Fajar_Alt'].mean():.2f}°")
        c3.metric("Rata-rata Awan", f"{df_stat['Awan_%'].mean():.1f}%")
        
        st.divider()
        st.subheader("🌍 Analisis Spasial & Geografis (Berdasarkan Koordinat Kota)")
        
        if 'Lintang' in df_stat.columns and 'Bujur' in df_stat.columns:
            df_map = df_stat.dropna(subset=['Lintang', 'Bujur', 'Fajar_Alt']).copy()
            if not df_map.empty:
                df_map['lat'] = pd.to_numeric(df_map['Lintang'], errors='coerce')
                df_map['lon'] = pd.to_numeric(df_map['Bujur'], errors='coerce')
                
                df_map_agg = df_map.groupby(['Kota', 'lat', 'lon']).agg(
                    Fajar_Rata2=('Fajar_Alt', 'mean'),
                    Total_Observasi=('Fajar_Alt', 'count')
                ).reset_index()
                
                st.markdown("**Peta Persebaran Titik Observasi Kawakib**")
                st.map(df_map_agg[['lat', 'lon']], zoom=5, use_container_width=True)
                
                st.markdown("**Tabel Agregasi Data Spasial**")
                df_tabel = df_map_agg.rename(columns={'lat': 'Lintang', 'lon': 'Bujur', 'Fajar_Rata2': 'Rata-rata Fajar (°)'})
                st.dataframe(df_tabel.style.format({'Rata-rata Fajar (°)': '{:.2f}'}), use_container_width=True)
        
        st.markdown("**Perbandingan Rata-rata Kedalaman Fajar per Kota**")
        df_loc = df_stat.groupby('Kota')['Fajar_Alt'].mean().dropna()
        if not df_loc.empty:
            st.bar_chart(df_loc)
        else:
            st.info("Data fajar yang valid belum tersedia untuk memuat grafik.")

        st.divider()
        st.subheader("Analisis Per Kriteria Lingkungan")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Berdasarkan Polusi Cahaya (Bortle)**")
            st.bar_chart(df_stat.groupby('Bortle')['Fajar_Alt'].mean().dropna())
        with col2:
            st.markdown("**Berdasarkan Koreksi Bulan**")
            st.bar_chart(df_stat.groupby('Koreksi_Bulan')['Fajar_Alt'].mean().dropna())
    else: st.info("Belum ada data untuk dianalisis.")

with tab_algoritma:
    st.header("📖 Metodologi & Landasan Matematis")
    st.markdown("Aplikasi Kawakib Analyzer menerapkan pipa pemrosesan data otonom dengan metode statistik astrofisika tingkat lanjut.")
    with st.expander("1. Pra-Pemrosesan Fotometri", expanded=True):
        st.markdown("* **Kalkulasi Ketinggian Matahari:** Presisi astronomis dengan Julian Date.\n* **Smoothing:** Filter *Savitzky-Golay* (Orde 2, Jendela 31) untuk membuang noise tanpa merusak puncak kurva.\n* **Koreksi Cahaya Bulan:** Kompensasi otomatis jika fase cahaya bulan > 5%.")
    with st.expander("2. Metode SIGMAG-STAB"):
        st.markdown("Deteksi fajar berdasarkan turunan pertama kurva gradien dengan ambang batas dinamis berbasis **MAD (Median Absolute Deviation)**.")
        st.latex(r"T = \mu - (k \cdot \sigma)")
    with st.expander("3. Metode SIGMOID"):
        st.markdown("Fitting data ke fungsi logistik (Kurva-S) menggunakan teknik *Non-Linear Least Squares*.")
        st.latex(r"y = \frac{L}{1 + e^{-k(x - x_0)}} + b")
    with st.expander("4. Diagnostik Awan & Analisis Spasial"):
        st.markdown("* **Deteksi Awan:** Deteksi anomali pada kecerlangan langit menggunakan *Rolling Standard Deviation* (60 menit jendela waktu).\n* **Analisis Spasial:** Ekstraksi otomatis koordinat Lintang/Bujur untuk memetakan korelasi elevasi dan geografis terhadap kedalaman sudut fajar antar kota.")
