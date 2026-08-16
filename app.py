import streamlit as st
import sys
import os
import re
import zipfile
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import time
import urllib.request
import json

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
# FUNGSI BACKEND (CLOUD, API & REVERSE GEOCODING)
# =====================================================================
try:
    cloudinary.config(
        cloud_name=st.secrets["cloudinary"]["cloud_name"].strip() if "cloudinary" in st.secrets else None,
        api_key=st.secrets["cloudinary"]["api_key"].strip() if "cloudinary" in st.secrets else None,
        api_secret=st.secrets["cloudinary"]["api_secret"].strip() if "cloudinary" in st.secrets else None,
        secure=True
    )
except: pass

def get_city_from_coords(lat, lon):
    if lat is None or lon is None: return "Unknown"
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Kawakib-SQM-Analyzer/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            address = data.get('address', {})
            city = address.get('city') or address.get('county') or address.get('state_district') or address.get('town') or address.get('village')
            if city:
                return city.replace("Kabupaten ", "").replace("Kota ", "")
    except: pass
    return "Unknown"

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

def sync_from_soof_drive():
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    service = build('drive', 'v3', credentials=creds)
    downloaded_paths = []
    try:
        page_token = None
        while True:
            results = service.files().list(
                q="mimeType != 'application/vnd.google-apps.folder' and (name contains '.dat' or name contains '.DAT' or name contains '.txt') and trashed = false",
                fields="nextPageToken, files(id, name)", pageSize=1000, pageToken=page_token
            ).execute()
            files = results.get('files', [])
            for file in files:
                file_path = os.path.join(tempfile.gettempdir(), file['name'])
                request = service.files().get_media(fileId=file['id'])
                with open(file_path, "wb") as f: f.write(request.execute())
                downloaded_paths.append(file_path)
            page_token = results.get('nextPageToken', None)
            if page_token is None: break
    except: pass
    return downloaded_paths

def get_satellite_cloud_cover(lat, lon, date_str):
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=cloudcover&timezone=auto"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            clouds = data.get('hourly', {}).get('cloudcover', [])
            valid = [c for c in clouds[3:6] if c is not None]
            if valid: return round(sum(valid)/len(valid), 1)
    except: pass
    try:
        url2 = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=cloudcover&timezone=auto"
        req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req2, timeout=5) as response:
            data = json.loads(response.read())
            clouds = data.get('hourly', {}).get('cloudcover', [])
            valid = [c for c in clouds[3:6] if c is not None]
            if valid: return round(sum(valid)/len(valid), 1)
    except: pass
    return None

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

def analyze_cloud_cover(am, onset_alt, window_minutes=15):
    if onset_alt is None: return 0.0, pd.DataFrame()
    onset_idx = (np.abs(am["sun_alt"] - onset_alt)).argmin()
    onset_dt = am["local_dt"].iloc[onset_idx]
    
    mask = (am["local_dt"] >= onset_dt - pd.Timedelta(minutes=window_minutes)) & (am["local_dt"] <= onset_dt + pd.Timedelta(minutes=window_minutes))
    df_win = am[mask].copy()
    
    r_win = min(11, len(df_win) if len(df_win) % 2 != 0 else len(df_win)-1)
    if r_win < 3: return 0.0, df_win
    
    df_win['rolling_std'] = df_win['mpsas_corrected'].rolling(r_win, center=True).std()
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
            
            cloud_pct, df_win = analyze_cloud_cover(am, onset_alt, window_minutes=15)
            sat_cloud_pct = get_satellite_cloud_cover(lat, lon, date_str) 
            
            base_series = am[am["sun_alt"] < -20]["mpsas_corrected"]
            baseline_mpsas = base_series.median() if not base_series.empty else am["mpsas_corrected"].max()
            lp_category, expected_alt = categorize_light_pollution(baseline_mpsas)
            
            kota_akurat = get_city_from_coords(lat, lon)
            if kota_akurat == "Unknown":
                kota = re.split(r'[-,\|]', site)[-1].strip()
            else:
                kota = kota_akurat
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(am["sun_alt"], am["mpsas_corrected"], color="#1A3C40", alpha=0.8, linewidth=1.5, label="SQM Terkoreksi")
            if is_corrected: ax.plot(am["sun_alt"], am["mpsas"], color="#808080", alpha=0.4, linestyle=":", label="SQM Mentah")
            if onset_alt is not None:
                ax.axvline(onset_alt, color="#1D9A9C", linestyle="--", linewidth=2, label=f"Titik Belok ({onset_alt:.2f}°)")
                ax.scatter([onset_alt], [onset_msas], color="#1D9A9C", s=60, zorder=5)
            cloudy_points = df_win[df_win['is_cloudy'] == True] if 'is_cloudy' in df_win.columns else pd.DataFrame()
            if not cloudy_points.empty: ax.scatter(cloudy_points["sun_alt"], cloudy_points["mpsas_corrected"], color="#d9534f", s=15, label="Indikasi Awan (SQM)", zorder=4)
            ax.invert_yaxis()
            ax.set_xlim(-30, -5)
            ax.set_xlabel("Ketinggian Matahari (Derajat)", fontweight='bold')
            ax.set_ylabel("Kecerlangan Langit (Mpsas)", fontweight='bold')
            ax.set_title(f"{site} | {date_str} [{method}]", color="#1A3C40", fontweight='bold')
            ax.grid(True, linestyle=":", alpha=0.6)
            
            onset_str = f"{onset_alt:.2f}°" if onset_alt is not None else "Tidak Ditemukan"
            sat_str = f"{sat_cloud_pct:.1f}%" if sat_cloud_pct is not None else "N/A"
            
            info_text = (f"Garis Dasar : {baseline_mpsas:.2f} Mpsas\n"
                         f"Awan SQM/Sat: {cloud_pct:.1f}% / {sat_str}\n"
                         f"Fase Bulan  : {'Aktif' if is_corrected else 'Pasif'}\n"
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
                "Awan_%": round(cloud_pct, 1), 
                "Awan_Satelit_%": sat_cloud_pct if sat_cloud_pct is not None else "",
                "Koreksi_Bulan": "Aktif" if is_corrected else "Pasif",
                "Garis_Dasar": round(baseline_mpsas, 2), "Fajar_Alt": round(onset_alt, 2) if onset_alt is not None else "",
                "Fajar_MSAS": round(onset_msas, 2) if onset_msas is not None else "",
                "Link_Grafik": plot_url, "Link_DataMentah": raw_url 
            })
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e: st.error(f"Gagal memproses {os.path.basename(path)}: {str(e)}")
        progress_bar.progress((idx + 1) / len(file_paths))
    status_text.text("")
    st.success("🎉 Seluruh pengamatan dan validasi satelit berhasil diproses.")

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
        Aplikasi web ini mengekstrak titik belok fajar sadiq secara otonom. Terintegrasi dengan radar satelit cuaca untuk validasi absolut.
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
        st.dataframe(
            df_cloud, 
            use_container_width=True,
            column_config={
                "Link_Grafik": st.column_config.LinkColumn("Link Grafik", display_text="🖼️ Lihat Grafik Plot"),
                "Link_DataMentah": st.column_config.LinkColumn("Link Data Mentah", display_text="📁 Buka File Mentah")
            }
        )
        st.download_button("⬇️ Unduh CSV", df_cloud.to_csv(index=False).encode('utf-8'), 'Rekap_Kawakib_Cloud.csv', 'text/csv')

with tab_dashboard:
    st.header("📊 Ringkasan Statistik & Kalibrasi Standar Kemenag")
    df_stat = load_data_from_google_sheets()
    
    if not df_stat.empty:
        df_stat['Fajar_Alt'] = pd.to_numeric(df_stat['Fajar_Alt'], errors='coerce')
        df_stat['Awan_%'] = pd.to_numeric(df_stat['Awan_%'], errors='coerce')
        df_stat['Garis_Dasar'] = pd.to_numeric(df_stat['Garis_Dasar'], errors='coerce')
        df_stat['Awan_Satelit_%'] = pd.to_numeric(df_stat.get('Awan_Satelit_%', pd.Series(dtype=float)), errors='coerce')
        
        if 'Kota' not in df_stat.columns and 'Lokasi' in df_stat.columns:
            df_stat['Kota'] = df_stat['Lokasi'].apply(lambda x: re.split(r'[-,\|]', str(x))[-1].strip())
            
        df_kemenag_ideal = df_stat[
            (df_stat['Garis_Dasar'] >= 20.5) & 
            (df_stat['Awan_%'] <= 5.0) & 
            (df_stat['Koreksi_Bulan'] == 'Pasif')
        ]
        
        rata_rata_kemenag = df_kemenag_ideal['Fajar_Alt'].mean() if not df_kemenag_ideal.empty else 0.0
        rata_rata_total = df_stat['Fajar_Alt'].mean()
        
        standar_kemenag = -20.0
        selisih_kemenag = rata_rata_kemenag - standar_kemenag
        selisih_total = rata_rata_total - standar_kemenag
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data Keseluruhan", len(df_stat))
        c2.metric("Rata-rata Standar Kemenag (Ideal)", f"{rata_rata_kemenag:.2f}°", f"{selisih_kemenag:+.2f}° dari -20°", delta_color="inverse")
        c3.metric("Total Rata-rata (Pembanding)", f"{rata_rata_total:.2f}°", f"{selisih_total:+.2f}° dari -20°", delta_color="inverse")
        c4.metric("Data Ideal Tersaring", len(df_kemenag_ideal))
        
        # --- BLOK 1: DISTRIBUSI BERDASARKAN KECERLANGAN (MPSAS) ---
        st.markdown(f"<br><b>Distribusi {len(df_kemenag_ideal)} Data Ideal Berdasarkan Kecerlangan Langit (Garis Dasar):</b>", unsafe_allow_html=True)
        
        df_bin1 = df_kemenag_ideal[(df_kemenag_ideal['Garis_Dasar'] >= 20.5) & (df_kemenag_ideal['Garis_Dasar'] < 21.0)]
        df_bin2 = df_kemenag_ideal[(df_kemenag_ideal['Garis_Dasar'] >= 21.0) & (df_kemenag_ideal['Garis_Dasar'] < 21.5)]
        df_bin3 = df_kemenag_ideal[df_kemenag_ideal['Garis_Dasar'] >= 21.5]
        
        len1, len2, len3 = len(df_bin1), len(df_bin2), len(df_bin3)
        mean1 = f"({df_bin1['Fajar_Alt'].mean():.2f}°)" if len1 > 0 else "(-)"
        mean2 = f"({df_bin2['Fajar_Alt'].mean():.2f}°)" if len2 > 0 else "(-)"
        mean3 = f"({df_bin3['Fajar_Alt'].mean():.2f}°)" if len3 > 0 else "(-)"
        
        bc1, bc2, bc3 = st.columns(3)
        bc1.info(f"**20.50 – 20.99 Mpsas:** \n### {len1} Data {mean1}")
        bc2.info(f"**21.00 – 21.49 Mpsas:** \n### {len2} Data {mean2}")
        bc3.info(f"**≥ 21.50 Mpsas:** \n### {len3} Data {mean3}")
        
        # --- BLOK 2: DISTRIBUSI BERDASARKAN KEDALAMAN (FAJAR ALT) ---
        st.markdown(f"<br><b>Distribusi {len(df_kemenag_ideal)} Data Ideal Berdasarkan Kedalaman Fajar (Titik Belok):</b>", unsafe_allow_html=True)
        
        # Menggunakan logika kedalaman: < -19.5 artinya lebih dalam secara minus (misal: -19.8, -20.2)
        df_fajar1 = df_kemenag_ideal[df_kemenag_ideal['Fajar_Alt'] <= -19.5]
        df_fajar2 = df_kemenag_ideal[(df_kemenag_ideal['Fajar_Alt'] > -19.5) & (df_kemenag_ideal['Fajar_Alt'] <= -19.0)]
        df_fajar3 = df_kemenag_ideal[(df_kemenag_ideal['Fajar_Alt'] > -19.0) & (df_kemenag_ideal['Fajar_Alt'] <= -18.5)]
        df_fajar4 = df_kemenag_ideal[df_kemenag_ideal['Fajar_Alt'] > -18.5]
        
        len_f1, len_f2, len_f3, len_f4 = len(df_fajar1), len(df_fajar2), len(df_fajar3), len(df_fajar4)
        
        mean_f1 = f"({df_fajar1['Fajar_Alt'].mean():.2f}°)" if len_f1 > 0 else "(-)"
        mean_f2 = f"({df_fajar2['Fajar_Alt'].mean():.2f}°)" if len_f2 > 0 else "(-)"
        mean_f3 = f"({df_fajar3['Fajar_Alt'].mean():.2f}°)" if len_f3 > 0 else "(-)"
        mean_f4 = f"({df_fajar4['Fajar_Alt'].mean():.2f}°)" if len_f4 > 0 else "(-)"
        
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.success(f"**Lebih dalam dari -19.5°:** \n### {len_f1} Data {mean_f1}")
        fc2.success(f"**-19.5° s/d -19.0°:** \n### {len_f2} Data {mean_f2}")
        fc3.success(f"**-19.0° s/d -18.5°:** \n### {len_f3} Data {mean_f3}")
        fc4.success(f"**Lebih dangkal dari -18.5°:** \n### {len_f4} Data {mean_f4}")
        # ---------------------------------------------------------------
        
        st.divider()
        st.subheader("📉 Komparasi Kedalaman Fajar per Kota (Filter Standar Kemenag: Garis Dasar ≥ 20.5 Mpsas)")
        
        df_loc = df_kemenag_ideal.groupby('Kota')['Fajar_Alt'].mean().dropna().reset_index()
        if df_loc.empty:
            df_loc = df_stat.groupby('Kota')['Fajar_Alt'].mean().dropna().reset_index()
            st.warning("Perhatian: Belum ada data yang memenuhi kriteria ideal. Grafik di bawah menampilkan seluruh rata-rata data yang tersedia.")
        
        if not df_loc.empty:
            bar_chart = alt.Chart(df_loc).mark_bar(color='#1D9A9C', cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Kota:N', title='Lokasi Pengamatan', sort='y'),
                y=alt.Y('Fajar_Alt:Q', title='Rata-rata Kedalaman (°)', scale=alt.Scale(domain=[-22, -15])),
                tooltip=['Kota', alt.Tooltip('Fajar_Alt:Q', format='.2f', title='Fajar (°)')]
            )
            kemenag_line = alt.Chart(pd.DataFrame({'Standar': [standar_kemenag]})).mark_rule(
                color='#d9534f', strokeWidth=2, strokeDash=[5, 5]
            ).encode(y='Standar:Q')
            kemenag_label = alt.Chart(pd.DataFrame({'Standar': [standar_kemenag], 'Label': ['Standar Kemenag (-20°)']})).mark_text(
                color='#d9534f', align='left', baseline='bottom', dx=5, dy=-5, fontSize=12, fontWeight='bold'
            ).encode(y='Standar:Q', text='Label:N')
            
            st.altair_chart(bar_chart + kemenag_line + kemenag_label, use_container_width=True)

        st.divider()
        st.subheader("🌍 Analisis Spasial & Faktor Lingkungan")
        
        if 'Lintang' in df_stat.columns and 'Bujur' in df_stat.columns:
            df_map = df_stat.dropna(subset=['Lintang', 'Bujur', 'Fajar_Alt']).copy()
            if not df_map.empty:
                df_map['lat'] = pd.to_numeric(df_map['Lintang'], errors='coerce')
                df_map['lon'] = pd.to_numeric(df_map['Bujur'], errors='coerce')
                df_map_agg = df_map.groupby(['Kota', 'lat', 'lon']).agg(
                    Fajar_Rata2=('Fajar_Alt', 'mean'),
                    Total_Observasi=('Fajar_Alt', 'count')
                ).reset_index()
                
                col_map, col_tab = st.columns([1, 1])
                with col_map:
                    st.markdown("**Peta Persebaran Titik Observasi**")
                    st.map(df_map_agg[['lat', 'lon']], zoom=4, use_container_width=True)
                with col_tab:
                    st.markdown("**Tabel Deviasi Kemenag per Koordinat**")
                    df_tabel = df_map_agg.rename(columns={'lat': 'Lintang', 'lon': 'Bujur', 'Fajar_Rata2': 'Rata-rata Fajar (°)'})
                    df_tabel['Deviasi dari Kemenag'] = df_tabel['Rata-rata Fajar (°)'] - (-20.0)
                    st.dataframe(df_tabel.style.format({'Rata-rata Fajar (°)': '{:.2f}', 'Deviasi dari Kemenag': '{:+.2f}'}), use_container_width=True)
        
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Berdasarkan Polusi Cahaya (Bortle)**")
            st.bar_chart(df_stat.groupby('Bortle')['Fajar_Alt'].mean().dropna())
        with col2:
            st.markdown("**Berdasarkan Koreksi Bulan**")
            st.bar_chart(df_stat.groupby('Koreksi_Bulan')['Fajar_Alt'].mean().dropna())
            
    else: st.info("Belum ada data untuk dianalisis.")

with tab_algoritma:
    st.header("📖 Metodologi, Landasan Matematis & Spesifikasi Algoritma")
    st.markdown(r"""
    Aplikasi ini dibangun sebagai instrumen riset falakiyah analitis yang menggabungkan astrometri klasik, pemrosesan sinyal digital (*Digital Signal Processing*), dan validasi satelit.

    ### 1. Ekstraksi Fajar: Metode SIGMAG-STAB
    *Smoothed Gradient - Median Absolute Deviation Stability* (SIGMAG-STAB) adalah algoritma utama yang dirancang untuk mendeteksi titik belok (*inflection point*) awal fajar sadiq pada kurva fotometri yang bising.

    **A. Penghalusan Derau (Savitzky-Golay Filter)**
    Data mentah dihaluskan menggunakan konvolusi polinomial untuk membuang *noise* frekuensi tinggi tanpa merusak bentuk asli transisi fajar:
    $$ Y_j^* = \frac{1}{N} \sum_{i=-m}^{m} C_i Y_{j+i} $$
    Di mana $Y_j^*$ adalah nilai kecerlangan yang dihaluskan, dan $C_i$ adalah koefisien konvolusi polinomial orde-2.

    **B. Kalkulasi Gradien (Turunan Pertama)**
    Mencari laju perubahan kecerlangan langit terhadap perubahan sudut matahari:
    $$ \nabla y = \frac{d(\text{Mpsas})}{d(\text{Altitude})} \approx \frac{y_{i+1} - y_{i-1}}{x_{i+1} - x_{i-1}} $$

    **C. Pendeteksian Anomali dengan MAD (*Median Absolute Deviation*)**
    Garis dasar malam (*baseline*) dihitung saat matahari berada di $\le -20^\circ$. Kestabilan malam diukur menggunakan MAD (pendekatan statistik robust yang kebal terhadap pencilan/outlier):
    $$ \text{MAD} = 1.4826 \times \text{median}(|X_i - \tilde{X}|) $$
    Fajar sadiq ditetapkan secara sah ketika nilai gradien $\nabla y$ menembus ambang batas deviasi negatif secara berturut-turut dalam *n-langkah* observasi.

    ---

    ### 2. Ekstraksi Fajar: Metode SIGMOID (*Curve Fitting*)
    Metode sekunder ini menggunakan regresi non-linear kuadrat terkecil (*Non-linear Least Squares*) yang memaksa kurva fajar malam hari untuk mengikuti fungsi logistik (Kurva-S). Sangat ideal untuk langit tipe 1 dan 2.

    Persamaan fungsi logistik yang digunakan:
    $$ f(x) = \frac{L}{1 + e^{-k(x - x_0)}} + b $$
    Dimana:
    * $L$ : Amplitudo maksimal kurva (Selisih gelap malam dan terang pagi)
    * $k$ : Laju kecuraman transisi fajar (Tingkat hamburan Rayleigh)
    * $x_0$ : Titik tengah transisi fajar
    * $b$ : Batas asimtotik (*Garis dasar/Baseline langit malam*)
    Titik awal fajar ditarik dari nilai $x$ (Altitude) di mana kurva mulai menyimpang dari nilai $b$.

    ---

    ### 3. Validasi Lingkungan Cerdas (*Smart Cloud & Satellite Validation*)
    Pendeteksian fajar seringkali mengalami *False Positive* (Positif Palsu) akibat kondisi mikroklimat lokal. Untuk memitigasi hal ini, aplikasi menggunakan validasi ganda:

    * **Validasi Mikro (Sensor SQM - *Jendela Kritis $\pm 15$ Menit*):** Algoritma membidik rentang waktu tepat 15 menit sebelum hingga sesudah titik belok fajar. Sistem menghitung *Rolling Standard Deviation* pada rentang sempit ini. Jika fluktuasi melebihi batas dinamis, observasi dilabeli memiliki gangguan awan.
    * **Validasi Makro (Satelit Open-Meteo):**
        Sistem mengirim titik koordinat observasi ke API Satelit Cuaca Global (*Reverse Geocoding*) untuk mengekstrak persentase tutupan awan dari luar angkasa. 
    
    > 💡 **Anomali Albedo Perkotaan (Kenapa Satelit Wajib Ada?):**
    Di kawasan urban dengan polusi cahaya ekstrem, awan pekat bertipe merata (*stratus*) bertindak sebagai reflektor yang memantulkan lampu kota kembali ke bumi. Akibatnya, sensor SQM merekam stabilitas cahaya buatan (seolah awan $0\% - 2\%$), padahal aslinya langit sedang mendung total. Kondisi ini membuat cahaya Fajar Sadiq tertahan awan dan baru terdeteksi pada kedalaman anomali (misal: $-13^\circ$). Validasi satelit berfungsi membongkar "ilusi stabilitas" ini dengan menampakkan persentase mendung yang sebenarnya (misal: $> 80\%$).

    ---

    ### 4. Klasifikasi Polusi Cahaya (Berdasarkan Disertasi Basthoni)
    Aplikasi ini menggunakan **4 Skala Penyederhanaan Polusi Cahaya**. Klasifikasi ini merupakan penyederhanaan dari 9 Skala Bortle Internasional, yang diadaptasi dan didasarkan secara empiris pada **Visibilitas Ketampakan Fajar Kadzib (*Zodiacal Light*)** di lokasi pengamatan (Merujuk pada *Disertasi Mochammad Basthoni*).

    Pembagian 4 kuadran klasifikasi tersebut adalah:
    1.  **Tipe 1 (Langit Gelap | $\ge 21.3$ Mpsas):** Bebas polusi cahaya. Fajar Kadzib tampak sangat jelas dan menjulang secara vertikal sebelum Fajar Sadiq menyingsing.
    2.  **Tipe 2 (Agak Gelap | $20.2 - 21.29$ Mpsas):** Area pedesaan/pinggiran. Fajar Kadzib masih dapat diobservasi secara visual meski kontrasnya mulai menurun akibat sebaran cahaya di horizon.
    3.  **Tipe 3 (Agak Terang | $19.1 - 20.19$ Mpsas):** Area suburban/transisi. Fajar Kadzib sangat sulit hingga hampir mustahil dibedakan dengan pendaran polusi cahaya kota di ufuk.
    4.  **Tipe 4 (Terang/Urban | $< 19.1$ Mpsas):** Pusat kota. Polusi cahaya absolut menenggelamkan Fajar Kadzib sepenuhnya. 

    > 🔬 **Filter Kalibrasi Kemenag:** Untuk menjaga integritas data empiris Fajar Sadiq yang murni, mesin analitik hanya memasukkan data pada rentang **Tipe 1 dan Tipe 2 (Garis Dasar $\ge 20.5$ Mpsas)** yang lolos verifikasi radar cuaca lokal ($\le 5\%$) dan bebas kontaminasi cahaya bulan.
    """)
