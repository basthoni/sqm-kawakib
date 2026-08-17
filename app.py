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
import io

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
# FUNGSI BACKEND (CLOUD, API & SMART INGESTION PIPELINE)
# =====================================================================
try:
    cloudinary.config(
        cloud_name=st.secrets["cloudinary"]["cloud_name"].strip() if "cloudinary" in st.secrets else None,
        api_key=st.secrets["cloudinary"]["api_key"].strip() if "cloudinary" in st.secrets else None,
        api_secret=st.secrets["cloudinary"]["api_secret"].strip() if "cloudinary" in st.secrets else None,
        secure=True
    )
except: pass

def normalisasi_ke_pysqm(input_path, output_path):
    df = pd.read_csv(input_path, sep=',')
    df['Local_Time'] = pd.to_datetime(df['Date/Time'])
    df['UTC_Time'] = df['Local_Time'] - pd.Timedelta(hours=7)
    df['Local_Str'] = df['Local_Time'].dt.strftime('%Y-%m-%dT%H:%M:%S.000')
    df['UTC_Str'] = df['UTC_Time'].dt.strftime('%Y-%m-%dT%H:%M:%S.000')
    df_final = pd.DataFrame({
        'col1': df['UTC_Str'], 'col2': df['Local_Str'], 'col3': df['Temp(C)'],
        'col4': 0.000, 'col5': 0.000, 'col6': df['MPSAS']      
    })
    with open(output_path, 'w') as f:
        f.write("# Definition of the community standard for skyglow observations 1.0\n")
        f.write("# Location name: Bosscha / Stasiun Pengamatan\n")
        f.write("# Position: -6.8276, 107.6163, 1310\n# Local timezone: UTC+7\n")
        f.write("# UTC Date & Time, Local Date & Time, Temperature, Counts, Frequency, MSAS\n# END OF HEADER\n")
    df_final.to_csv(output_path, sep=';', index=False, header=False, mode='a')

def normalisasi_lapan_ke_pysqm(input_path, output_path):
    lon, lat = None, None
    data_lines = []
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                match_lon = re.search(r'lon[a-z]*\s*[:=]?\s*([-+]?\d*\.\d+|\d+)', line, re.IGNORECASE)
                match_lat = re.search(r'lat[a-z]*\s*[:=]?\s*([-+]?\d*\.\d+|\d+)', line, re.IGNORECASE)
                if match_lon: lon = float(match_lon.group(1))
                if match_lat: lat = float(match_lat.group(1))
            elif line.strip(): data_lines.append(line)

    timezone_offset = int(round(lon / 15.0)) if lon is not None else 7
    lat_val = lat if lat is not None else -6.8276 
    lon_val = lon if lon is not None else 107.6163
    data_str = '\n'.join(data_lines)
    df = pd.read_csv(io.StringIO(data_str), sep=r'\s+', names=['UTC_DateTime', 'Temp', 'MPSAS', 'Q'])
    df['UTC_Time'] = pd.to_datetime(df['UTC_DateTime'])
    df['Local_Time'] = df['UTC_Time'] + pd.Timedelta(hours=timezone_offset)
    df_final = pd.DataFrame({
        'col1': df['UTC_Time'].dt.strftime('%Y-%m-%dT%H:%M:%S.000'),
        'col2': df['Local_Time'].dt.strftime('%Y-%m-%dT%H:%M:%S.000'),
        'col3': df['Temp'], 'col4': df['Q'], 'col5': 0.000, 'col6': df['MPSAS']
    })
    with open(output_path, 'w') as f:
        f.write("# Definition of the community standard for skyglow observations 1.0\n")
        f.write(f"# Position: {lat_val}, {lon_val}, 0\n# Local timezone: UTC+{timezone_offset}\n")
        f.write("# UTC Date & Time, Local Date & Time, Temperature, Counts, Frequency, MSAS\n# END OF HEADER\n")
    df_final.to_csv(output_path, sep=';', index=False, header=False, mode='a')

def proses_file_masuk(file_input):
    file_standar_sementara = file_input + "_standardized.dat"
    try:
        with open(file_input, 'r', encoding='utf-8', errors='ignore') as f:
            content_sample = f.read(2000)
    except: content_sample = ""
    if 'Date/Time' in content_sample and ',' in content_sample:
        normalisasi_ke_pysqm(file_input, file_standar_sementara)
        return file_standar_sementara
    elif 'UTC_DateTime' in content_sample or 'MPSAS' in content_sample or ('#' in content_sample and 'lon' in content_sample.lower()):
        normalisasi_lapan_ke_pysqm(file_input, file_standar_sementara)
        return file_standar_sementara
    return file_input

def get_city_from_coords(lat, lon):
    if lat is None or lon is None: return "Unknown"
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Kawakib-SQM-Analyzer/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            address = data.get('address', {})
            city = address.get('city') or address.get('county') or address.get('state_district') or address.get('town') or address.get('village')
            if city: return city.replace("Kabupaten ", "").replace("Kota ", "")
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
        for key in data_dict.keys():
            if key not in headers: headers.append(key)
        sheet.update(range_name='A1', values=[headers])
        
        row_to_update = None
        for i, row in enumerate(existing_data[1:], start=2):
            if len(row) > 2 and row[0] == str(data_dict["Tanggal"]) and row[2] == str(data_dict["Lokasi"]) and row[5] == str(data_dict["Metode"]):
                row_to_update = i; break
        values = [str(data_dict.get(h, "")) for h in headers]
        if row_to_update:
            cell_list = sheet.range(f'A{row_to_update}:{chr(65 + len(headers) - 1)}{row_to_update}')
            for cell, val in zip(cell_list, values): cell.value = val
            sheet.update_cells(cell_list)
        else: sheet.append_row(values)
        return True
    except: return False

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
        results = service.files().list(
            q="mimeType != 'application/vnd.google-apps.folder' and (name contains '.dat' or name contains '.DAT' or name contains '.txt') and trashed = false",
            fields="files(id, name)", pageSize=1000
        ).execute()
        for file in results.get('files', []):
            file_path = os.path.join(tempfile.gettempdir(), file['name'])
            request = service.files().get_media(fileId=file['id'])
            with open(file_path, "wb") as f: f.write(request.execute())
            downloaded_paths.append(file_path)
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
    return None

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
    processed_path = proses_file_masuk(file_path)
    site, lat, lon, utc_offset, data_start = read_header_and_find_data_start(processed_path)
    if lat is None or lon is None: lat, lon, utc_offset = -7.972, 114.425, 7
    df = pd.read_csv(processed_path, skiprows=data_start, sep=";", header=None, names=["utc","local","temp","cnt","hz","mpsas"], engine="python", on_bad_lines="skip")
    df["local_dt"] = pd.to_datetime(df["local"], errors="coerce")
    df = df.dropna(subset=["local_dt","mpsas"])
    am = df[(df["local_dt"].dt.hour < 12) & (df["mpsas"] > 0)].copy()
    if not am.empty:
        am["sun_alt"] = solar_alt(am["local_dt"], lat, lon, utc_offset)
        am = am.sort_values("sun_alt").reset_index(drop=True)
    date_str = am["local_dt"].iloc[0].strftime("%Y-%m-%d") if not am.empty else "Unknown"
    return am, site, lat, lon, utc_offset, date_str, processed_path

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
            am, site, lat, lon, utc_offset, date_str, processed_path = load_sqm_data(path)
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
            kota = get_city_from_coords(lat, lon)
            if kota == "Unknown": kota = re.split(r'[-,\|]', site)[-1].strip()
            
            fig_width = max(6, min(15, len(am) / 50))
            fig, ax = plt.subplots(figsize=(fig_width, 5))
            ax.plot(am["sun_alt"], am["mpsas_corrected"], color="#1A3C40", alpha=0.8, linewidth=1.5, label="SQM Terkoreksi")
            if is_corrected: ax.plot(am["sun_alt"], am["mpsas"], color="#808080", alpha=0.4, linestyle=":", label="SQM Mentah")
            if onset_alt is not None:
                ax.axvline(onset_alt, color="#1D9A9C", linestyle="--", linewidth=2, label=f"Titik Belok ({onset_alt:.2f}°)")
                ax.scatter([onset_alt], [onset_msas], color="#1D9A9C", s=60, zorder=5)
            ax.invert_yaxis()
            ax.set_xlim(-30, -5)
            ax.set_xlabel("Ketinggian Matahari (Derajat)", fontweight='bold')
            ax.set_ylabel("Kecerlangan Langit (Mpsas)", fontweight='bold')
            ax.set_title(f"{site} | {date_str} [{method}]", color="#1A3C40", fontweight='bold')
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            
            plot_url, raw_url = "", ""
            if not df_existing.empty and {"Tanggal", "Lokasi", "Metode"}.issubset(df_existing.columns):
                match = df_existing[(df_existing["Tanggal"].astype(str).str.strip() == str(date_str)) & (df_existing["Lokasi"].astype(str).str.strip() == str(site)) & (df_existing["Metode"].astype(str).str.strip() == str(method))]
                if not match.empty:
                    plot_url = str(match.iloc[0].get("Link_Grafik", match.iloc[0].get("Link Grafik", ""))).strip()
                    raw_url = str(match.iloc[0].get("Link_DataMentah", match.iloc[0].get("Link Data Mentah", ""))).strip()

            if not plot_url: plot_url = upload_plot_to_cloudinary(fig, f"Plot_{site}_{date_str}_{method}".replace(" ", "_"))
            if not raw_url: raw_url = upload_raw_to_cloudinary(processed_path, f"Raw_{site}_{date_str}_{method}.dat".replace(" ", "_"))
            
            save_to_google_sheets({
                "Tanggal": date_str, "Kota": kota, "Lokasi": site, "Lintang": lat, "Bujur": lon, 
                "Metode": method, "Bortle": lp_category.split("(")[-1].replace(")",""),
                "Awan_%": round(cloud_pct, 1), "Awan_Satelit_%": sat_cloud_pct if sat_cloud_pct is not None else "",
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
    st.success("🎉 Selesai.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    method = st.selectbox("Metode Ekstraksi Fajar", ["SIGMAG-STAB", "SIGMOID"])
    if st.button("📡 Tarik Data dari SOOF (Drive)"):
        with st.spinner("Menyedot data..."):
            file_paths = sync_from_soof_drive()
            if file_paths: process_and_save_data(file_paths, method, load_data_from_google_sheets())

st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 4px;">
        <img src="{KAWAKIB_LOGO_URL}" style="height: 48px; width: auto; object-fit: contain;">
        <h1 style='font-family: Lora, serif; color: #1A3C40; font-size: 1.65rem; margin: 0;'>KAWAKIB INSTITUTE: SQM Fajar Analyzer</h1>
    </div>
    <div style="border-bottom: 2px solid #1D9A9C; margin-top: 8px; margin-bottom: 10px;"></div>
""", unsafe_allow_html=True)

tab_analisis, tab_histori, tab_statistik_utama, tab_algoritma = st.tabs([
    "🚀 Analisis Data", "☁️ Basis Data", "📊 Statistik & Analisis", "📖 Metodologi & Algoritma"
])

with tab_analisis:
    uploaded_files = st.file_uploader("Unggah File Observasi", accept_multiple_files=True, type=['dat', 'DAT', 'txt', 'TXT', 'zip', 'ZIP'])
    if uploaded_files and st.button("Mulai Kalkulasi Fotometri 🚀"):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            for up in uploaded_files:
                p = os.path.join(temp_dir, up.name)
                with open(p, "wb") as f: f.write(up.getbuffer())
                if up.name.endswith(('.zip', '.ZIP')):
                    with zipfile.ZipFile(p, 'r') as z: z.extractall(temp_dir)
                    for root, _, files in os.walk(temp_dir):
                        for f in files:
                            if f.endswith(('.dat', '.txt', '.DAT', '.TXT')): file_paths.append(os.path.join(root, f))
                else: file_paths.append(p)
            if file_paths: process_and_save_data(file_paths, method, load_data_from_google_sheets())

with tab_histori:
    st.header("☁️ Basis Data Fotometri Terpusat")
    if st.button("🔄 Sinkronisasi"): st.rerun()
    df_cloud = load_data_from_google_sheets()
    if df_cloud.empty: st.info("Basis data kosong.")
    else:
        # Deteksi pintar nama kolom link di database utama
        col_g = "Link_Grafik" if "Link_Grafik" in df_cloud.columns else ("Link Grafik" if "Link Grafik" in df_cloud.columns else None)
        col_d = "Link_DataMentah" if "Link_DataMentah" in df_cloud.columns else ("Link Data Mentah" if "Link Data Mentah" in df_cloud.columns else None)
        cfg_db = {}
        if col_g: cfg_db[col_g] = st.column_config.LinkColumn("Plot Fajar", display_text="🖼️ Lihat Grafik Plot")
        if col_d: cfg_db[col_d] = st.column_config.LinkColumn("Data Mentah", display_text="📁 Buka File Mentah")
        
        st.dataframe(df_cloud, use_container_width=True, column_config=cfg_db)

with tab_statistik_utama:
    st.header("📊 Pusat Analisis Statistik & Korelasi Variabel")
    sub_ideal, sub_anomali, sub_korelasi, sub_peta = st.tabs([
        "🌟 Data Ideal (Kemenag)", "⚠️ Data Anomali & Pemeriksaan", "📈 Korelasi Variabel", "🗺️ Peta Spasial"
    ])
    
    df_stat = load_data_from_google_sheets()
    
    if not df_stat.empty:
        # Konversi tipe data numerik dengan paksa (ganti teks aneh dengan NaN)
        for col in ['Fajar_Alt', 'Awan_%', 'Garis_Dasar', 'Lintang', 'Bujur']:
            if col in df_stat.columns:
                df_stat[col] = pd.to_numeric(df_stat[col], errors='coerce')
        
        if 'Kota' not in df_stat.columns and 'Lokasi' in df_stat.columns:
            df_stat['Kota'] = df_stat['Lokasi'].apply(lambda x: re.split(r'[-,\|]', str(x))[-1].strip())
            
        # =====================================================================
        # KEMBALIKAN LAYOUT CANTIK SUB-TAB IDEAL KEMENAG
        # =====================================================================
        with sub_ideal:
            st.subheader("🌟 Analisis Data Ideal (Standar Kemenag)")
            df_kemenag_ideal = df_stat[(df_stat['Garis_Dasar'] >= 20.5) & (df_stat['Awan_%'] <= 5.0) & (df_stat['Koreksi_Bulan'] == 'Pasif')]
            
            rata_rata_kemenag = df_kemenag_ideal['Fajar_Alt'].mean() if not df_kemenag_ideal.empty else 0.0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Data Keseluruhan", len(df_stat))
            c2.metric("Rata-rata Standar Kemenag (Ideal)", f"{rata_rata_kemenag:.2f}°", f"{rata_rata_kemenag - (-20.0):+.2f}° dari -20°", delta_color="inverse")
            c3.metric("Data Ideal Tersaring", len(df_kemenag_ideal))
            
            st.markdown(f"<br><b>Distribusi {len(df_kemenag_ideal)} Data Ideal Berdasarkan Kedalaman Fajar (Titik Belok):</b>", unsafe_allow_html=True)
            df_fajar1 = df_kemenag_ideal[df_kemenag_ideal['Fajar_Alt'] <= -19.5]
            df_fajar2 = df_kemenag_ideal[(df_kemenag_ideal['Fajar_Alt'] > -19.5) & (df_kemenag_ideal['Fajar_Alt'] <= -19.0)]
            df_fajar3 = df_kemenag_ideal[(df_kemenag_ideal['Fajar_Alt'] > -19.0) & (df_kemenag_ideal['Fajar_Alt'] <= -18.5)]
            df_fajar4 = df_kemenag_ideal[df_kemenag_ideal['Fajar_Alt'] > -18.5]
            
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.success(f"**Lebih dalam dari -19.5°:** \n### {len(df_fajar1)} Data ({df_fajar1['Fajar_Alt'].mean():.2f}°)" if not df_fajar1.empty else "**< -19.5°:** \n### 0 Data (-)")
            fc2.success(f"**-19.5° s/d -19.0°:** \n### {len(df_fajar2)} Data ({df_fajar2['Fajar_Alt'].mean():.2f}°)" if not df_fajar2.empty else "**-19.5° s/d -19.0°:** \n### 0 Data (-)")
            fc3.success(f"**-19.0° s/d -18.5°:** \n### {len(df_fajar3)} Data ({df_fajar3['Fajar_Alt'].mean():.2f}°)" if not df_fajar3.empty else "**-19.0° s/d -18.5°:** \n### 0 Data (-)")
            fc4.success(f"**Lebih dangkal dari -18.5°:** \n### {len(df_fajar4)} Data ({df_fajar4['Fajar_Alt'].mean():.2f}°)" if not df_fajar4.empty else "**> -18.5°:** \n### 0 Data (-)")


        # =====================================================================
        # KEMBALIKAN LAYOUT CANTIK SUB-TAB ANOMALI
        # =====================================================================
        with sub_anomali:
            st.subheader("⚠️ Analisis Data Anomali & Tabel Pemeriksaan Kasus")
            df_anomali = df_stat[~((df_stat['Garis_Dasar'] >= 20.5) & (df_stat['Awan_%'] <= 5.0) & (df_stat['Koreksi_Bulan'] == 'Pasif'))].dropna(subset=['Fajar_Alt'])
            
            if df_anomali.empty:
                st.success("Luar biasa! Tidak ada data anomali.")
            else:
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric("Total Data Ditolak", len(df_anomali), "Non-Ideal", delta_color="off")
                ac2.metric("Rata-rata Kedalaman", f"{df_anomali['Fajar_Alt'].mean():.2f}°", "Cenderung Dangkal", delta_color="inverse")
                ac3.metric("Rata-rata Garis Dasar", f"{df_anomali['Garis_Dasar'].mean():.2f} Mpsas", "Tercemar Polusi", delta_color="inverse")
                
                st.markdown(f"<br><b>Distribusi {len(df_anomali)} Data Anomali Berdasarkan Kedalaman (Rentang 0.5°):</b>", unsafe_allow_html=True)
                df_a1 = df_anomali[df_anomali['Fajar_Alt'] <= -18.5]
                df_a2 = df_anomali[(df_anomali['Fajar_Alt'] > -18.5) & (df_anomali['Fajar_Alt'] <= -18.0)]
                df_a3 = df_anomali[(df_anomali['Fajar_Alt'] > -18.0) & (df_anomali['Fajar_Alt'] <= -17.5)]
                df_a4 = df_anomali[df_anomali['Fajar_Alt'] > -17.5]
                
                anc1, anc2, anc3, anc4 = st.columns(4)
                anc1.warning(f"**< -18.5°:** \n### {len(df_a1)} Data")
                anc2.warning(f"**-18.5° s/d -18.0°:** \n### {len(df_a2)} Data")
                anc3.warning(f"**-18.0° s/d -17.5°:** \n### {len(df_a3)} Data")
                anc4.warning(f"**> -17.5°:** \n### {len(df_a4)} Data")

                st.markdown("<br>", unsafe_allow_html=True)
                hist_anom = alt.Chart(df_anomali).mark_bar(color='#d9534f', opacity=0.8).encode(
                    alt.X("Fajar_Alt:Q", bin=alt.Bin(step=0.5), title="Ketinggian Matahari (Derajat) - Rentang 0.5°", scale=alt.Scale(domain=[-20, -10])),
                    alt.Y('count()', title='Jumlah Kasus Anomali'),
                    tooltip=['count()', alt.Tooltip('mean(Fajar_Alt):Q', format='.2f', title='Rata-rata Alt')]
                ).properties(height=300)
                st.altair_chart(hist_anom, use_container_width=True)

                st.divider()
                st.subheader("🔍 Tabel Rincian Data Berdasarkan Kelompok Anomali")
                pilihan_kelompok = st.selectbox(
                    "Pilih Kelompok Anomali untuk Diperiksa:",
                    [
                        f"Kelompok 1: Kedalaman < -18.5° ({len(df_a1)} Data)",
                        f"Kelompok 2: Kedalaman -18.5° s/d -18.0° ({len(df_a2)} Data)",
                        f"Kelompok 3: Kedalaman -18.0° s/d -17.5° ({len(df_a3)} Data)",
                        f"Kelompok 4: Kedalaman > -17.5° ({len(df_a4)} Data)"
                    ]
                )
                
                if "Kelompok 1" in pilihan_kelompok: df_tampil = df_a1
                elif "Kelompok 2" in pilihan_kelompok: df_tampil = df_a2
                elif "Kelompok 3" in pilihan_kelompok: df_tampil = df_a3
                else: df_tampil = df_a4
                
                # Deteksi pintar nama kolom link di database untuk tabel anomali
                link_g_anom = "Link_Grafik" if "Link_Grafik" in df_tampil.columns else ("Link Grafik" if "Link Grafik" in df_tampil.columns else None)
                link_d_anom = "Link_DataMentah" if "Link_DataMentah" in df_tampil.columns else ("Link Data Mentah" if "Link Data Mentah" in df_tampil.columns else None)
                
                cols_to_show_anom = ['Tanggal', 'Kota', 'Lokasi', 'Metode', 'Garis_Dasar', 'Awan_%', 'Koreksi_Bulan', 'Fajar_Alt']
                cfg_anom = {}
                if link_g_anom:
                    cols_to_show_anom.append(link_g_anom)
                    cfg_anom[link_g_anom] = st.column_config.LinkColumn("Plot Fajar", display_text="🖼️ Lihat Grafik Plot")
                if link_d_anom:
                    cols_to_show_anom.append(link_d_anom)
                    cfg_anom[link_d_anom] = st.column_config.LinkColumn("Data Mentah", display_text="📁 Buka File Mentah")

                st.dataframe(
                    df_tampil[cols_to_show_anom], 
                    use_container_width=True,
                    column_config=cfg_anom
                )

        # =====================================================================
        # BAGIAN KORELASI YANG SUDAH AMAN
        # =====================================================================
        with sub_korelasi:
            st.subheader("📈 Analisis Korelasi Antar Variabel Astrometri")
            st.markdown("Visualisasi regresi untuk menguji hubungan sebab-akibat antara faktor lingkungan (Polusi Cahaya & Awan Ufuk Timur) terhadap pergeseran titik belok fajar.")
            
            df_corr = df_stat.dropna(subset=['Garis_Dasar', 'Fajar_Alt', 'Awan_%']).copy()
            
            if df_corr.empty:
                st.warning("⚠️ Data belum mencukupi untuk diregresi atau beberapa kolom tidak terbaca sebagai angka bulat.")
            else:
                col_k1, col_k2 = st.columns(2)
                
                with col_k1:
                    st.markdown("**1. Korelasi Polusi Cahaya vs Titik Belok Fajar**")
                    scatter_lp = alt.Chart(df_corr).mark_circle(size=70, color='#1D9A9C').encode(
                        x=alt.X('Garis_Dasar:Q', title='Garis Dasar Kecerlangan (Mpsas)', scale=alt.Scale(zero=False)),
                        y=alt.Y('Fajar_Alt:Q', title='Titik Belok Fajar (°)', scale=alt.Scale(domain=[-22, -12])),
                        tooltip=['Kota', 'Tanggal', 'Garis_Dasar', 'Fajar_Alt']
                    ).properties(height=320)
                    reg_lp = scatter_lp.transform_regression('Garis_Dasar', 'Fajar_Alt').mark_line(color='#d9534f', strokeWidth=2)
                    st.altair_chart(scatter_lp + reg_lp, use_container_width=True)

                with col_k2:
                    st.markdown("**2. Korelasi Gangguan Awan Ufuk vs Titik Belok Fajar**")
                    scatter_cloud = alt.Chart(df_corr).mark_circle(size=70, color='#4A90E2').encode(
                        x=alt.X('Awan_%:Q', title='Persentase Awan SQM Ufuk (%)', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('Fajar_Alt:Q', title='Titik Belok Fajar (°)', scale=alt.Scale(domain=[-22, -12])),
                        tooltip=['Kota', 'Tanggal', 'Awan_%', 'Fajar_Alt']
                    ).properties(height=320)
                    reg_cloud = scatter_cloud.transform_regression('Awan_%', 'Fajar_Alt').mark_line(color='#d9534f', strokeWidth=2)
                    st.altair_chart(scatter_cloud + reg_cloud, use_container_width=True)
                
                st.divider()
                st.subheader("📋 Tabel Data Korelasi (Dengan Tautan Interaktif)")
                
                link_g_name = "Link_Grafik" if "Link_Grafik" in df_corr.columns else ("Link Grafik" if "Link Grafik" in df_corr.columns else None)
                link_d_name = "Link_DataMentah" if "Link_DataMentah" in df_corr.columns else ("Link Data Mentah" if "Link Data Mentah" in df_corr.columns else None)
                
                kolom_penting = ['Tanggal', 'Kota', 'Lokasi', 'Garis_Dasar', 'Awan_%', 'Fajar_Alt']
                cfg_korelasi = {}
                if link_g_name: 
                    kolom_penting.append(link_g_name)
                    cfg_korelasi[link_g_name] = st.column_config.LinkColumn("Plot Fajar", display_text="🖼️ Lihat Grafik Plot")
                if link_d_name: 
                    kolom_penting.append(link_d_name)
                    cfg_korelasi[link_d_name] = st.column_config.LinkColumn("Data Mentah", display_text="📁 Buka File Mentah")

                st.dataframe(
                    df_corr[kolom_penting],
                    use_container_width=True,
                    hide_index=True,
                    column_config=cfg_korelasi
                )

        with sub_peta:
            st.subheader("🗺️ Peta Persebaran Stasiun Pengamatan")
            df_peta = df_stat.dropna(subset=['Lintang', 'Bujur']).copy()
            if not df_peta.empty:
                st.map(df_peta.rename(columns={'Lintang': 'lat', 'Bujur': 'lon'})[['lat', 'lon']], zoom=4)
    else: 
        st.info("Basis data masih kosong.")

with tab_algoritma:
    st.header("📖 Metodologi, Landasan Matematis & Spesifikasi Algoritma")
    st.markdown(r"""
    Aplikasi ini dibangun sebagai instrumen riset falakiyah analitis yang menggabungkan astrometri klasik, pemrosesan sinyal digital (*Digital Signal Processing*), dan validasi orientasi ufuk timur.

    ### 1. Ekstraksi Fajar: Metode SIGMAG-STAB & Filter Ufuk Timur
    *Smoothed Gradient - Median Absolute Deviation Stability* (SIGMAG-STAB) dipadukan dengan pembacaan sensor SQM yang terarah presisi menghadap langsung ke ufuk timur tempat fajar menyingsing.

    **A. Penghalusan Derau (Savitzky-Golay Filter)**
    Data mentah dihaluskan menggunakan konvolusi polinomial untuk membuang *noise* frekuensi tinggi tanpa merusak bentuk asli transisi fajar:
    $$ Y_j^* = \frac{1}{N} \sum_{i=-m}^{m} C_i Y_{j+i} $$

    **B. Filter Mendung Berbasis Sensor Ufuk (Jendela Kritis $\pm 15$ Menit)**
    Alih-alih menggunakan estimasi satelit makro yang mengasumsikan tutupan awan dari atas kepala (zenit), aplikasi ini memprioritaskan **Rolling Standard Deviation** dari sensor SQM tepat pada rentang 15 menit sebelum hingga sesudah titik belok fajar. Hal ini memastikan bahwa kejernihan di titik sasaran ufuk timur terekam secara mutlak dan objektif.

    ---

    ### 2. Klasifikasi Polusi Cahaya (Berdasarkan Disertasi Basthoni)
    Aplikasi ini menggunakan **4 Skala Penyederhanaan Polusi Cahaya** (adaptasi 9 Skala Bortle Internasional) yang didasarkan secara empiris pada **Visibilitas Ketampakan Fajar Kadzib (*Zodiacal Light*)**:
    1.  **Tipe 1 ($\ge 21.3$ Mpsas):** Langit Gelap (Fajar Kadzib tampak sangat jelas).
    2.  **Tipe 2 ($20.2 - 21.29$ Mpsas):** Langit Agak Gelap (Pedesaan/Pinggiran).
    3.  **Tipe 3 ($19.1 - 20.19$ Mpsas):** Langit Agak Terang (Suburban).
    4.  **Tipe 4 ($< 19.1$ Mpsas):** Langit Terang/Urban (Pusat kota dengan polusi cahaya tinggi).

    > 🔬 **Filter Kalibrasi Kemenag:** Data ideal yang masuk ke dalam hitungan rata-rata nasional mensyaratkan Garis Dasar $\ge 20.5$ Mpsas, gangguan awan ufuk timur $\le 5\%$, dan bebas dari kontaminasi cahaya bulan.
    """)
