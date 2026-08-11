import os
import re
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import json
import time
import urllib.request

# Hindari error jika matplotlib dijalankan tanpa layar (headless server)
import matplotlib
matplotlib.use('Agg')

try:
    import ephem
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ephem"])
    import ephem

import cloudinary
import cloudinary.uploader
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ==========================================
# KONFIGURASI KREDENSIAL DARI GITHUB SECRETS
# ==========================================
GSHEETS_PERMANEN_URL = "https://docs.google.com/spreadsheets/d/1E4RpTfcPeQorW3r9cjpZ5cp31dpa7N_oXRZksRWdxG4/edit?gid=0#gid=0"

def get_gcp_creds():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON tidak ditemukan di Environment Variables!")
    return json.loads(creds_json)

try:
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME").strip() if os.environ.get("CLOUDINARY_CLOUD_NAME") else None,
        api_key=os.environ.get("CLOUDINARY_API_KEY").strip() if os.environ.get("CLOUDINARY_API_KEY") else None,
        api_secret=os.environ.get("CLOUDINARY_API_SECRET").strip() if os.environ.get("CLOUDINARY_API_SECRET") else None,
        secure=True
    )
except Exception as e: 
    print(f"Error Cloudinary Config: {e}")

# ==========================================
# FUNGSI UPLOAD & SINKRONISASI
# ==========================================
def upload_plot_to_cloudinary(fig, filename):
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, format="png", bbox_inches="tight", dpi=100)
            tmp_path = tmp.name
        response = cloudinary.uploader.upload(tmp_path, folder="kawakib_arsip", public_id=filename.replace(".png", ""))
        os.remove(tmp_path)
        return response.get("secure_url")
    except Exception as e: 
        print(f"Gagal upload plot: {e}")
        return ""

def upload_raw_to_cloudinary(file_path, filename):
    try:
        response = cloudinary.uploader.upload(file_path, resource_type="raw", folder="kawakib_raw_data", public_id=filename)
        return response.get("secure_url")
    except Exception as e: 
        print(f"Gagal upload raw: {e}")
        return ""

def get_gsheets_client():
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(get_gcp_creds(), scopes=scopes)
    return gspread.authorize(creds)

def save_to_google_sheets(data_dict):
    try:
        client = get_gsheets_client()
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
            
        idx_tgl = headers.index("Tanggal") if "Tanggal" in headers else -1
        idx_lok = headers.index("Lokasi") if "Lokasi" in headers else -1
        idx_met = headers.index("Metode") if "Metode" in headers else -1
        
        row_to_update = None
        if idx_tgl != -1 and idx_lok != -1 and idx_met != -1:
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
            print(f"Update baris duplikat di GSheets untuk {data_dict['Lokasi']} tgl {data_dict['Tanggal']}")
        else: 
            sheet.append_row(values)
            print(f"Menambahkan baris baru di GSheets untuk {data_dict['Lokasi']} tgl {data_dict['Tanggal']}")
        return True
    except Exception as e:
        print(f"Error saving to sheets: {e}")
        return False

def load_data_from_google_sheets():
    try:
        client = get_gsheets_client()
        sheet = client.open_by_url(GSHEETS_PERMANEN_URL).sheet1
        return pd.DataFrame(sheet.get_all_records())
    except: return pd.DataFrame()

def sync_from_soof_drive():
    print("Memulai koneksi ke Google Drive (Mode Robot)...")
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_info(get_gcp_creds(), scopes=scopes)
    service = build('drive', 'v3', credentials=creds)
    downloaded_paths = []
    
    try:
        page_token = None
        while True:
            results = service.files().list(
                q="mimeType != 'application/vnd.google-apps.folder' and (name contains '.dat' or name contains '.DAT' or name contains '.txt') and trashed = false",
                fields="nextPageToken, files(id, name)",
                pageSize=1000,
                pageToken=page_token
            ).execute()
            
            files = results.get('files', [])
            for file in files:
                file_path = os.path.join(tempfile.gettempdir(), file['name'])
                request = service.files().get_media(fileId=file['id'])
                with open(file_path, "wb") as f:
                    f.write(request.execute())
                downloaded_paths.append(file_path)
                
            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break
        print(f"Berhasil menemukan & mendownload {len(downloaded_paths)} file data dari Drive.")
    except Exception as e:
        print(f"Error saat menarik file dari Drive: {e}")
        
    return downloaded_paths

# =====================================================================
# INTEGRASI API SATELIT CUACA (OPEN-METEO)
# =====================================================================
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

# ==========================================
# FUNGSI ASTRONOMI 
# ==========================================
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

# ==========================================
# EKSEKUSI UTAMA (MAIN LOOP)
# ==========================================
if __name__ == "__main__":
    method = "SIGMAG-STAB" 
    file_paths = sync_from_soof_drive()
    
    if not file_paths:
        print("Selesai: Tidak ada file yang ditemukan.")
    else:
        df_existing = load_data_from_google_sheets()
        
        for idx, path in enumerate(file_paths):
            print(f"[{idx+1}/{len(file_paths)}] Memproses: {os.path.basename(path)}")
            try:
                am, site, lat, lon, utc_offset, date_str = load_sqm_data(path)
                if am.empty: 
                    continue
                
                am, is_corrected = apply_moonlight_correction(am, lat, lon, utc_offset)
                bin_deg, n_consec = get_dynamic_params(am)
                
                if method == "SIGMAG-STAB":
                    onset_alt, onset_msas = analyze_sigmag(am, bin_deg, n_consec)
                else:
                    onset_alt, onset_msas = analyze_sigmoid(am)
                
                cloud_pct, df_win = analyze_cloud_cover(am, onset_alt)
                sat_cloud_pct = get_satellite_cloud_cover(lat, lon, date_str) # CEK KE SATELIT
                
                base_series = am[am["sun_alt"] < -20]["mpsas_corrected"]
                baseline_mpsas = base_series.median() if not base_series.empty else am["mpsas_corrected"].max()
                lp_category, _ = categorize_light_pollution(baseline_mpsas)
                kota = re.split(r'[-,\|]', site)[-1].strip()
                
                plot_url, raw_url = "", ""
                if not df_existing.empty and {"Tanggal", "Lokasi", "Metode"}.issubset(df_existing.columns):
                    match = df_existing[(df_existing["Tanggal"].astype(str).str.strip() == str(date_str)) & (df_existing["Lokasi"].astype(str).str.strip() == str(site)) & (df_existing["Metode"].astype(str).str.strip() == str(method))]
                    if not match.empty:
                        plot_url = str(match.iloc[0].get("Link_Grafik", "")).strip()
                        raw_url = str(match.iloc[0].get("Link_DataMentah", "")).strip()

                if not plot_url or not raw_url:
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
                    
                    if not plot_url:
                        plot_url = upload_plot_to_cloudinary(fig, f"Plot_{site}_{date_str}_{method}".replace(" ", "_"))
                    if not raw_url:
                        raw_url = upload_raw_to_cloudinary(path, f"Raw_{site}_{date_str}_{method}.dat".replace(" ", "_"))
                    plt.close(fig)
                
                # Simpan Hasilnya ke Google Sheets
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
                
                # JEDA NAPAS ROBOT AGAR TIDAK KENA ERROR 429 GOOGLE SHEETS
                print("--> Istirahat 3 detik agar tidak diblokir Google...")
                time.sleep(3) 
                
            except Exception as e:
                print(f"--> [GAGAL] Error saat memproses {os.path.basename(path)}: {e}")
                time.sleep(3) 

    print("=========================================")
    print("SINKRONISASI OTOMATIS SELESAI DENGAN SUKSES!")
    print("=========================================")
