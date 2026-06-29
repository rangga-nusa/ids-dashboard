# ============================================================
# DASHBOARD UTAMA - SISTEM DETEKSI INTRUSI ADAPTIF
# Navigasi sidebar dengan empat halaman:
#   1. Beranda      - status model aktif (dari database)
#   2. Pemantauan   - grafik adaptif vs statis, riwayat retraining
#   3. Pengujian    - upload file CSV, prediksi dengan model sungguhan
#   4. Benchmark    - perbandingan algoritma dan teknik imbalance
#
# Model dilatih dari DATA PENUH CICIDS2017 (dimuat dari database).
#
# Cara menjalankan:
#   pip install streamlit plotly pandas numpy scikit-learn xgboost imbalanced-learn
#   streamlit run dashboard_utama.py
# ============================================================

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from xgboost import XGBClassifier

# ---------- DEFINISI KELAS (wajib untuk memuat pickle) ----------
class XGBPembungkus:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.peta_maju = {}
        self.peta_balik = {}
    def fit(self, X, y):
        kelas_unik = np.unique(y)
        self.peta_maju = {asli: i for i, asli in enumerate(kelas_unik)}
        self.peta_balik = {i: asli for asli, i in self.peta_maju.items()}
        y_petakan = np.array([self.peta_maju[v] for v in y])
        self.model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                   tree_method="hist", n_jobs=-1,
                                   random_state=self.random_state, eval_metric="mlogloss")
        self.model.fit(X, y_petakan)
        return self
    def predict(self, X):
        pred = self.model.predict(X)
        return np.array([self.peta_balik[int(p)] for p in pred])

st.set_page_config(page_title="Sistem Deteksi Intrusi Adaptif",
                   page_icon="🛡️", layout="wide")

NAVY="#0E1B33"; TEAL="#1D9E75"; TEAL_LT="#5DCAA5"; AMBER="#EF9F27"
CORAL="#D85A30"; BLUE="#378ADD"; GRAY="#9DB0CE"
FOLDER_MODEL = "model_tersimpan"
DB = os.path.join(FOLDER_MODEL, "metadata_model.db")

# ============ DATA HASIL EKSPERIMEN (DATA PENUH) ============
F1_ADAPTIF_URUT = [1.0000, 0.6429, 0.5000, 0.4995, 0.7821, 0.7500, 0.6243, 0.4978, 0.4285]
F1_STATIS_URUT  = [0.1951, 0.2490, 0.5000, 0.3333, 0.1432, 0.2456, 0.1573, 0.1370, 0.2454]
F1_ADAPTIF_ACAK = [0.7235, 0.7710, 0.7737, 0.9916, 0.9040, 0.9097, 0.9182, 0.9785, 0.8436]
F1_STATIS_ACAK  = [0.7235, 0.7710, 0.7737, 0.7189, 0.7686, 0.7300, 0.7400, 0.7250, 0.7290]
RINGKASAN = {"berurutan": {"adaptif":0.6361,"statis":0.2449,"selisih":0.3912},
             "acak": {"adaptif":0.8682,"statis":0.7614,"selisih":0.1067}}
RIWAYAT_RETRAIN = pd.DataFrame([
    {"Batch":1,"Pemicu":"drift","F1 Sebelum":0.1951,"F1 Sesudah":1.0000,"Keputusan":"Promote"},
    {"Batch":2,"Pemicu":"drift","F1 Sebelum":0.3966,"F1 Sesudah":0.6429,"Keputusan":"Promote"},
    {"Batch":3,"Pemicu":"performa turun","F1 Sebelum":0.2500,"F1 Sesudah":0.5000,"Keputusan":"Promote"},
    {"Batch":4,"Pemicu":"drift & performa turun","F1 Sebelum":0.1974,"F1 Sesudah":0.4995,"Keputusan":"Promote"},
    {"Batch":5,"Pemicu":"drift","F1 Sebelum":0.2432,"F1 Sesudah":0.7821,"Keputusan":"Promote"},
    {"Batch":6,"Pemicu":"drift & performa turun","F1 Sebelum":0.1231,"F1 Sesudah":0.7500,"Keputusan":"Promote"},
    {"Batch":7,"Pemicu":"drift","F1 Sebelum":0.1769,"F1 Sesudah":0.6243,"Keputusan":"Promote"},
    {"Batch":8,"Pemicu":"drift","F1 Sebelum":0.1791,"F1 Sesudah":0.4978,"Keputusan":"Promote"},
    {"Batch":9,"Pemicu":"drift","F1 Sebelum":0.1381,"F1 Sesudah":0.4285,"Keputusan":"Promote"},
])
BENCH_ALGORITMA = pd.DataFrame([
    {"Algoritma":"XGBoost","F1 Macro":0.9259,"Waktu Latih (detik)":115.2},
    {"Algoritma":"Random Forest","F1 Macro":0.7825,"Waktu Latih (detik)":39.6},
    {"Algoritma":"Decision Tree","F1 Macro":0.7818,"Waktu Latih (detik)":34.2},
])
BENCH_IMBALANCE = pd.DataFrame([
    {"Teknik":"Tanpa Penanganan","F1 Macro":0.7981},
    {"Teknik":"Class Weighting","F1 Macro":0.8619},
    {"Teknik":"SMOTE","F1 Macro":0.9259},
])

# ============ FUNGSI MUAT MODEL ============
@st.cache_resource
def muat_model_aktif():
    if not os.path.exists(DB):
        return None, None
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("""SELECT id_model, versi, tanggal_latih, f1_score, path_file, status
                   FROM model_champion WHERE status='aktif' ORDER BY id_model DESC LIMIT 1""")
    b = cur.fetchone(); conn.close()
    if b is None:
        return None, None
    meta = {"id":b[0],"versi":b[1],"tanggal":b[2],"f1":b[3],"path":b[4],"status":b[5]}
    if not os.path.exists(meta["path"]):
        return None, meta
    with open(meta["path"], "rb") as f:
        return pickle.load(f), meta

@st.cache_data
def ambil_semua_model():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT id_model, versi, tanggal_latih, f1_score, status FROM model_champion ORDER BY id_model DESC", conn)
    conn.close()
    return df

def simpan_riwayat_pengujian(nama_file, jumlah_baris, jumlah_serangan, akurasi):
    """Simpan ringkasan satu pengujian ke tabel riwayat_pengujian."""
    from datetime import datetime
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS riwayat_pengujian (
        id_uji INTEGER PRIMARY KEY AUTOINCREMENT,
        waktu TEXT, nama_file TEXT, jumlah_baris INTEGER,
        jumlah_serangan INTEGER, akurasi REAL)""")
    cur.execute("""INSERT INTO riwayat_pengujian
        (waktu, nama_file, jumlah_baris, jumlah_serangan, akurasi)
        VALUES (?,?,?,?,?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nama_file,
         int(jumlah_baris), int(jumlah_serangan),
         (round(float(akurasi),4) if akurasi is not None else None)))
    conn.commit(); conn.close()

def ambil_riwayat_pengujian():
    if not os.path.exists(DB):
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT id_uji, waktu, nama_file, jumlah_baris, jumlah_serangan, akurasi "
            "FROM riwayat_pengujian ORDER BY id_uji DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

paket, meta = muat_model_aktif()

# ============ SIDEBAR NAVIGASI ============
st.sidebar.markdown(f"""
<div style="background:{NAVY}; padding:14px; border-radius:10px; margin-bottom:14px; text-align:center">
  <span style="color:white; font-size:18px; font-weight:bold">🛡️ IDS Adaptif</span><br>
  <span style="color:{TEAL_LT}; font-size:12px">Continuous Retraining</span>
</div>
""", unsafe_allow_html=True)
halaman = st.sidebar.radio("Navigasi", ["Beranda", "Pemantauan", "Pengujian", "Benchmark"])
st.sidebar.markdown("---")
if meta:
    st.sidebar.caption(f"Model aktif: #{meta['id']} (F1 {meta['f1']:.4f})")
st.sidebar.caption("Dataset: CICIDS2017 (penuh)")

def header(judul, sub):
    st.markdown(f"""
    <div style="background:{NAVY}; padding:16px 22px; border-radius:12px; margin-bottom:16px">
      <h1 style="color:white; margin:0; font-size:24px">{judul}</h1>
      <p style="color:{TEAL_LT}; margin:3px 0 0 0; font-size:14px">{sub}</p>
    </div>""", unsafe_allow_html=True)

# ============ HALAMAN 1: BERANDA ============
if halaman == "Beranda":
    header("🛡️ Dashboard Pemantauan Deteksi Intrusi Adaptif",
           "Sistem klasifikasi serangan dengan continuous retraining — Dataset CICIDS2017 penuh")
    if meta is None or paket is None:
        st.error("Model/database tidak ditemukan. Pastikan folder 'model_tersimpan' ada di samping file ini.")
        st.stop()

    st.subheader("Status Model Champion Aktif")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ID Model", f"#{meta['id']}")
    c2.metric("Versi", meta["versi"])
    c3.metric("F1-Score", f"{meta['f1']:.4f}")
    c4.metric("Status", meta["status"].upper())
    st.caption(f"Model dilatih pada {meta['tanggal']} dari dataset CICIDS2017 penuh.")

    st.markdown("---")
    st.subheader("Ringkasan Hasil Eksperimen (Data Penuh)")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Skenario Berurutan**")
        st.metric("Keunggulan Adaptif", f"+{RINGKASAN['berurutan']['selisih']:.4f}",
                  f"adaptif {RINGKASAN['berurutan']['adaptif']:.4f} vs statis {RINGKASAN['berurutan']['statis']:.4f}")
    with cc2:
        st.markdown("**Skenario Diacak**")
        st.metric("Keunggulan Adaptif", f"+{RINGKASAN['acak']['selisih']:.4f}",
                  f"adaptif {RINGKASAN['acak']['adaptif']:.4f} vs statis {RINGKASAN['acak']['statis']:.4f}")

    with st.expander("Riwayat model di database (champion & arsip)"):
        st.dataframe(ambil_semua_model().style.format({"f1_score":"{:.4f}"}),
                     use_container_width=True, hide_index=True)

# ============ HALAMAN 2: PEMANTAUAN ============
elif halaman == "Pemantauan":
    header("Pemantauan Performa Model", "Perbandingan model adaptif vs statis sepanjang waktu")
    skenario = st.radio("Pilih skenario:", ["Batch Berurutan", "Batch Diacak"], horizontal=True)
    key = "berurutan" if skenario=="Batch Berurutan" else "acak"
    adaptif = F1_ADAPTIF_URUT if key=="berurutan" else F1_ADAPTIF_ACAK
    statis = F1_STATIS_URUT if key=="berurutan" else F1_STATIS_ACAK
    bx = list(range(1, len(adaptif)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bx, y=adaptif, mode="lines+markers", name="Adaptif (retraining)",
                             line=dict(color=BLUE, width=3), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=bx, y=statis, mode="lines+markers", name="Statis (tanpa retraining)",
                             line=dict(color=CORAL, width=3, dash="dash"), marker=dict(size=9, symbol="square")))
    fig.update_layout(xaxis_title="Batch (urutan waktu)", yaxis_title="F1-Score (macro)",
                      yaxis=dict(range=[0,1.05]), height=430, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    if key=="berurutan":
        st.info("Model statis jatuh tajam saat menghadapi jenis serangan baru. Model adaptif pulih lewat retraining berulang.")
    else:
        st.info("Pada skenario diacak, kedua model lebih berdekatan karena tiap batch sudah memuat campuran serangan.")

    st.markdown("---")
    st.subheader("Riwayat Retraining (Champion-Challenger)")
    st.caption("Skenario berurutan. Model hanya diganti bila challenger lebih unggul.")
    st.dataframe(RIWAYAT_RETRAIN.style.format({"F1 Sebelum":"{:.4f}","F1 Sesudah":"{:.4f}"}),
                 use_container_width=True, hide_index=True)

# ============ HALAMAN 3: PENGUJIAN ============
elif halaman == "Pengujian":
    header("Pengujian Model — Unggah Data Jaringan", "Unggah file CSV untuk diprediksi oleh model champion aktif")
    if meta is None or paket is None:
        st.error("Model/database tidak ditemukan.")
        st.stop()
    model = paket["model"]; encoder = paket["encoder"]; fitur = paket["fitur"]

    st.caption(f"Model aktif: #{meta['id']} (F1 {meta['f1']:.4f}, dari data penuh). "
               "Jika file memiliki kolom 'Label', akurasi dihitung otomatis.")
    berkas = st.file_uploader("Unggah file CSV (format CICIDS2017)", type=["csv"])

    if berkas is not None:
        try:
            df = pd.read_csv(berkas, low_memory=False)
            df.columns = df.columns.str.strip()
            st.success(f"File termuat: {len(df):,} baris, {df.shape[1]} kolom.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}"); st.stop()

        punya_label = "Label" in df.columns
        X_input = df.drop(columns=["Label"]) if punya_label else df.copy()
        X_input = X_input.select_dtypes(include=[np.number])
        for kol in fitur:
            if kol not in X_input.columns:
                X_input[kol] = 0
        X_input = X_input[fitur].replace([np.inf,-np.inf], np.nan).fillna(0)

        if st.button("Prediksi Sekarang", type="primary"):
            with st.spinner("Model memprediksi..."):
                nomor = model.predict(X_input)
                nama = encoder.inverse_transform(nomor.astype(int))
            df_hasil = df.copy(); df_hasil["Prediksi"] = nama
            total = len(df_hasil)
            jml_serangan = int((df_hasil["Prediksi"]!="BENIGN").sum())

            st.markdown("---"); st.subheader("Hasil Pengujian")
            k1,k2,k3 = st.columns(3)
            k1.metric("Total Diuji", f"{total:,}")
            k2.metric("Normal", f"{total-jml_serangan:,}")
            k3.metric("Serangan", f"{jml_serangan:,}")
            if punya_label:
                akurasi = (pd.Series(nama).values == df["Label"].astype(str).values).mean()
                st.metric("Akurasi terhadap Label Asli", f"{akurasi*100:.2f}%")
            else:
                akurasi = None

            # Simpan ringkasan pengujian ke database (riwayat operasional)
            try:
                simpan_riwayat_pengujian(berkas.name, total, jml_serangan, akurasi)
                st.success("Ringkasan pengujian tersimpan ke database (riwayat_pengujian).")
            except Exception as e:
                st.warning(f"Hasil tampil, namun gagal menyimpan riwayat: {e}")

            dist = df_hasil["Prediksi"].value_counts()
            figd = go.Figure(go.Bar(x=dist.values, y=dist.index, orientation="h",
                marker_color=[TEAL if k=="BENIGN" else CORAL for k in dist.index],
                text=dist.values, textposition="outside"))
            figd.update_layout(title="Distribusi Hasil Prediksi", height=350,
                               xaxis_title="Jumlah Aliran", margin=dict(t=40,b=20))
            st.plotly_chart(figd, use_container_width=True)

            kk = [c for c in df_hasil.columns if c not in ["Label","Prediksi"]][:4]
            tampil = kk + (["Label","Prediksi"] if punya_label else ["Prediksi"])
            st.subheader("Rincian Prediksi (100 baris pertama)")
            st.dataframe(df_hasil[tampil].head(100), use_container_width=True, hide_index=True)

            csv_unduh = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh Hasil Prediksi (CSV)", data=csv_unduh,
                               file_name="hasil_prediksi.csv", mime="text/csv")
    else:
        st.info("Silakan unggah file CSV untuk memulai pengujian.")

    # ---------- RIWAYAT PENGUJIAN DARI DATABASE ----------
    st.markdown("---")
    st.subheader("Riwayat Pengujian Tersimpan")
    st.caption("Catatan setiap pengujian yang pernah dilakukan, tersimpan di database SQLite.")
    df_riwayat = ambil_riwayat_pengujian()
    if df_riwayat.empty:
        st.info("Belum ada riwayat pengujian. Lakukan pengujian pertama dengan mengunggah file di atas.")
    else:
        tampil_riwayat = df_riwayat.rename(columns={
            "id_uji":"ID", "waktu":"Waktu", "nama_file":"Nama File",
            "jumlah_baris":"Jumlah Baris", "jumlah_serangan":"Serangan Terdeteksi",
            "akurasi":"Akurasi"})
        st.dataframe(
            tampil_riwayat.style.format({"Akurasi":lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "-"}),
            use_container_width=True, hide_index=True)

# ============ HALAMAN 4: BENCHMARK ============
elif halaman == "Benchmark":
    header("Benchmark Perbandingan Metode", "Justifikasi pemilihan algoritma dan teknik penanganan data")
    st.subheader("Perbandingan Algoritma")
    col1, col2 = st.columns([3,2])
    with col1:
        figa = go.Figure(go.Bar(x=BENCH_ALGORITMA["Algoritma"], y=BENCH_ALGORITMA["F1 Macro"],
            marker_color=[TEAL, BLUE, GRAY], text=BENCH_ALGORITMA["F1 Macro"], textposition="outside"))
        figa.update_layout(yaxis=dict(range=[0,1.05], title="F1 Macro"), height=340, margin=dict(t=20,b=20))
        st.plotly_chart(figa, use_container_width=True)
    with col2:
        st.dataframe(BENCH_ALGORITMA.style.format({"F1 Macro":"{:.4f}","Waktu Latih (detik)":"{:.1f}"}),
                     use_container_width=True, hide_index=True)
    st.caption("XGBoost unggul hampir 15 poin F1 atas pesaingnya.")

    st.markdown("---")
    st.subheader("Perbandingan Teknik Penanganan Data Tidak Seimbang")
    col3, col4 = st.columns([3,2])
    with col3:
        figb = go.Figure(go.Bar(x=BENCH_IMBALANCE["Teknik"], y=BENCH_IMBALANCE["F1 Macro"],
            marker_color=[CORAL, AMBER, TEAL], text=BENCH_IMBALANCE["F1 Macro"], textposition="outside"))
        figb.update_layout(yaxis=dict(range=[0,1.05], title="F1 Macro"), height=340, margin=dict(t=20,b=20))
        st.plotly_chart(figb, use_container_width=True)
    with col4:
        st.dataframe(BENCH_IMBALANCE.style.format({"F1 Macro":"{:.4f}"}),
                     use_container_width=True, hide_index=True)
    st.caption("SMOTE memberi hasil terbaik, menaikkan F1 dari 0,7981 menjadi 0,9259.")

st.markdown("---")
st.caption("Sistem Deteksi Intrusi Adaptif Berbasis Machine Learning — Mekanisme Continuous Retraining. Dataset CICIDS2017.")
