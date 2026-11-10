import gc
import io
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

# --- INTEGRASI DATABASE CLOUD: SQL (Supabase) dan NoSQL (Firebase RTDB) ---
try:
    from supabase import create_client
except Exception:
    create_client = None
try:
    import firebase_admin
    from firebase_admin import credentials, db
except Exception:
    firebase_admin = None
try:
    import xlsxwriter  # noqa: F401

    ADA_XLSX = True
except Exception:
    ADA_XLSX = False

# ============================================================================
# KONFIGURASI
# ============================================================================
MODEL_PATH = "model_cicids_1juta.pkl"
CONTOH_PATH = "contoh_data.csv"  # file data contoh (opsional) di folder yang sama
LABEL_COL = "Attack Type"
NORMAL = "Normal Traffic"
MAX_LAPORAN_LENGKAP = 200_000  # di atas ini, CSV hanya berisi kolom hasil
MAX_SERANGAN_LENGKAP = 50_000  # batas baris daftar serangan dengan semua kolom
MAX_BARIS_EXCEL = 20_000  # batas baris pada sheet Detail_Serangan
MAX_SQL_LENGKAP = 50_000  # di atas ini, tabel SQL hanya berisi kolom hasil
MAX_SQL_DUMP = 100_000  # batas baris agar file .sql (teks) tidak terlalu besar
MAX_BARIS_KUERI = 1_000  # baris maksimum yang ditampilkan dari satu kueri
AMBANG_KEYAKINAN = 0.70  # keyakinan di bawah ini ditandai "perlu ditinjau"
WIB = timezone(timedelta(hours=7))

WARNA = {
    "Normal Traffic": "#2ed573",
    "DoS": "#ff4757",
    "DDoS": "#c0392b",
    "Port Scanning": "#ffa502",
    "Brute Force": "#8e44ad",
    "Web Attacks": "#1e90ff",
    "Bots": "#576574",
}
SKALA_WARNA = alt.Scale(domain=list(WARNA.keys()), range=list(WARNA.values()))

# Hasil evaluasi pada data uji (20% = 504.151 baris) saat pelatihan di Google Colab
METRIK_MODEL = pd.DataFrame(
    {
        "Kelas": [
            "Bots",
            "Brute Force",
            "DDoS",
            "DoS",
            "Normal Traffic",
            "Port Scanning",
            "Web Attacks",
        ],
        "Presisi": [0.89, 1.00, 1.00, 1.00, 1.00, 0.99, 0.99],
        "Recall": [0.47, 1.00, 1.00, 1.00, 1.00, 1.00, 0.96],
        "F1-score": [0.62, 1.00, 1.00, 1.00, 1.00, 0.99, 0.97],
        "Jumlah Data Uji": [389, 1830, 25603, 38749, 419012, 18139, 429],
    }
)

# ============================================================================
# INISIALISASI DATABASE CLOUD (SUPABASE)
# ============================================================================
# 1. Supabase (SQL)
supabase_client = None
supabase_galat = None
try:
    if create_client is None:
        raise RuntimeError("pustaka supabase belum ada di requirements.txt")
    supabase_client = create_client(
        st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
    )
except Exception as e:
    supabase_client = None
    supabase_galat = f"{type(e).__name__}: {e}"

# 2. Firebase Realtime Database (NoSQL)
firebase_ok = False
firebase_galat = None
try:
    if firebase_admin is None:
        raise RuntimeError("pustaka firebase-admin belum ada di requirements.txt")
    if not firebase_admin._apps:
        if "FIREBASE_JSON" in st.secrets:
            info_kunci = json.loads(st.secrets["FIREBASE_JSON"])
        elif "FIREBASE_CREDENTIALS" in st.secrets:
            info_kunci = dict(st.secrets["FIREBASE_CREDENTIALS"])
        else:
            raise RuntimeError("FIREBASE_JSON belum diisi di Secrets")
        firebase_admin.initialize_app(
            credentials.Certificate(info_kunci),
            {"databaseURL": st.secrets["FIREBASE_URL"]},
        )
    firebase_ok = True
except Exception as e:
    firebase_ok = False
    firebase_galat = f"{type(e).__name__}: {e}"

# ============================================================================
# TAMPILAN
# ============================================================================
st.set_page_config(
    page_title="NIDS Security Operations Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, .stApp, [data-testid="stSidebar"] {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    .stApp { background: #f4f6fb; }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px; }
    header[data-testid="stHeader"] { background: transparent; }
    h2, h3 { color: #1e272e; font-weight: 700; letter-spacing: -0.01em; }

    /* ---------- Banner utama ---------- */
    .hero {
        background: linear-gradient(120deg, #0f1b4d 0%, #1e3799 55%, #4a69bd 100%);
        border-radius: 22px;
        padding: 34px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #ffffff;
        box-shadow: 0 14px 34px rgba(30, 55, 153, 0.25);
        margin-bottom: 22px;
        overflow: hidden;
    }
    .hero .kicker {
        font-size: 0.72rem; letter-spacing: 0.2em; font-weight: 700;
        color: #ffd166; margin-bottom: 8px;
    }
    .hero h1 {
        color: #ffffff; font-size: 2.4rem; font-weight: 800;
        margin: 0 0 8px 0; padding: 0; letter-spacing: -0.02em; line-height: 1.1;
    }
    .hero p { color: #dfe6ff; max-width: 640px; margin: 0 0 16px 0; font-size: 1rem; line-height: 1.55; }
    .chip {
        display: inline-block; background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.28); color: #ffffff;
        padding: 5px 13px; border-radius: 999px; font-size: 0.8rem;
        margin: 0 8px 6px 0;
    }
    .hero-art { flex: 0 0 auto; margin-left: 24px; opacity: 0.95; }
    @media (max-width: 900px) { .hero-art { display: none; } .hero { padding: 26px 24px; } }

    /* ---------- Kartu angka ---------- */
    .kartu {
        background: #ffffff; border: 1px solid #e6eaf2; border-radius: 16px;
        padding: 14px 16px; display: flex; gap: 12px; align-items: center;
        box-shadow: 0 4px 14px rgba(15, 27, 77, 0.06);
        height: 118px; box-sizing: border-box; overflow: hidden;
        container-type: inline-size;
    }
    .kartu .isi { flex: 1 1 auto; min-width: 0; }
    .kartu .ikon {
        width: 44px; height: 44px; border-radius: 12px; flex: 0 0 auto;
        display: flex; align-items: center; justify-content: center; font-size: 20px;
    }
    .kartu .cat { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .kartu .lbl {
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
        overflow: hidden;
    }
    @container (max-width: 150px) {
        .kartu .ikon { display: none; }
        .kartu .val { font-size: clamp(0.95rem, calc(100cqw / 6), 1.65rem); }
    }
    .kartu .lbl { color: #636e72; font-size: 0.78rem; font-weight: 600; line-height: 1.3; }
    .kartu .val {
        color: #1e272e; font-weight: 800; line-height: 1.2; white-space: nowrap;
        font-size: clamp(1rem, calc((100cqw - 70px) / 5.6), 1.65rem);
    }
    .kartu .cat { color: #8395a7; font-size: 0.74rem; line-height: 1.3; }
    .kartu.netral .ikon { background: #e8edff; }
    .kartu.normal .ikon { background: #e3f9ec; }
    .kartu.bahaya .ikon { background: #ffe5e0; }
    .kartu.waspada .ikon { background: #fff1d6; }
    .kartu.normal .val { color: #12a150; }
    .kartu.bahaya .val { color: #e84118; }
    .kartu.waspada .val { color: #c27c0e; }

    /* ---------- Meter risiko ---------- */
    .meter {
        background: #ffffff; border: 1px solid #e6eaf2; border-radius: 16px;
        padding: 16px 20px 12px 20px; margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(15, 27, 77, 0.06);
    }
    .meter-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }
    .pill { padding: 5px 14px; border-radius: 999px; color: #ffffff; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.05em; }
    .pill.aman { background: #2ed573; }
    .pill.sedang { background: #ffa502; }
    .pill.kritis { background: #ff4757; }
    .meter-val { color: #636e72; font-size: 0.9rem; font-weight: 500; }
    .meter-bar {
        height: 12px; border-radius: 999px; overflow: hidden;
        background: linear-gradient(90deg, #ffe7bf 0%, #ffe7bf 20%, #ffd0d0 20%, #ffd0d0 100%);
    }
    .meter-fill { height: 100%; border-radius: 999px; }
    .meter-fill.aman { background: #2ed573; }
    .meter-fill.sedang { background: #ffa502; }
    .meter-fill.kritis { background: #ff4757; }
    .meter-zona { position: relative; height: 20px; margin-top: 6px; }
    .meter-zona span { position: absolute; font-size: 0.72rem; color: #8395a7; }

    /* ---------- Wawasan ---------- */
    .insight {
        background: #ffffff; border: 1px solid #e6eaf2; border-radius: 16px;
        padding: 14px 18px; display: flex; gap: 12px; align-items: center;
        box-shadow: 0 4px 14px rgba(15, 27, 77, 0.06); margin: 14px 0 10px 0;
    }
    .insight .i { font-size: 22px; }
    .insight .t { color: #636e72; font-size: 0.78rem; font-weight: 600; }
    .insight .d { color: #1e272e; font-size: 1rem; font-weight: 700; }

    /* ---------- Langkah (halaman awal) ---------- */
    .langkah-wrap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 6px 0 18px 0; }
    @media (max-width: 900px) { .langkah-wrap { grid-template-columns: 1fr; } }
    .langkah {
        background: #ffffff; border: 1px solid #e6eaf2; border-radius: 16px;
        padding: 20px; box-shadow: 0 4px 14px rgba(15, 27, 77, 0.06);
    }
    .langkah .no {
        width: 34px; height: 34px; border-radius: 50%; background: #1e3799; color: #fff;
        display: flex; align-items: center; justify-content: center; font-weight: 800; margin-bottom: 12px;
    }
    .langkah b { color: #1e272e; font-size: 1.02rem; display: block; margin-bottom: 4px; }
    .langkah span { color: #636e72; font-size: 0.88rem; line-height: 1.5; }

    /* ---------- Tab, panel, sidebar, tombol ---------- */
    .stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: #ffffff; padding: 6px; border-radius: 14px;
        border: 1px solid #e6eaf2 !important; box-shadow: 0 4px 14px rgba(15, 27, 77, 0.05);
    }
    .stTabs [role="tab"] { height: 42px; border-radius: 10px; padding: 0 18px; font-weight: 600; }
    .stTabs [role="tab"][aria-selected="true"] { background: #1e3799; }
    .stTabs [role="tab"][aria-selected="true"] * { color: #ffffff !important; }
    .stTabs .react-aria-SelectionIndicator,
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-testid="stTabPanel"] { padding-top: 18px; }
    [class*="st-key-panel"] {
        background: #ffffff; border-radius: 16px !important; border: 1px solid #e6eaf2 !important;
        box-shadow: 0 4px 14px rgba(15, 27, 77, 0.06);
    }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6eaf2; }
    .stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)

HERO_SVG = (
    '<svg width="230" height="150" viewBox="0 0 230 150" xmlns="http://www.w3.org/2000/svg">'
    '<g stroke="rgba(255,255,255,0.38)" stroke-width="1.6">'
    '<line x1="28" y1="38" x2="92" y2="22"/><line x1="92" y1="22" x2="128" y2="74"/>'
    '<line x1="28" y1="38" x2="128" y2="74"/><line x1="128" y1="74" x2="196" y2="40"/>'
    '<line x1="128" y1="74" x2="168" y2="122"/><line x1="128" y1="74" x2="62" y2="118"/>'
    '<line x1="196" y1="40" x2="168" y2="122"/><line x1="62" y1="118" x2="168" y2="122"/>'
    "</g>"
    '<circle cx="28" cy="38" r="8" fill="#8fd3ff"/><circle cx="92" cy="22" r="6" fill="#bde0fe"/>'
    '<circle cx="196" cy="40" r="9" fill="#8fd3ff"/><circle cx="62" cy="118" r="7" fill="#bde0fe"/>'
    '<circle cx="168" cy="122" r="10" fill="#8fd3ff"/>'
    '<circle cx="128" cy="74" r="20" fill="#ffd166" opacity="0.25"/>'
    '<circle cx="128" cy="74" r="13" fill="#ffd166"/>'
    "</svg>"
)

st.markdown(
    '<div class="hero"><div class="hero-text">'
    '<div class="kicker">NETWORK INTRUSION DETECTION SYSTEM</div>'
    "<h1>NIDS Security Center</h1>"
    "<p>Deteksi dan klasifikasi serangan jaringan berbasis <b>Random Forest</b> pada"
    " dataset <b>CICIDS2017</b>. Upload data trafik, lalu dapatkan ringkasan risiko,"
    " grafik, dan laporan dalam hitungan detik.</p>"
    '<span class="chip">🌲 Random Forest</span><span class="chip">🗂️ CICIDS2017</span>'
    '<span class="chip">🎯 Akurasi uji 99,87%</span><span class="chip">⚡ Analisis data batch</span>'
    '</div><div class="hero-art">' + HERO_SVG + "</div></div>",
    unsafe_allow_html=True,
)


# ============================================================================
# MODEL
# ============================================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_status = "Aktif & Stabil (Akurasi uji: 99.87%)"
except Exception as e:
    model = None
    model_status = f"Error: {e}"


# ============================================================================
# FUNGSI BANTU: GRAFIK
# ============================================================================
def grafik_batang(counts, judul, horizontal=False):
    data = counts.rename_axis("Kategori").reset_index(name="Jumlah")
    if horizontal:
        return (
            alt.Chart(data, title=judul)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("Jumlah:Q", title="Jumlah baris"),
                y=alt.Y("Kategori:N", sort="-x", title=None),
                color=alt.Color("Kategori:N", scale=SKALA_WARNA, legend=None),
                tooltip=["Kategori:N", alt.Tooltip("Jumlah:Q", format=",")],
            )
            .properties(height=320)
        )
    return (
        alt.Chart(data, title=judul)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "Kategori:N",
                sort="-y",
                axis=alt.Axis(labelAngle=-30, title=None),
            ),
            y=alt.Y("Jumlah:Q", title="Jumlah baris"),
            color=alt.Color("Kategori:N", scale=SKALA_WARNA, legend=None),
            tooltip=["Kategori:N", alt.Tooltip("Jumlah:Q", format=",")],
        )
        .properties(height=320)
    )


def grafik_f1():
    data = METRIK_MODEL[["Kelas", "F1-score"]]
    return (
        alt.Chart(data, title="F1-score per kelas (data uji)")
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("F1-score:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Kelas:N", sort="-x", title=None),
            color=alt.Color("Kelas:N", scale=SKALA_WARNA, legend=None),
            tooltip=["Kelas:N", alt.Tooltip("F1-score:Q", format=".2f")],
        )
        .properties(height=280)
    )


def grafik_fitur(fi):
    data = fi.rename_axis("Fitur").reset_index(name="Kepentingan")
    return (
        alt.Chart(data, title="10 fitur paling berpengaruh")
        .mark_bar(color="#1e3799", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("Kepentingan:Q"),
            y=alt.Y("Fitur:N", sort="-x", title=None),
            tooltip=["Fitur:N", alt.Tooltip("Kepentingan:Q", format=".3f")],
        )
        .properties(height=320)
    )


def grafik_cm(cm):
    data = cm.reset_index().melt(
        id_vars="Label_Asli", var_name="Prediksi", value_name="Jumlah"
    )
    batas = float(data["Jumlah"].max()) * 0.3
    dasar = alt.Chart(data).encode(
        x=alt.X("Prediksi:N", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Label_Asli:N", title="Label asli"),
    )
    heat = dasar.mark_rect().encode(
        color=alt.Color(
            "Jumlah:Q", scale=alt.Scale(type="symlog", scheme="blues"), legend=None
        ),
        tooltip=["Label_Asli:N", "Prediksi:N", alt.Tooltip("Jumlah:Q", format=",")],
    )
    teks = dasar.mark_text(fontSize=12).encode(
        text=alt.Text("Jumlah:Q", format=","),
        color=alt.condition(
            alt.datum.Jumlah > batas, alt.value("white"), alt.value("black")
        ),
    )
    return (heat + teks).properties(height=380)


# ============================================================================
# FUNGSI BANTU: KOMPONEN TAMPILAN
# ============================================================================
def fmt_id(n):
    """Format angka gaya Indonesia: 1.234.567"""
    return f"{int(n):,}".replace(",", ".")


def kartu(ikon, label, nilai, tone, catatan=""):
    return (
        f'<div class="kartu {tone}"><div class="ikon">{ikon}</div><div class="isi">'
        f'<div class="lbl">{label}</div><div class="val">{nilai}</div>'
        f'<div class="cat">{catatan}</div></div></div>'
    )


def insight(ikon, judul, isi):
    return (
        f'<div class="insight"><div class="i">{ikon}</div><div>'
        f'<div class="t">{judul}</div><div class="d">{isi}</div></div></div>'
    )


def meter_risiko(pct):
    if pct == 0:
        tone, teks = "aman", "STATUS: AMAN"
    elif pct < 20:
        tone, teks = "sedang", "RISIKO: SEDANG"
    else:
        tone, teks = "kritis", "RISIKO: KRITIS"
    lebar = min(max(pct, 1.5), 100.0)
    return (
        f'<div class="meter"><div class="meter-top"><span class="pill {tone}">{teks}</span>'
        f'<span class="meter-val">{pct:.2f}% dari trafik terdeteksi sebagai serangan</span></div>'
        f'<div class="meter-bar"><div class="meter-fill {tone}" style="width:{lebar}%"></div></div>'
        '<div class="meter-zona"><span style="left:0">Sedang (&lt; 20%)</span>'
        '<span style="left:20%">Kritis (&ge; 20%)</span></div></div>'
    )


def rapikan(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#eef1f6",
            domain=False,
            tickColor="#d5dbe5",
            labelColor="#57606f",
            titleColor="#57606f",
            labelFontSize=12,
        )
        .configure_title(
            anchor="start", fontSize=15, fontWeight=600, color="#1e272e", offset=12
        )
    )


def tampil(chart, stretch=True):
    st.altair_chart(rapikan(chart), width="stretch" if stretch else "content")


def grafik_donat(n_normal, n_serangan):
    total = n_normal + n_serangan
    pct = (n_serangan / total * 100) if total else 0
    data = pd.DataFrame(
        {"Jenis": ["Normal", "Serangan"], "Jumlah": [n_normal, n_serangan]}
    )
    arc = (
        alt.Chart(data)
        .mark_arc(innerRadius=48, outerRadius=76, cornerRadius=4)
        .encode(
            theta=alt.Theta("Jumlah:Q"),
            color=alt.Color(
                "Jenis:N",
                scale=alt.Scale(
                    domain=["Normal", "Serangan"], range=["#2ed573", "#ff4757"]
                ),
                legend=None,
            ),
            tooltip=["Jenis:N", alt.Tooltip("Jumlah:Q", format=",")],
        )
    )
    teks = (
        alt.Chart(pd.DataFrame({"t": [f"{pct:.1f}%"]}))
        .mark_text(fontSize=22, fontWeight="bold", color="#1e272e")
        .encode(text="t:N", x=alt.value(90), y=alt.value(90))
    )
    return (arc + teks).properties(width=180, height=180)


# ============================================================================
# FUNGSI BANTU: LAPORAN
# ============================================================================
def ke_csv(df_sumber, kolom_lengkap, kolom_ringkas, batas):
    """CSV lengkap jika baris <= batas, selain itu hanya kolom ringkas."""
    if len(df_sumber) <= batas:
        data = df_sumber.to_csv(columns=kolom_lengkap, index_label="no_baris")
        return data.encode("utf-8"), False
    data = df_sumber.to_csv(columns=kolom_ringkas, index_label="no_baris")
    return data.encode("utf-8"), True


def buat_excel(
    out, counts, total, n_serangan, n_ditinjau, nama, evaluasi, kolom_depan
):
    """Laporan Excel multi-sheet. Mengembalikan bytes, atau None jika gagal."""
    if not ADA_XLSX:
        return None
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            wb = writer.book
            f_judul = wb.add_format(
                {"bold": True, "font_size": 16, "font_color": "#1e3799"}
            )
            f_label = wb.add_format(
                {"bold": True, "bg_color": "#dfe6e9", "border": 1}
            )
            f_nilai = wb.add_format({"border": 1})
            f_head = wb.add_format(
                {
                    "bold": True,
                    "bg_color": "#1e3799",
                    "font_color": "#ffffff",
                    "border": 1,
                }
            )
            f_warn = wb.add_format({"bg_color": "#ffeaa7"})
            f_catatan = wb.add_format({"italic": True, "font_color": "#636e72"})

            # ---------------- Sheet Ringkasan ----------------
            ws = wb.add_worksheet("Ringkasan")
            writer.sheets["Ringkasan"] = ws
            ws.set_column(0, 0, 44)
            ws.set_column(1, 3, 18)
            ws.write(0, 0, "Laporan Deteksi Intrusi Jaringan (NIDS)", f_judul)
            ws.write(
                1,
                0,
                f"Catatan: sheet Detail_Serangan memuat maksimal {MAX_BARIS_EXCEL:,}"
                " baris serangan pertama.",
                f_catatan,
            )
            info = [
                ("Sumber data", str(nama)),
                (
                    "Tanggal analisis",
                    datetime.now(WIB).strftime("%d-%m-%Y %H:%M WIB"),
                ),
                ("Model", "Random Forest (dataset CICIDS2017)"),
                ("Total data dianalisis", int(total)),
                ("Trafik normal", int(total - n_serangan)),
                ("Terdeteksi serangan", int(n_serangan)),
                (
                    "Persentase serangan (%)",
                    round(n_serangan / total * 100, 3) if total else 0,
                ),
                (
                    f"Baris keyakinan < {int(AMBANG_KEYAKINAN * 100)}% (perlu ditinjau)",
                    int(n_ditinjau),
                ),
            ]
            if evaluasi is not None:
                info.append(
                    (
                        "Akurasi pada data ini (%)",
                        round(evaluasi["akurasi"] * 100, 2),
                    )
                )
            for i, (k, v) in enumerate(info):
                ws.write(3 + i, 0, k, f_label)
                ws.write(3 + i, 1, v, f_nilai)

            r0 = 3 + len(info) + 2
            ws.write(r0 - 1, 0, "Distribusi seluruh kategori", f_label)
            for j, h in enumerate(["Kategori", "Jumlah", "Persen (%)"]):
                ws.write(r0, j, h, f_head)
            for i, (kat, jml) in enumerate(counts.items()):
                ws.write(r0 + 1 + i, 0, str(kat), f_nilai)
                ws.write(r0 + 1 + i, 1, int(jml), f_nilai)
                ws.write(
                    r0 + 1 + i, 2, round(int(jml) / total * 100, 3), f_nilai
                )

            serangan = counts.drop(NORMAL, errors="ignore")
            if len(serangan) > 0:
                r1 = r0 + len(counts) + 4
                ws.write(
                    r1 - 1, 0, "Distribusi jenis serangan (tanpa Normal)", f_label
                )
                for j, h in enumerate(["Jenis serangan", "Jumlah"]):
                    ws.write(r1, j, h, f_head)
                for i, (kat, jml) in enumerate(serangan.items()):
                    ws.write(r1 + 1 + i, 0, str(kat), f_nilai)
                    ws.write(r1 + 1 + i, 1, int(jml), f_nilai)
                chart = wb.add_chart({"type": "column"})
                chart.add_series(
                    {
                        "name": "Jumlah serangan",
                        "categories": [
                            "Ringkasan",
                            r1 + 1,
                            0,
                            r1 + len(serangan),
                            0,
                        ],
                        "values": [
                            "Ringkasan",
                            r1 + 1,
                            1,
                            r1 + len(serangan),
                            1,
                        ],
                        "data_labels": {"value": True},
                        "fill": {"color": "#ff4757"},
                    }
                )
                chart.set_title({"name": "Distribusi Jenis Serangan"})
                chart.set_legend({"none": True})
                ws.insert_chart(3, 4, chart, {"x_scale": 1.5, "y_scale": 1.4})

            # ---------------- Sheet Detail_Serangan ----------------
            mask = out["Hasil_Prediksi"] != NORMAL
            opsional = [
                c
                for c in [
                    "Destination Port",
                    "Flow Duration",
                    "Total Fwd Packets",
                    "Flow Bytes/s",
                    "Flow Packets/s",
                ]
                if c in out.columns
            ]
            det = (
                out.loc[mask, kolom_depan + opsional]
                .head(MAX_BARIS_EXCEL)
                .copy()
            )
            for c in ["Hasil_Prediksi", "Label_Asli"]:
                if c in det.columns:
                    det[c] = det[c].astype(str)
            det["Perlu_Ditinjau"] = np.where(det["Perlu_Ditinjau"], "Ya", "Tidak")
            for c in opsional:
                det[c] = det[c].astype("float64").round(4)
            det.to_excel(
                writer, sheet_name="Detail_Serangan", index_label="no_baris"
            )
            ws2 = writer.sheets["Detail_Serangan"]
            ws2.freeze_panes(1, 0)
            ws2.autofilter(0, 0, len(det), len(det.columns))
            ws2.set_column(0, len(det.columns), 18)
            kol = list(det.columns).index("Perlu_Ditinjau") + 1
            ws2.conditional_format(
                1,
                kol,
                max(len(det), 1),
                kol,
                {
                    "type": "cell",
                    "criteria": "==",
                    "value": '"Ya"',
                    "format": f_warn,
                },
            )

            # ---------------- Sheet Evaluasi (jika ada label) ----------------
            if evaluasi is not None:
                evaluasi["tabel"].to_excel(
                    writer, sheet_name="Evaluasi", index_label="Kelas"
                )
                evaluasi["cm"].to_excel(writer, sheet_name="Confusion_Matrix")
                writer.sheets["Evaluasi"].set_column(0, 5, 18)
                writer.sheets["Confusion_Matrix"].set_column(0, 10, 16)
        return buf.getvalue()
    except Exception:
        return None


# ============================================================================
# FUNGSI BANTU: DATABASE CLOUD (SUPABASE)
# ============================================================================
MAX_LOG_SUPABASE = 1000  # baris pertama yang disimpan sebagai log per analisis


def simpan_ke_supabase(h):
    """Simpan satu analisis: 1 baris ringkasan (analisis_run) + sampel baris log.

    Mengembalikan (run_id, jumlah_baris_log).
    """
    if supabase_client is None:
        raise RuntimeError("Supabase belum terhubung.")
    ev = h["evaluasi"]
    run = (
        supabase_client.table("analisis_run")
        .insert(
            {
                "sumber_data": str(h["nama"]),
                "waktu_analisis": h["waktu"],
                "total_baris": int(h["total"]),
                "jumlah_serangan": int(h["n_serangan"]),
                "perlu_ditinjau": int(h["n_ditinjau"]),
                "akurasi": float(ev["akurasi"]) if ev else None,
                "distribusi": {str(k): int(v) for k, v in h["counts"].items()},
            }
        )
        .execute()
    )
    run_id = int(run.data[0]["id"])
    try:
        sampel = h["out"].head(MAX_LOG_SUPABASE)
        baris = [
            {
                "run_id": run_id,
                "no_baris": int(n) + 1,
                "Hasil_Prediksi": str(p),
                "Keyakinan_Persen": float(k),
                "Perlu_Ditinjau": bool(t),
            }
            for n, p, k, t in zip(
                sampel.index.tolist(),
                sampel["Hasil_Prediksi"].astype(str).tolist(),
                sampel["Keyakinan_%"].astype(float).tolist(),
                sampel["Perlu_Ditinjau"].astype(bool).tolist(),
            )
        ]
        for a in range(0, len(baris), 500):
            supabase_client.table("log_serangan").insert(baris[a : a + 500]).execute()
    except Exception:
        # batalkan ringkasan agar tidak ada riwayat yang setengah jadi
        try:
            supabase_client.table("analisis_run").delete().eq("id", run_id).execute()
        except Exception:
            pass
        raise
    return run_id, len(baris)


def baca_riwayat(batas=50):
    r = (
        supabase_client.table("analisis_run")
        .select("*")
        .order("id", desc=True)
        .limit(batas)
        .execute()
    )
    return pd.DataFrame(r.data)


def baca_log(run_id, batas=100):
    r = (
        supabase_client.table("log_serangan")
        .select("no_baris,Hasil_Prediksi,Keyakinan_Persen,Perlu_Ditinjau")
        .eq("run_id", int(run_id))
        .order("no_baris")
        .limit(batas)
        .execute()
    )
    return pd.DataFrame(r.data)


# ============================================================================
# FUNGSI BANTU: DATABASE NoSQL (Firebase Realtime Database)
# ============================================================================
# Struktur pohon JSON di Realtime Database:
#   /analisis/<id_unik>        -> ringkasan satu analisis (dokumen fleksibel)
#   /katalog_serangan/<kelas>  -> data menu "Info Serangan" yang ditampilkan app
KATALOG_SERANGAN = {
    "Normal Traffic": {
        "tingkat": "Aman",
        "deskripsi": "Lalu lintas jaringan yang wajar, seperti membuka web, email, dan transfer file sehari-hari.",
        "saran": "Tetap dipantau dan gunakan sebagai pembanding (baseline) untuk mengenali perilaku yang tidak biasa.",
    },
    "DoS": {
        "tingkat": "Tinggi",
        "deskripsi": "Serangan membanjiri satu layanan dari satu sumber sampai tidak mampu melayani pengguna yang sah (contoh di CICIDS2017: Hulk, GoldenEye, Slowloris).",
        "saran": "Batasi laju permintaan (rate limiting), blokir IP sumber, dan pasang firewall di depan layanan.",
    },
    "DDoS": {
        "tingkat": "Kritis",
        "deskripsi": "Serangan DoS yang dilancarkan dari banyak sumber sekaligus (biasanya botnet), sehingga sulit diblokir per alamat IP.",
        "saran": "Gunakan layanan anti-DDoS atau CDN, filter di sisi hulu (ISP), dan siapkan kapasitas cadangan.",
    },
    "Port Scanning": {
        "tingkat": "Sedang",
        "deskripsi": "Pemindaian port untuk mencari layanan yang terbuka. Biasanya tahap pengintaian sebelum serangan yang lebih serius.",
        "saran": "Tutup port yang tidak dipakai, aktifkan firewall dan alert IDS, serta batasi koneksi baru per sumber.",
    },
    "Brute Force": {
        "tingkat": "Tinggi",
        "deskripsi": "Percobaan login berulang untuk menebak kata sandi (contoh di CICIDS2017: FTP-Patator dan SSH-Patator).",
        "saran": "Batasi percobaan login, pasang fail2ban, aktifkan autentikasi dua faktor, dan wajibkan kata sandi yang kuat.",
    },
    "Web Attacks": {
        "tingkat": "Tinggi",
        "deskripsi": "Serangan ke aplikasi web, seperti brute force login web, XSS, dan SQL injection.",
        "saran": "Validasi input, gunakan parameterized query, pasang WAF, dan selalu perbarui aplikasi.",
    },
    "Bots": {
        "tingkat": "Tinggi",
        "deskripsi": "Perangkat yang terinfeksi malware dan dikendalikan penyerang (botnet), berkomunikasi dengan server komando.",
        "saran": "Isolasi perangkat yang terinfeksi, blokir alamat server komando, jalankan antivirus/EDR, dan ganti kredensial.",
    },
}


def simpan_ke_firebase(h, sql_run_id=None):
    """Simpan ringkasan satu analisis sebagai dokumen JSON. Mengembalikan kunci uniknya."""
    if not firebase_ok:
        raise RuntimeError("Firebase belum terhubung.")
    ev = h["evaluasi"]
    data = {
        "sumber_data": str(h["nama"]),
        "waktu_analisis": h["waktu"],
        "dibuat_utc": datetime.now(timezone.utc).isoformat(),
        "total_baris": int(h["total"]),
        "jumlah_serangan": int(h["n_serangan"]),
        "perlu_ditinjau": int(h["n_ditinjau"]),
        "distribusi": {str(k): int(v) for k, v in h["counts"].items()},
    }
    if ev:
        data["akurasi"] = float(ev["akurasi"])
    if sql_run_id is not None:
        data["sql_run_id"] = int(sql_run_id)
    return db.reference("analisis").push(data).key


def baca_riwayat_firebase(batas=50):
    snap = db.reference("analisis").order_by_key().limit_to_last(batas).get()
    if not snap:
        return pd.DataFrame()
    baris = []
    for kunci, nilai in snap.items():
        d = dict(nilai)
        d["kunci"] = kunci
        baris.append(d)
    df = pd.DataFrame(baris).sort_values("kunci", ascending=False)
    return df.reset_index(drop=True)


def baca_katalog():
    return db.reference("katalog_serangan").get() or {}


def isi_katalog():
    db.reference("katalog_serangan").set(KATALOG_SERANGAN)


# ============================================================================
# FUNGSI BANTU: DATABASE SQL (SQLite Lokal)
# ============================================================================
KOLOM_HASIL = ["Hasil_Prediksi", "Keyakinan_%", "Perlu_Ditinjau", "Label_Asli"]

CONTOH_KUERI = {
    "Jumlah baris per kategori": (
        "SELECT Hasil_Prediksi AS kategori, COUNT(*) AS jumlah\n"
        "FROM hasil_prediksi\nGROUP BY Hasil_Prediksi\nORDER BY jumlah DESC"
    ),
    "Rata-rata keyakinan per kategori": (
        "SELECT Hasil_Prediksi AS kategori,\n"
        "       ROUND(AVG(Keyakinan_persen), 2) AS rata_rata_keyakinan\n"
        "FROM hasil_prediksi\nGROUP BY Hasil_Prediksi\nORDER BY"
        " rata_rata_keyakinan"
    ),
    "Serangan dengan keyakinan rendah": (
        "SELECT no_baris, Hasil_Prediksi, Keyakinan_persen\n"
        "FROM hasil_prediksi\n"
        "WHERE Hasil_Prediksi <> 'Normal Traffic' AND Perlu_Ditinjau = 1\n"
        "ORDER BY Keyakinan_persen ASC\nLIMIT 100"
    ),
    "Ringkasan analisis": "SELECT * FROM info_analisis",
    "Tabel ringkasan kategori": "SELECT * FROM ringkasan_kategori",
}


def _nama_sql(nama):
    nama = str(nama).replace("%", "persen")
    nama = re.sub(r"[^0-9A-Za-z]+", "_", nama).strip("_")
    return nama or "kolom"


def _nama_sql_unik(daftar):
    terpakai, hasil = {}, []
    for nama in map(_nama_sql, daftar):
        terpakai[nama] = terpakai.get(nama, 0) + 1
        hasil.append(
            nama if terpakai[nama] == 1 else f"{nama}_{terpakai[nama]}"
        )
    return hasil


def buat_database_sql(h):
    """Bangun database SQLite dari hasil analisis."""
    out = h["out"]
    if len(out) <= MAX_SQL_LENGKAP:
        kolom = [k for k in h["kolom_urut"] if k in out.columns]
    else:
        kolom = [k for k in KOLOM_HASIL if k in out.columns]
    df = out[kolom].copy()
    for k in df.columns:
        if str(df[k].dtype) == "category":
            df[k] = df[k].astype(str)
    if "Perlu_Ditinjau" in df.columns:
        df["Perlu_Ditinjau"] = df["Perlu_Ditinjau"].astype(int)
    df.columns = _nama_sql_unik(df.columns)

    counts = h["counts"]
    ringkasan = pd.DataFrame(
        {
            "kategori": counts.index.astype(str),
            "jumlah": counts.values.astype(int),
            "persen": (counts.values / max(h["total"], 1) * 100).round(3),
        }
    )
    info = pd.DataFrame(
        {
            "sumber_data": [h["nama"]],
            "waktu_analisis": [h["waktu"]],
            "total_baris": [int(h["total"])],
            "jumlah_serangan": [int(h["n_serangan"])],
            "perlu_ditinjau": [int(h["n_ditinjau"])],
            "akurasi": [h["evaluasi"]["akurasi"] if h["evaluasi"] else None],
            "kolom_lengkap": [int(len(out) <= MAX_SQL_LENGKAP)],
        }
    )

    fd, path = tempfile.mkstemp(suffix=".db", prefix="nids_")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        df.to_sql(
            "hasil_prediksi",
            conn,
            index=True,
            index_label="no_baris",
            chunksize=50_000,
        )
        ringkasan.to_sql("ringkasan_kategori", conn, index=False)
        info.to_sql("info_analisis", conn, index=False)
        if h["evaluasi"]:
            ev = h["evaluasi"]["tabel"].reset_index()
            ev.columns = _nama_sql_unik(["kelas"] + list(ev.columns[1:]))
            ev.to_sql("evaluasi_kelas", conn, index=False)
        conn.execute(
            "CREATE INDEX idx_hasil_prediksi ON hasil_prediksi(Hasil_Prediksi)"
        )
        conn.commit()
        dump = None
        if len(df) <= MAX_SQL_DUMP:
            dump = "\n".join(conn.iterdump()) + "\n"
    finally:
        conn.close()
    with open(path, "rb") as f:
        data_db = f.read()
    return {
        "path": path,
        "db": data_db,
        "dump": dump.encode("utf-8") if dump else None,
        "baris": int(len(df)),
        "lengkap": len(out) <= MAX_SQL_LENGKAP,
    }


def jalankan_kueri(path, sql):
    """Jalankan satu kueri baca-saja. Mengembalikan (DataFrame, terpotong)."""
    teks = sql.strip().rstrip(";").strip()
    if not teks.lower().startswith(("select", "with")):
        raise ValueError(
            "Hanya kueri SELECT (atau WITH ... SELECT) yang diizinkan."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(teks)
        kolom = [d[0] for d in cur.description]
        baris = cur.fetchmany(MAX_BARIS_KUERI + 1)
    finally:
        conn.close()
    terpotong = len(baris) > MAX_BARIS_KUERI
    return pd.DataFrame(baris[:MAX_BARIS_KUERI], columns=kolom), terpotong


# ============================================================================
# PROSES ANALISIS
# ============================================================================
def analisis(sumber, nama):
    """Baca file SEKALI, prediksi, simpan ke database, dan simpan hasil ke session_state."""
    st.session_state.pop("hasil", None)
    gc.collect()

    if hasattr(sumber, "seek"):
        sumber.seek(0)
    df = pd.read_csv(sumber)
    df.columns = df.columns.str.strip()

    label_asli = (
        df[LABEL_COL].astype(str).to_numpy() if LABEL_COL in df.columns else None
    )

    fitur = list(getattr(model, "feature_names_in_", []))
    if fitur:
        hilang = [c for c in fitur if c not in df.columns]
        if hilang:
            st.error(
                "Kolom fitur berikut tidak ditemukan di file: "
                + ", ".join(hilang[:5])
                + (" ..." if len(hilang) > 5 else "")
            )
            return
        X = df[fitur].astype("float32")
    else:
        X = df.select_dtypes(include=[np.number]).astype("float32")
        fitur = list(X.columns)
    del df

    if not np.isfinite(X.to_numpy()).all():
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Satu kali prediksi: kelas + tingkat keyakinan
    proba = model.predict_proba(X)
    keyakinan = proba.max(axis=1)
    preds = np.asarray(model.classes_)[proba.argmax(axis=1)].astype(str)
    del proba

    out = X
    out["Hasil_Prediksi"] = pd.Categorical(preds)
    out["Keyakinan_%"] = np.round(keyakinan * 100, 2)
    out["Perlu_Ditinjau"] = keyakinan < AMBANG_KEYAKINAN
    kolom_depan = ["Hasil_Prediksi", "Keyakinan_%", "Perlu_Ditinjau"]
    if label_asli is not None:
        out["Label_Asli"] = pd.Categorical(label_asli)
        kolom_depan.append("Label_Asli")
    kolom_urut = kolom_depan + fitur

    counts = pd.Series(preds).value_counts()
    total = len(preds)
    mask = preds != NORMAL
    n_serangan = int(mask.sum())
    n_ditinjau = int((keyakinan < AMBANG_KEYAKINAN).sum())

    # Wawasan otomatis
    wawasan = {}
    if n_serangan > 0:
        serangan = counts.drop(NORMAL, errors="ignore")
        wawasan["jenis"] = str(serangan.index[0])
        wawasan["jenis_jumlah"] = int(serangan.iloc[0])
        if "Destination Port" in out.columns:
            port = out.loc[mask, "Destination Port"].round().astype("int64")
            wawasan["port"] = {
                int(k): int(v) for k, v in port.value_counts().head(3).items()
            }

    # Evaluasi langsung (hanya jika file memiliki label)
    evaluasi = None
    if label_asli is not None:
        rap = classification_report(
            label_asli, preds, output_dict=True, zero_division=0
        )
        rap.pop("accuracy", None)
        tabel = pd.DataFrame(rap).T.rename(
            columns={
                "precision": "Presisi",
                "recall": "Recall",
                "f1-score": "F1-score",
                "support": "Jumlah",
            }
        )
        tabel["Jumlah"] = tabel["Jumlah"].astype(int)
        cm = pd.crosstab(
            pd.Series(label_asli, name="Label_Asli"),
            pd.Series(preds, name="Prediksi"),
        )
        evaluasi = {
            "akurasi": float(accuracy_score(label_asli, preds)),
            "tabel": tabel.round(4),
            "cm": cm,
        }

    # Laporan
    csv_utama, ringkas = ke_csv(
        out, kolom_urut, kolom_depan, MAX_LAPORAN_LENGKAP
    )
    if ringkas:
        csv_label, csv_nama = "⬇️ Hasil Prediksi (.CSV)", "hasil_prediksi_nids.csv"
    else:
        csv_label, csv_nama = "⬇️ Laporan Lengkap (.CSV)", "laporan_lengkap_nids.csv"

    sub = out.loc[mask]
    csv_serangan, _ = ke_csv(sub, kolom_urut, kolom_depan, MAX_SERANGAN_LENGKAP)
    del sub

    excel = buat_excel(
        out, counts, total, n_serangan, n_ditinjau, nama, evaluasi, kolom_depan
    )


    st.session_state.pop("sql", None)
    st.session_state.pop("run_tersimpan", None)
    st.session_state["hasil"] = {
        "out": out,
        "nama": nama,
        "waktu": datetime.now(WIB).strftime("%d-%m-%Y %H:%M WIB"),
        "counts": counts,
        "total": total,
        "n_serangan": n_serangan,
        "n_ditinjau": n_ditinjau,
        "wawasan": wawasan,
        "evaluasi": evaluasi,
        "kolom_urut": kolom_urut,
        "csv_utama": csv_utama,
        "csv_label": csv_label,
        "csv_nama": csv_nama,
        "ringkas": ringkas,
        "csv_serangan": csv_serangan,
        "excel": excel,
    }
    gc.collect()


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Kontrol Panel")
st.sidebar.info(
    f"**Status Model:**\nAlgoritma: Random Forest\nStatus: {model_status}"
)

if supabase_client is None:
    st.sidebar.warning(
        f"Supabase (SQL) belum terhubung. Penyebab: {supabase_galat}"
    )
else:
    st.sidebar.success("Supabase (SQL) terhubung")
if not firebase_ok:
    st.sidebar.warning(
        f"Firebase (NoSQL) belum terhubung. Penyebab: {firebase_galat}"
    )
else:
    st.sidebar.success("Firebase Realtime DB (NoSQL) terhubung")

st.sidebar.header("📂 Sumber Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV data jaringan", type=["csv"]
)
klik_unggah = False
if uploaded_file is not None:
    klik_unggah = st.sidebar.button(
        "🚀 Jalankan Analisis & Deteksi", type="primary"
    )

klik_contoh = False
if os.path.exists(CONTOH_PATH):
    st.sidebar.markdown("atau")
    klik_contoh = st.sidebar.button("🧪 Pakai data contoh")

st.sidebar.caption(
    "Prototipe analisis data batch: memproses file CSV yang diunggah,"
    " bukan memantau trafik secara langsung."
)

if (klik_unggah or klik_contoh) and model is None:
    st.error("Model gagal dimuat, analisis tidak dapat dijalankan.")
elif klik_unggah or klik_contoh:
    with st.spinner("Model sedang memproses inferensi data..."):
        try:
            if klik_unggah:
                analisis(uploaded_file, uploaded_file.name)
            else:
                analisis(CONTOH_PATH, "Data contoh (contoh_data.csv)")
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")
    if "hasil" in st.session_state:
        st.toast("✅ Analisis keamanan selesai!")

hasil = st.session_state.get("hasil")

# ============================================================================
# KONTEN UTAMA (TAB)
# ============================================================================
tab_hasil, tab_eval, tab_model, tab_laporan, tab_sql, tab_riwayat, tab_info = st.tabs(
    [
        "📊 Hasil Analisis",
        "🎯 Evaluasi Langsung",
        "🧠 Performa Model",
        "📥 Laporan",
        "🗄️ Database",
        "🕘 Riwayat",
        "📚 Info Serangan",
    ]
)

# ---------------------------------------------------------------- Hasil Analisis
with tab_hasil:
    if hasil is None:
        st.markdown(
            '<div class="langkah-wrap">'
            '<div class="langkah"><div class="no">1</div><b>Upload data</b>'
            "<span>Pilih file CSV trafik jaringan (fitur CICIDS2017) lewat panel"
            " di kiri, atau pakai data contoh.</span></div>"
            '<div class="langkah"><div class="no">2</div><b>Jalankan analisis</b>'
            "<span>Model Random Forest mengklasifikasikan tiap baris menjadi"
            " trafik normal atau jenis serangan tertentu.</span></div>"
            '<div class="langkah"><div class="no">3</div><b>Lihat dan unduh'
            " hasil</b><span>Pantau ringkasan risiko, grafik, evaluasi, lalu"
            " unduh laporan Excel atau CSV.</span></div></div>",
            unsafe_allow_html=True,
        )
        st.info(
            "👈 Mulai dari panel kiri: upload CSV lalu klik **Jalankan Analisis &"
            " Deteksi**, atau coba **Pakai data contoh**."
        )
    else:
        out = hasil["out"]
        counts = hasil["counts"]
        total = hasil["total"]
        n_serangan = hasil["n_serangan"]
        n_ditinjau = hasil["n_ditinjau"]
        wawasan = hasil["wawasan"]
        kolom_urut = hasil["kolom_urut"]
        persen_serangan = (n_serangan / total) * 100 if total > 0 else 0

        st.caption(
            f"Sumber data: **{hasil['nama']}** · dianalisis {hasil['waktu']}"
        )
        st.markdown(meter_risiko(persen_serangan), unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            kartu(
                "📁",
                "Total data dianalisis",
                fmt_id(total),
                "netral",
                "baris trafik",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            kartu(
                "🟢",
                "Trafik normal",
                fmt_id(total - n_serangan),
                "normal",
                f"{100 - persen_serangan:.2f}% dari total",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            kartu(
                "🔴",
                "Serangan terdeteksi",
                fmt_id(n_serangan),
                "bahaya",
                f"{persen_serangan:.2f}% dari total",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            kartu(
                "🟡",
                f"Keyakinan < {int(AMBANG_KEYAKINAN * 100)}%",
                fmt_id(n_ditinjau),
                "waspada",
                "perlu ditinjau manual",
            ),
            unsafe_allow_html=True,
        )

        if "jenis" in wawasan:
            w1, w2 = st.columns(2)
            pj = wawasan["jenis_jumlah"] / total * 100
            w1.markdown(
                insight(
                    "🔎",
                    "Serangan terbanyak",
                    f"{wawasan['jenis']} · {fmt_id(wawasan['jenis_jumlah'])} baris"
                    f" ({pj:.2f}%)",
                ),
                unsafe_allow_html=True,
            )
            if "port" in wawasan:
                daftar = ", ".join(
                    f"{p} ({fmt_id(n)})" for p, n in wawasan["port"].items()
                )
                w2.markdown(
                    insight("🌐", "Port tujuan tersering pada serangan", daftar),
                    unsafe_allow_html=True,
                )

        st.subheader("📈 Komposisi dan Jenis Serangan")
        with st.container(border=True, key="panel_grafik"):
            g1, g2 = st.columns([2, 3])
            with g1:
                st.markdown("**Komposisi trafik**")
                tampil(
                    grafik_donat(total - n_serangan, n_serangan), stretch=False
                )
                st.caption("🟢 Normal  ·  🔴 Serangan")
            with g2:
                serangan_saja = counts.drop(NORMAL, errors="ignore")
                if len(serangan_saja) > 0:
                    tampil(
                        grafik_batang(
                            serangan_saja,
                            "Jenis serangan terdeteksi (tanpa Normal)",
                        )
                    )
                else:
                    st.success("Tidak ada serangan terdeteksi pada data ini.")

        st.subheader("🗂️ Seluruh Kategori Trafik")
        with st.container(border=True, key="panel_rekap"):
            r1, r2 = st.columns([3, 2])
            with r1:
                tampil(
                    grafik_batang(
                        counts, "Jumlah baris per kategori", horizontal=True
                    )
                )
            with r2:
                rekap = counts.rename_axis("Kategori").reset_index(
                    name="Jumlah"
                )
                rekap["Persen (%)"] = (rekap["Jumlah"] / total * 100).round(3)
                st.dataframe(rekap, hide_index=True, width="stretch")

        st.subheader("🔍 Filter & Eksplorasi Data")
        with st.container(border=True, key="panel_filter"):
            f1, f2 = st.columns([2, 1])
            with f1:
                pilihan = st.selectbox(
                    "Pilih kategori trafik yang ingin ditampilkan:",
                    ["Semua Kategori"]
                    + sorted(
                        str(x) for x in out["Hasil_Prediksi"].cat.categories
                    ),
                )
            with f2:
                st.write("")
                hanya_ragu = st.checkbox(
                    f"Hanya keyakinan < {int(AMBANG_KEYAKINAN * 100)}%"
                )

            df_f = out
            if pilihan != "Semua Kategori":
                df_f = df_f[df_f["Hasil_Prediksi"] == pilihan]
            if hanya_ragu:
                df_f = df_f[df_f["Perlu_Ditinjau"]]

            st.dataframe(df_f.head(100)[kolom_urut], width="stretch")
            st.caption(
                f"Menampilkan 100 baris pertama dari {len(df_f):,} baris yang"
                " sesuai filter."
            )

# ------------------------------------------------------------- Evaluasi Langsung
with tab_eval:
    st.subheader("🎯 Evaluasi Langsung terhadap Label Asli")
    if hasil is None:
        st.info("Belum ada hasil analisis. Jalankan analisis terlebih dahulu.")
    elif hasil["evaluasi"] is None:
        st.info(
            "File ini tidak memiliki kolom **Attack Type**, sehingga evaluasi"
            " langsung tidak dapat dihitung. Upload file yang masih berlabel"
            " (atau pakai data contoh) untuk mengaktifkan fitur ini."
        )
    else:
        ev = hasil["evaluasi"]
        e1, e2, e3 = st.columns(3)
        try:
            macro = float(ev["tabel"].loc["macro avg", "F1-score"])
        except Exception:
            macro = float("nan")
        e1.markdown(
            kartu(
                "🎯",
                "Akurasi pada file ini",
                f"{ev['akurasi'] * 100:.2f}%".replace(".", ","),
                "netral",
                "dihitung dari label asli",
            ),
            unsafe_allow_html=True,
        )
        e2.markdown(
            kartu(
                "📐",
                "Macro F1-score",
                f"{macro:.3f}".replace(".", ","),
                "netral",
                "rata-rata antar kelas",
            ),
            unsafe_allow_html=True,
        )
        e3.markdown(
            kartu(
                "🗃️",
                "Total baris dievaluasi",
                fmt_id(total),
                "netral",
                "baris berlabel",
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**Presisi, recall, dan F1-score per kelas**")
        st.dataframe(ev["tabel"].round(3), width="stretch")
        st.markdown(
            "**Confusion matrix** (baris = label asli, kolom = prediksi model)"
        )
        tampil(grafik_cm(ev["cm"]))
        st.caption(
            "Catatan: sebagian baris mungkin sudah dipakai saat pelatihan,"
            " sehingga angka ini bersifat demonstrasi. Bukti akurasi utama ada"
            " di tab **Performa Model** (data uji 20% yang tidak dipakai"
            " melatih)."
        )

# ---------------------------------------------------------------- Performa Model
with tab_model:
    st.subheader("🧠 Performa Model pada Data Uji")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        kartu("🎯", "Akurasi", "99,87%", "normal", "pada data uji"),
        unsafe_allow_html=True,
    )
    m2.markdown(
        kartu(
            "📐",
            "Macro F1-score",
            "0,94",
            "netral",
            "rata-rata antar kelas",
        ),
        unsafe_allow_html=True,
    )
    m3.markdown(
        kartu(
            "🗃️",
            "Total data latih + uji",
            "2.520.751",
            "netral",
            "baris",
        ),
        unsafe_allow_html=True,
    )
    m4.markdown(
        kartu(
            "🧪",
            "Data uji (20%)",
            "504.151",
            "netral",
            "tidak dipakai melatih",
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    t1, t2 = st.columns([3, 2])
    with t1:
        st.dataframe(
            METRIK_MODEL,
            hide_index=True,
            width="stretch",
            column_config={
                "Presisi": st.column_config.NumberColumn(format="%.2f"),
                "Recall": st.column_config.NumberColumn(format="%.2f"),
                "F1-score": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    with t2:
        tampil(grafik_f1())

    st.warning(
        "⚠️ Kelas **Bots** paling sulit dikenali (recall 0,47) karena sangat"
        " sedikit: hanya 1.948 baris (sekitar 0,08%) dari seluruh data. Akurasi"
        " keseluruhan tinggi terutama karena Normal Traffic mendominasi"
        " (sekitar 83%), sehingga F1-score per kelas lebih jujur untuk menilai"
        " model."
    )

    st.markdown("---")
    st.subheader("🔬 Fitur yang Paling Menentukan Klasifikasi")
    if (
        model is not None
        and hasattr(model, "feature_importances_")
        and hasattr(model, "feature_names_in_")
    ):
        fi = (
            pd.Series(
                model.feature_importances_, index=model.feature_names_in_
            )
            .sort_values(ascending=False)
            .head(10)
        )
        tampil(grafik_fitur(fi))
        st.caption(
            "Semakin tinggi nilainya, semakin sering fitur tersebut dipakai model"
            " untuk membedakan trafik normal dari tiap jenis serangan."
        )
    else:
        st.info("Informasi fitur penting tidak tersedia pada model ini.")

# --------------------------------------------------------------------- Laporan
with tab_laporan:
    st.subheader("📥 Ekspor Laporan")
    if hasil is None:
        st.info("Belum ada hasil analisis. Jalankan analisis terlebih dahulu.")
    else:
        d1, d2, d3 = st.columns(3)
        with d1:
            with st.container(border=True, key="panel_dl_csv"):
                st.markdown("#### 📄 Data + Prediksi")
                st.caption(
                    "Seluruh data beserta kolom prediksi, keyakinan, dan"
                    " penanda tinjau."
                )
                st.download_button(
                    hasil["csv_label"],
                    data=hasil["csv_utama"],
                    file_name=hasil["csv_nama"],
                    mime="text/csv",
                    key="dl_csv",
                    width="stretch",
                )
        with d2:
            with st.container(border=True, key="panel_dl_xlsx"):
                st.markdown("#### 📗 Laporan Excel")
                st.caption(
                    "Ringkasan bergrafik, detail serangan, dan evaluasi (jika"
                    " berlabel)."
                )
                if hasil["excel"] is not None:
                    st.download_button(
                        "Unduh laporan (.xlsx)",
                        data=hasil["excel"],
                        file_name="laporan_nids.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_xlsx",
                        width="stretch",
                    )
                else:
                    st.caption("Laporan Excel tidak tersedia.")
        with d3:
            with st.container(border=True, key="panel_dl_serangan"):
                st.markdown("#### 🚨 Daftar Serangan")
                st.caption(
                    "Hanya baris yang diprediksi sebagai serangan, siap"
                    " ditindaklanjuti."
                )
                st.download_button(
                    "Unduh daftar serangan (.csv)",
                    data=hasil["csv_serangan"],
                    file_name="daftar_serangan_nids.csv",
                    mime="text/csv",
                    key="dl_serangan",
                    width="stretch",
                )
        if hasil["ringkas"]:
            st.caption(
                f"Data lebih dari {MAX_LAPORAN_LENGKAP:,} baris, sehingga file"
                " CSV hanya berisi kolom hasil (prediksi, keyakinan, dan"
                " penanda tinjau)."
            )
        st.caption(
            "Laporan Excel berisi sheet Ringkasan (dengan grafik),"
            " Detail_Serangan, dan, jika file berlabel, sheet Evaluasi serta"
            " Confusion_Matrix."
        )


# ------------------------------------------------------------------ Database SQL
with tab_sql:
    st.subheader("🗄️ Database (SQL & NoSQL)")
    st.markdown("#### ☁️ Simpan ke database (SQL dan NoSQL)")
    if supabase_client is None and not firebase_ok:
        st.info("Belum ada database yang terhubung. Lihat status di sidebar.")
    elif hasil is None:
        st.caption("Jalankan analisis dulu, lalu simpan hasilnya ke database di sini.")
    else:
        keadaan = st.session_state.setdefault("run_tersimpan", {"sql": None, "fb": None})
        perlu_sql = supabase_client is not None and keadaan["sql"] is None
        perlu_fb = firebase_ok and keadaan["fb"] is None
        if perlu_sql or perlu_fb:
            tujuan = []
            if perlu_sql:
                tujuan.append(
                    f"**SQL (Supabase)**: ringkasan + {MAX_LOG_SUPABASE:,} baris pertama sebagai log"
                )
            if perlu_fb:
                tujuan.append("**NoSQL (Firebase Realtime DB)**: ringkasan analisis")
            st.caption("Akan disimpan ke: " + " · ".join(tujuan))
            if st.button(
                "💾 Simpan hasil ini ke database",
                type="primary",
                key="btn_simpan_db",
            ):
                galat = []
                if perlu_sql:
                    try:
                        with st.spinner("Menyimpan ke Supabase (SQL)..."):
                            rid, n_log = simpan_ke_supabase(hasil)
                        keadaan["sql"] = {"id": rid, "n": n_log}
                    except Exception as e:
                        galat.append(f"SQL (Supabase): {e}")
                if perlu_fb:
                    try:
                        with st.spinner("Menyimpan ke Firebase (NoSQL)..."):
                            kunci = simpan_ke_firebase(
                                hasil, keadaan["sql"]["id"] if keadaan["sql"] else None
                            )
                        keadaan["fb"] = {"kunci": kunci}
                    except Exception as e:
                        galat.append(f"NoSQL (Firebase): {e}")
                if galat:
                    for g in galat:
                        st.error(f"Gagal menyimpan ke {g}")
                else:
                    st.rerun()
        if keadaan["sql"]:
            st.success(
                f"SQL: tersimpan sebagai **analisis #{keadaan['sql']['id']}**"
                f" ({keadaan['sql']['n']:,} baris log)."
            )
        if keadaan["fb"]:
            st.success("NoSQL: ringkasan tersimpan di Firebase Realtime Database.")
        if keadaan["sql"] or keadaan["fb"]:
            st.caption("Lihat hasilnya di tab Riwayat.")
    st.divider()
    st.markdown("#### 💾 Ekspor file database (lokal)")
    if hasil is None:
        st.info("Belum ada hasil analisis. Jalankan analisis terlebih dahulu.")
    else:
        st.caption(
            "Simpan hasil prediksi sebagai database SQLite (.db) lokal, atau"
            " jalankan kueri SQL langsung."
        )
        if "sql" not in st.session_state:
            if st.button(
                "🛠️ Buat database SQL dari hasil ini",
                type="primary",
                key="btn_sql",
            ):
                with st.spinner("Membuat database..."):
                    st.session_state["sql"] = buat_database_sql(hasil)
                st.rerun()
        sql_state = st.session_state.get("sql")
        if sql_state is not None and not os.path.exists(sql_state["path"]):
            st.session_state.pop("sql", None)
            sql_state = None
            st.warning("Database sementara hilang. Silakan buat ulang.")
        if sql_state is not None:
            q1, q2 = st.columns(2)
            with q1:
                with st.container(border=True, key="panel_dl_db"):
                    st.markdown("#### 🗃️ File Database (.db)")
                    st.caption("Bisa dibuka di DB Browser for SQLite atau DBeaver.")
                    st.download_button(
                        "Unduh database (.db)",
                        data=sql_state["db"],
                        file_name="nids_hasil.db",
                        mime="application/x-sqlite3",
                        key="dl_db",
                        width="stretch",
                    )
            with q2:
                with st.container(border=True, key="panel_dl_sqlscript"):
                    st.markdown("#### 📜 Skrip SQL (.sql)")
                    st.caption(
                        "Berisi CREATE TABLE + INSERT, bisa diimpor ke"
                        " MySQL/PostgreSQL."
                    )
                    if sql_state["dump"] is not None:
                        st.download_button(
                            "Unduh skrip (.sql)",
                            data=sql_state["dump"],
                            file_name="nids_hasil.sql",
                            mime="application/sql",
                            key="dl_sqlscript",
                            width="stretch",
                        )
                    else:
                        st.caption(
                            f"Tidak tersedia: data lebih dari {MAX_SQL_DUMP:,}"
                            " baris. Gunakan file .db."
                        )
            if not sql_state["lengkap"]:
                st.caption(
                    f"Data lebih dari {MAX_SQL_LENGKAP:,} baris, sehingga"
                    " tabel hasil_prediksi hanya berisi kolom hasil (prediksi,"
                    " keyakinan, penanda tinjau, label asli)."
                )
            st.markdown(
                "**Tabel:** `hasil_prediksi`, `ringkasan_kategori`,"
                " `info_analisis`"
                + (", `evaluasi_kelas`" if hasil["evaluasi"] else "")
            )

            st.markdown("##### ▶️ Jalankan kueri SQL")
            pilihan = st.selectbox(
                "Contoh kueri", list(CONTOH_KUERI), key="contoh_kueri"
            )
            teks_sql = st.text_area(
                "Kueri (hanya SELECT)",
                value=CONTOH_KUERI[pilihan],
                height=140,
                key=f"teks_sql_{pilihan}",
            )
            if st.button("Jalankan", key="btn_jalankan_sql"):
                try:
                    df_kueri, terpotong = jalankan_kueri(
                        sql_state["path"], teks_sql
                    )
                except Exception as e:
                    st.error(f"Kueri gagal: {e}")
                else:
                    st.dataframe(df_kueri, hide_index=True, width="stretch")
                    st.caption(
                        f"{len(df_kueri):,} baris ditampilkan"
                        + (
                            f" (dibatasi {MAX_BARIS_KUERI:,} baris pertama)."
                            if terpotong
                            else "."
                        )
                    )


# ---------------------------------------------------------------------- Riwayat
def _angka(x):
    return 0 if pd.isna(x) else int(x)


with tab_riwayat:
    st.subheader("🕘 Riwayat Analisis")
    dari_fb = firebase_ok
    if not firebase_ok and supabase_client is None:
        st.info("Belum ada database yang terhubung. Lihat status di sidebar.")
    else:
        st.caption(
            "Sumber data: "
            + (
                "**Firebase Realtime Database (NoSQL)**"
                if dari_fb
                else "**Supabase (SQL)**"
            )
        )
        try:
            riwayat = baca_riwayat_firebase() if dari_fb else baca_riwayat()
        except Exception as e:
            riwayat = None
            st.error(f"Gagal membaca riwayat: {e}")
        if riwayat is not None and riwayat.empty:
            st.info(
                "Belum ada analisis yang tersimpan. Jalankan analisis, lalu klik"
                " **Simpan hasil ini ke database** di tab Database."
            )
        elif riwayat is not None:
            nama_kol = {
                "sumber_data": "Sumber data",
                "waktu_analisis": "Waktu",
                "total_baris": "Total baris",
                "jumlah_serangan": "Serangan",
                "perlu_ditinjau": "Perlu ditinjau",
                "akurasi": "Akurasi",
            }
            if dari_fb:
                nama_kol["sql_run_id"] = "ID di SQL"
            else:
                nama_kol = {"id": "ID", **nama_kol}
            kol = [c for c in nama_kol if c in riwayat.columns]
            tabel_r = riwayat[kol].rename(columns=nama_kol)
            if "Akurasi" in tabel_r.columns:
                tabel_r["Akurasi"] = tabel_r["Akurasi"].map(
                    lambda v: "-" if pd.isna(v) else f"{v * 100:.2f}%".replace(".", ",")
                )
            fmt_angka = st.column_config.NumberColumn(format="localized")
            st.dataframe(
                tabel_r,
                hide_index=True,
                width="stretch",
                column_config={
                    "Total baris": fmt_angka,
                    "Serangan": fmt_angka,
                    "Perlu ditinjau": fmt_angka,
                },
            )

            def _label(i):
                r = riwayat.iloc[i]
                return f"{r.get('sumber_data', '-')} · {r.get('waktu_analisis', '-')}"

            pilih_i = st.selectbox(
                "Lihat detail analisis",
                list(range(len(riwayat))),
                format_func=_label,
                key="pilih_riwayat",
            )
            baris = riwayat.iloc[pilih_i]
            jml = _angka(baris.get("total_baris"))
            ser = _angka(baris.get("jumlah_serangan"))
            tin = _angka(baris.get("perlu_ditinjau"))
            st.caption(
                f"Sumber data: **{baris.get('sumber_data', '-')}** · dianalisis"
                f" {baris.get('waktu_analisis', '-')}"
            )
            r1, r2, r3 = st.columns(3)
            r1.markdown(
                kartu("📁", "Total data", fmt_id(jml), "netral", "baris"),
                unsafe_allow_html=True,
            )
            r2.markdown(
                kartu(
                    "🔴",
                    "Serangan terdeteksi",
                    fmt_id(ser),
                    "bahaya",
                    f"{(ser / jml * 100) if jml else 0:.2f}% dari total",
                ),
                unsafe_allow_html=True,
            )
            r3.markdown(
                kartu("🟡", "Perlu ditinjau", fmt_id(tin), "waspada", "keyakinan rendah"),
                unsafe_allow_html=True,
            )
            st.write("")
            dist = baris.get("distribusi")
            if isinstance(dist, dict) and dist:
                with st.container(border=True, key="panel_riwayat_dist"):
                    seri = pd.Series(dist).sort_values(ascending=False)
                    tampil(
                        grafik_batang(
                            seri,
                            "Jumlah baris per kategori",
                            horizontal=True,
                        )
                    )
            id_sql = baris.get("sql_run_id") if dari_fb else baris.get("id")
            if supabase_client is not None and id_sql is not None and not pd.isna(id_sql):
                try:
                    log = baca_log(int(id_sql))
                except Exception as e:
                    log = None
                    st.error(f"Gagal membaca log SQL: {e}")
                if log is not None and not log.empty:
                    st.markdown("**Contoh baris log dari database SQL (100 pertama)**")
                    st.dataframe(log, hide_index=True, width="stretch")


# --------------------------------------------------------------- Info Serangan
with tab_info:
    st.subheader("📚 Info Jenis Serangan")
    if not firebase_ok:
        st.info(
            "Data halaman ini dibaca dari Firebase Realtime Database (NoSQL)."
            " Hubungkan Firebase dulu, lihat status di sidebar."
        )
    else:
        try:
            katalog = baca_katalog()
        except Exception as e:
            katalog = None
            st.error(f"Gagal membaca data dari Firebase: {e}")
        if katalog is not None and not katalog:
            st.info(
                "Data katalog belum ada di Firebase. Klik tombol di bawah untuk"
                " mengisinya (cukup sekali)."
            )
            if st.button("📥 Isi data katalog ke Firebase", key="btn_isi_katalog"):
                try:
                    isi_katalog()
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal mengisi katalog: {e}")
        elif katalog:
            st.caption(
                "Data di halaman ini dibaca langsung dari **Firebase Realtime"
                " Database (NoSQL)**."
            )
            urutan = [k for k in WARNA if k in katalog] + sorted(
                k for k in katalog if k not in WARNA
            )
            kelas_pill = {"Aman": "aman", "Sedang": "sedang"}
            kiri, kanan = st.columns(2)
            for n, kelas in enumerate(urutan):
                item = katalog[kelas] or {}
                tingkat = str(item.get("tingkat", "-"))
                warna = WARNA.get(kelas, "#1e3799")
                with (kiri if n % 2 == 0 else kanan):
                    with st.container(border=True, key=f"panel_katalog_{n}"):
                        st.markdown(
                            f'<span style="display:inline-block;width:12px;height:12px;'
                            f'border-radius:50%;background:{warna};margin-right:8px"></span>'
                            f"<b style='font-size:1.1rem'>{kelas}</b> "
                            f'<span class="pill {kelas_pill.get(tingkat, "kritis")}"'
                            f' style="margin-left:8px">{tingkat.upper()}</span>',
                            unsafe_allow_html=True,
                        )
                        st.write(item.get("deskripsi", ""))
                        st.caption("🛡️ Saran: " + str(item.get("saran", "-")))