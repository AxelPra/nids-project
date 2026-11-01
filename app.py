import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Batas baris untuk laporan lengkap (semua kolom). Di atas ini, laporan hanya berisi hasil prediksi.
MAX_LAPORAN_LENGKAP = 200_000

# Konfigurasi Halaman
st.set_page_config(
    page_title="NIDS Security Operations Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# Custom CSS agar tampilannya mewah, modern, dan tidak kaku
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stMetric label {
        color: #4b6584 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #2f3542 !important;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .risk-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Utama
st.title("🛡️ Network Intrusion Detection System (NIDS) Security Center")
st.markdown(
    "Dashboard analisis keamanan jaringan berbasis *Machine Learning*"
    " (**Random Forest** - Dataset **CICIDS2017**). Memproses data trafik"
    " yang diunggah dalam bentuk file CSV."
)
st.markdown("---")


# Load Model
@st.cache_resource
def load_model():
  return joblib.load("model_cicids_1juta.pkl")


try:
  model = load_model()
  model_status = "Aktif & Stabil (Akurasi: 99.87%)"
except Exception as e:
  model = None
  model_status = f"Error: {e}"

# Sidebar untuk Navigasi & Upload
st.sidebar.header("⚙️ Kontrol Panel")
st.sidebar.info(
    f"**Status Model:**\nAlgoritma: Random Forest\nStatus: {model_status}"
)

st.sidebar.header("📂 Sumber Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV data jaringan", type=["csv"]
)

if uploaded_file is not None:
  # File hanya dibaca SEKALI, saat tombol diklik (bukan di setiap rerun)
  if st.button("🚀 Jalankan Analisis & Deteksi", type="primary"):
    if model is None:
      st.error("Model gagal dimuat, analisis tidak dapat dijalankan.")
    else:
      # Hapus hasil lama agar memori tidak menumpuk
      for key in ["df_result", "preds", "csv_result", "report_label", "report_name"]:
        st.session_state.pop(key, None)

      with st.spinner("Model sedang memproses inferensi data..."):
        df_test = pd.read_csv(uploaded_file)

        # Perkecil memori: float64 -> float32
        num_cols = df_test.select_dtypes(include=[np.number]).columns
        df_test[num_cols] = df_test[num_cols].astype("float32")

        preds = model.predict(df_test[num_cols])
        df_test["Hasil_Prediksi"] = pd.Categorical(preds)

        # Buat file laporan SEKALI saja
        if len(df_test) <= MAX_LAPORAN_LENGKAP:
          csv_bytes = df_test.to_csv(index=False).encode("utf-8")
          report_label = "Download Laporan Lengkap (.CSV)"
          report_name = "laporan_audit_keamanan_nids.csv"
        else:
          csv_bytes = (
              df_test[["Hasil_Prediksi"]]
              .to_csv(index_label="no_baris")
              .encode("utf-8")
          )
          report_label = "Download Hasil Prediksi (.CSV)"
          report_name = "hasil_prediksi_nids.csv"

        st.session_state["df_result"] = df_test
        st.session_state["preds"] = preds
        st.session_state["csv_result"] = csv_bytes
        st.session_state["report_label"] = report_label
        st.session_state["report_name"] = report_name
        st.success("Analisis keamanan selesai!")

  # Jika sudah ada hasil prediksi
  if "df_result" in st.session_state:
    df_res = st.session_state["df_result"]
    preds = st.session_state["preds"]

    counts = pd.Series(preds).value_counts()
    total_data = len(df_res)
    total_serangan = total_data - counts.get("Normal Traffic", 0)
    persen_serangan = (
        (total_serangan / total_data) * 100 if total_data > 0 else 0
    )

    # 1. Indikator Status & Level Risiko Keamanan
    st.subheader("📊 Ringkasan Eksekutif Keamanan Jaringan")

    if persen_serangan == 0:
      st.markdown(
          '<span class="risk-badge" style="background-color: #2ed573;">STATUS:'
          " AMAN (0% Ancaman)</span>",
          unsafe_allow_html=True,
      )
    elif persen_serangan < 20:
      st.markdown(
          '<span class="risk-badge" style="background-color: #ffa502;">RISIKO:'
          f" SEDANG ({persen_serangan:.2f}% Anomali)</span>",
          unsafe_allow_html=True,
      )
    else:
      st.markdown(
          '<span class="risk-badge" style="background-color: #ff4757;">RISIKO:'
          f" KRITIS ({persen_serangan:.2f}% Serangan Tinggi)</span>",
          unsafe_allow_html=True,
      )

    # Kotak Metrik Utama (3 Kolom)
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric(label="📁 Total Data Diuji", value=f"{total_data:,} paket")
    with col2:
      st.metric(
          label="🟢 Trafik Normal",
          value=f"{counts.get('Normal Traffic', 0):,}",
      )
    with col3:
      st.metric(
          label="🔴 Total Anomali / Serangan",
          value=f"{total_serangan:,}",
          delta=f"{-total_serangan if total_serangan > 0 else 0} ancaman",
          delta_color="inverse",
      )

    st.markdown("---")

    # 2. Visualisasi Grafik & Detail Statistik Interaktif
    st.subheader("📈 Distribusi Kategori Trafik & Serangan Jaringan")
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
      st.bar_chart(counts)
      st.caption(
          "Grafik batang di atas menunjukkan frekuensi kemunculan setiap jenis"
          " kategori trafik berdasarkan hasil prediksi model."
      )

    with col_table:
      st.write("**Rekapitulasi Kategori:**")
      st.dataframe(counts, width="stretch")

    st.markdown("---")

    # 3. Filter Berdasarkan Kategori Serangan
    st.subheader("🔍 Filter & Eksplorasi Berdasarkan Kategori Trafik")
    unique_labels = [str(x) for x in df_res["Hasil_Prediksi"].unique()]
    selected_filter = st.selectbox(
        "Pilih kategori trafik yang ingin ditampilkan:",
        options=["Semua Kategori"] + unique_labels,
    )

    if selected_filter == "Semua Kategori":
      df_filtered = df_res
    else:
      df_filtered = df_res[df_res["Hasil_Prediksi"] == selected_filter]

    st.dataframe(df_filtered.head(100), width="stretch")
    st.caption(
        f"Menampilkan 100 baris pertama dari {len(df_filtered):,} baris data"
        f" untuk kategori: **{selected_filter}**"
    )

    # 4. Tombol Download Laporan CSV
    crypto_col1, crypto_col2 = st.columns([1, 2])
    with crypto_col1:
      st.markdown("---")
      st.subheader("📥 Ekspor Laporan")
      st.download_button(
          label=st.session_state["report_label"],
          data=st.session_state["csv_result"],
          file_name=st.session_state["report_name"],
          mime="text/csv",
      )
      if total_data > MAX_LAPORAN_LENGKAP:
        st.caption(
            f"Data lebih dari {MAX_LAPORAN_LENGKAP:,} baris, sehingga laporan"
            " hanya berisi hasil prediksi per baris."
        )
else:
  st.info(
      "👈 Silakan *upload* file CSV sampel data jaringan melalui panel di"
      " sebelah kiri untuk memulai audit keamanan."
  )

  with st.expander("📖 Panduan Singkat Penggunaan Dashboard"):
    st.markdown("""
        1. Siapkan file CSV pengujian jaringan dataset CICIDS2017.
        2. Masukkan file melalui tombol **Browse files** di panel *sidebar* kiri.
        3. Klik tombol utama **🚀 Jalankan Analisis & Deteksi**.
        4. Amati **Badge Level Risiko**, kartu metrik eksekutif, dan grafik distribusi serangan.
        5. Gunakan fitur **Filter Kategori Trafik** untuk melihat detail jenis trafik tertentu.
        6. Unduh laporan hasil audit dalam format CSV.
        """)