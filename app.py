import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(
    page_title="NIDS Dashboard CICIDS2017", 
    page_icon="🛡️", 
    layout="wide"
)

# Custom CSS agar kartu metrik kontras, jelas, dan tidak kaku
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

# Header Utama
st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("Aplikasi cerdas deteksi dan klasifikasi serangan siber berbasis *Machine Learning* (**Random Forest**) menggunakan dataset **CICIDS2017**.")
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('model_cicids_1juta.pkl')

try:
    model = load_model()
    model_status = "Aktif (Akurasi: 99.87%)"
except Exception as e:
    model_status = f"Error: {e}"

# Sidebar untuk Navigasi & Upload
st.sidebar.header("⚙️ Kontrol Panel")
st.sidebar.info(f"**Status Model:**\n- Algoritma: Random Forest\n- Status: {model_status}")

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV data jaringan", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Membaca file data jaringan...'):
        df_test = pd.read_csv(uploaded_file)
    
    # Tombol Aksi Utama
    if st.button("🚀 Jalankan Analisis & Deteksi", type="primary"):
        with st.spinner('Model sedang memproses inferensi data...'):
            X_infer = df_test.select_dtypes(include=[np.number])
            preds = model.predict(X_infer)
            df_test['Hasil_Prediksi'] = preds
            
            st.session_state['df_result'] = df_test
            st.session_state['preds'] = preds
            st.success("Analisis selesai!")

    # Jika sudah ada hasil prediksi
    if 'df_result' in st.session_state:
        df_res = st.session_state['df_result']
        preds = st.session_state['preds']
        
        counts = pd.Series(preds).value_counts()
        total_data = len(df_res)
        total_serangan = total_data - counts.get('Normal Traffic', 0)
        
        # 1. Kotak Metrik Informasi Utama
        st.subheader("📊 Ringkasan Eksekutif Keamanan Jaringan")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📁 Total Data Diuji", value=f"{total_data:,}")
        with col2:
            st.metric(label="🟢 Trafik Normal", value=f"{counts.get('Normal Traffic', 0):,}")
        with col3:
            st.metric(label="🔴 Total Anomali / Serangan", value=f"{total_serangan:,}", delta=f"{-total_serangan if total_serangan > 0 else 0}")

        # Peringatan Status Keamanan
        if total_serangan > 0:
            st.error(f"⚠️ **PERINGATAN KEAMANAN:** Terdeteksi **{total_serangan}** aktivitas mencurigakan / serangan pada jaringan yang diuji!")
        else:
            st.success("✅ **STATUS AMAN:** Seluruh trafik jaringan teridentifikasi dalam kondisi normal tanpa ancaman.")

        st.markdown("---")

        # 2. Visualisasi Grafik & Detail Statistik
        st.subheader("📈 Distribusi Jenis Trafik & Serangan")
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.bar_chart(counts)
        with col_table:
            st.write("**Rekapitulasi Jumlah:**")
            st.dataframe(counts, use_container_width=True)

        # 3. Preview Tabel Data & Hasil Prediksi
        st.subheader("🔍 Detail Sampel Data dan Hasil Klasifikasi")
        st.dataframe(df_res.head(100), use_container_width=True)
        st.caption("Menampilkan 100 baris pertama dari data yang telah diuji.")

        # 4. Tombol Download Laporan CSV
        st.markdown("---")
        st.subheader("📥 Ekspor Laporan Hasil Deteksi")
        csv_result = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Laporan Lengkap (.CSV)",
            data=csv_result,
            file_name='laporan_deteksi_nids.csv',
            mime='text/csv',
        )
else:
    st.info("👈 Silakan *upload* file CSV sampel data jaringan melalui panel di sebelah kiri untuk memulai pengujian.")
    
    with st.expander("📖 Panduan Singkat Penggunaan Dashboard"):
        st.markdown("""
        1. Siapkan file CSV data pengujian jaringan.
        2. Klik tombol **Browse files** di *sidebar* sebelah kiri untuk meng-upload file.
        3. Klik tombol utama **🚀 Jalankan Analisis & Deteksi**.
        4. Lihat ringkasan metrik keamanan bergaya kartu, grafik distribusi serangan, dan tabel hasil klasifikasi secara *real-time*.
        5. Unduh laporan lengkap dalam format CSV menggunakan tombol ekspor yang tersedia.
        """)
