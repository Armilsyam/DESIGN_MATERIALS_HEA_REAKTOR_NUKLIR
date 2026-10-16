import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="In-Silico HEA Simulator", layout="wide", page_icon="⚛️")
st.title("⚛️ Simulator In-Silico: High-Entropy Alloys (Fe-Cr-Ni-Mo-Al)")
st.markdown("""
Aplikasi prediksi *Yield Strength* dan stabilitas fasa termodinamika menggunakan arsitektur **Physics-Informed Neural Network (PINN)**.
""")

# (Opsional) Memuat model yang sudah disimpan dari Colab
# @st.cache_resource
# def load_pinn_model():
#     return load_model('HEA_PINN_Evaluator.h5', compile=False)
# model = load_pinn_model()

st.sidebar.header("Input Komposisi Material")
st.sidebar.write("Masukkan rasio fraksi atomik (Sistem akan menormalisasi total menjadi 100%)")

# 2. Modul Input Pengguna di Sidebar
fe = st.sidebar.number_input("Besi (Fe) %", min_value=0.0, max_value=100.0, value=48.20)
cr = st.sidebar.number_input("Kromium (Cr) %", min_value=0.0, max_value=100.0, value=20.50)
ni = st.sidebar.number_input("Nikel (Ni) %", min_value=0.0, max_value=100.0, value=9.50)
mo = st.sidebar.number_input("Molybdenum (Mo) %", min_value=0.0, max_value=100.0, value=3.10)
al = st.sidebar.number_input("Aluminium (Al) %", min_value=0.0, max_value=100.0, value=18.70)

# 3. Normalisasi Komposisi (Memastikan Total 1.0)
total = fe + cr + ni + mo + al
if total == 0:
    st.sidebar.error("Total komposisi tidak boleh 0!")
else:
    fe_norm, cr_norm, ni_norm, mo_norm, al_norm = fe/total, cr/total, ni/total, mo/total, al/total

st.sidebar.markdown(f"**Total Rasio Aktual: {total:.2f}%**")

# 4. Tombol Prediksi dan Inferensi Model
if st.button("🚀 Prediksi Performa Material", use_container_width=True):
    # Format input ke bentuk tensor (1, 5)
    input_data = np.array([[fe_norm, cr_norm, ni_norm, mo_norm, al_norm]])
    
    with st.spinner("Mengeksekusi model PINN..."):
        # Simulasi output prediksi (Ganti dengan model.predict di sistem nyata)
        # prediksi = model.predict(input_data)[0]
        # yield_str, omega, delta = prediksi[0], prediksi[1], prediksi[2]
        
        # Output dummy yang disesuaikan dengan Katalog Opt-1 untuk contoh ini
        yield_str, omega, delta = 1450.0, 1.18, 5.2
        
    # 5. Dashboard Visualisasi Hasil (Metrik)
    st.subheader("📊 Hasil Prediksi Komputasi")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Kekuatan Luluh (Yield Strength)", f"{yield_str:.2f} MPa", "Kapasitas Mekanik")
    col2.metric("Parameter Stabilitas (Ω)", f"{omega:.2f}", "Batas Fisika: >= 1.1")
    col3.metric("Distorsi Kisi (δ)", f"{delta:.2f} %", "Batas Fisika: <= 6.6%")
    
    # 6. Sistem Peringatan Termodinamika Dini (Early Warning System)
    st.markdown("---")
    st.subheader("⚠️ Sistem Peringatan Termodinamika")
    
    fasa_aman = True
    if omega < 1.1:
        st.error(f"Peringatan! Parameter Ω ({omega:.2f}) di bawah 1.1. Rawan terbentuk fasa intermetalik getas.")
        fasa_aman = False
    
    if delta > 6.6:
        st.error(f"Peringatan! Distorsi kisi δ ({delta:.2f}%) melampaui batas 6.6%. Rawan retak struktural.")
        fasa_aman = False
        
    if fasa_aman:
        fasa_prediksi = "BCC (Body-Centered Cubic)" if yield_str >= 1000 else "FCC (Face-Centered Cubic)"
        st.success(f"Status Termodinamika: AMAN. Prediksi fasa dominan adalah **{fasa_prediksi}**.")
