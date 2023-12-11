import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Fungsi untuk menghitung statistik audio
def calculate_statistics(audio_path):
    y, sr = librosa.load(audio_path)

    # Menghitung statistik
    mean = np.mean(y)
    std_dev = np.std(y)
    max_value = np.max(y)
    min_value = np.min(y)
    median = np.median(y)
    skewness = skew(y)
    kurt = kurtosis(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    mode_value, _ = mode(y)
    iqr = q3 - q1

    if isinstance(mode_value, (list, np.ndarray)) and len(mode_value) > 0:
        mode_value_result = mode_value[0]
    else:
        mode_value_result = mode_value

    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
    zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
    zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
    zcr_skew = skew(librosa.feature.zero_crossing_rate(y=y)[0])

    rms = np.sum(y**2) / len(y)
    rms_median = np.median(y**2)
    rms_std_dev = np.std(y**2)
    rms_kurtosis = kurtosis(y**2)
    rms_skew = skew(y**2)

    return [mean, std_dev, max_value, min_value, median, skewness, kurt, q1, q3, mode_value_result, iqr, zcr_mean, zcr_median, zcr_std_dev, zcr_kurtosis, zcr_skew, rms, rms_median, rms_std_dev, rms_kurtosis, rms_skew]

# Memuat model dan skalar yang telah dilatih
with open('zscore1.pkl', 'rb') as file:
    normalisasi_zscore = pickle.load(file)

with open('minmax1.pkl', 'rb') as file:
    minmaxscaler = pickle.load(file)

# Memuat model KNN untuk kategori emosi dengan normalisasi ZScore
with open('model_zscore_terbaik.pkl', 'rb') as file:
    knn_zscore = pickle.load(file)

# Memuat model KNN untuk kategori emosi dengan normalisasi MinMax
with open('model_knn_terbaik_minmax.pkl', 'rb') as file:
    knn_minmax = pickle.load(file)

# Aplikasi Streamlit
st.title("Deteksi Emosi Audio")
st.markdown("**Nama  : Muhammad Bagus Effendy Yahya**")
st.markdown("**NIM   : 210411100115**")
st.write("Unggah file audio untuk mendeteksi emosinya.")

uploaded_file = st.file_uploader("Upload Audio (Maksimal 10 file)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Deteksi Audio"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)

        # Normalisasi data menggunakan ZScore
        data_ternormalisasi_zscore = normalisasi_zscore.transform([statistik])[0]

        # Prediksi label emosi dengan normalisasi ZScore
        label_emosi_zscore = knn_zscore.predict(data_ternormalisasi_zscore.reshape(1, -1))[0]

        # Normalisasi data menggunakan MinMax
        data_ternormalisasi_minmax = minmaxscaler.transform([statistik])[0]

        # Prediksi label emosi dengan normalisasi MinMax
        label_emosi_minmax = knn_minmax.predict(data_ternormalisasi_minmax.reshape(1, -1))[0]
        
        tabs = st.tabs(["Extract", "Prediction"])
        with tabs[0]:
            st.write("Statistik Audio:")
            st.write(pd.DataFrame(statistik, columns=["Statistics"]))
        
        with tabs[1]:
            st.write("Emosi Terdeteksi (ZScore):", label_emosi_zscore)
            st.write("Emosi Terdeteksi (MinMax):", label_emosi_minmax)

        # Hapus file audio yang diunggah
        os.remove(audio_path)
    
