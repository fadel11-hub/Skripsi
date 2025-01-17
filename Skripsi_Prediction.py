# Import library yang diperlukan
import streamlit as st  # Framework untuk membangun aplikasi berbasis web
import pandas as pd  # Library untuk manipulasi data tabular
import numpy as np  # Library untuk perhitungan numerik
import matplotlib.pyplot as plt  # Library untuk visualisasi data
from sklearn.preprocessing import MinMaxScaler  # Digunakan untuk normalisasi data
from tensorflow.keras.models import load_model  # Untuk memuat model deep learning
from datetime import timedelta  # Untuk manipulasi tanggal
import logging  # Logging untuk debugging
import os  # Untuk operasi file

# Mengatur logging untuk debugging
logging.basicConfig(level=logging.INFO)

# Fungsi untuk memuat model LSTM dan scaler
def load_model_and_scaler():
    # Mengecek apakah file model LSTM tersedia
    if not os.path.exists("lstm_model.h5"):
        st.error("File model 'lstm_model.h5' tidak ditemukan. Harap unggah model terlebih dahulu.")
        return None, None
    # Memuat model LSTM yang telah disimpan
    model = load_model("lstm_model.h5")
    # Menginisialisasi MinMaxScaler untuk normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    return model, scaler

# Fungsi untuk memprediksi harga saham masa depan
def predict_future_prices(model, scaler, data, window_size, days_ahead):
    try:
        # Melakukan normalisasi data menggunakan scaler
        scaled_data = scaler.transform(data)
        # Mengambil data terakhir berdasarkan window size untuk input model
        last_known_data = scaled_data[-window_size:]

        # Daftar untuk menyimpan hasil prediksi
        predictions = []
        for _ in range(days_ahead):
            # Membentuk data input dengan dimensi yang sesuai untuk model
            input_data = last_known_data.reshape(1, window_size, 1)
            # Melakukan prediksi menggunakan model
            predicted_price = model.predict(input_data)[0, 0]
            predictions.append(predicted_price)
            # Memperbarui data input dengan hasil prediksi
            last_known_data = np.append(last_known_data[1:], predicted_price)

        # Mengubah hasil prediksi kembali ke skala aslinya
        predictions_original = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions_original.flatten()
    except Exception as e:
        # Menangkap error dan menampilkan pesan di Streamlit
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return []

# Fungsi utama untuk aplikasi Streamlit
def main():
    # Membuat judul dan deskripsi aplikasi
    st.title("Stock Price Prediction using BILSTM Model")
    st.write("Predicting the future stock prices for PT Telkom Indonesia using an LSTM model.")

    # Mengunggah file dataset
    uploaded_file = st.file_uploader("Upload a CSV file containing stock data", type="csv")

    if uploaded_file:
        # Membaca dataset dari file yang diunggah
        df = pd.read_csv(uploaded_file)
        # Mengecek apakah kolom 'Date' ada dalam dataset
        if 'Date' not in df.columns:
            st.error("Dataset harus memiliki kolom 'Date'.")
            return
        # Mengubah kolom 'Date' menjadi format datetime dan menjadikannya sebagai indeks
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Menampilkan preview dataset
        st.subheader("Dataset Preview")
        st.write(df.head())

        # Memilih kolom untuk prediksi
        column = st.selectbox("Pilih kolom untuk prediksi:", df.columns)
        # Memastikan kolom yang dipilih berupa data numerik
        if not np.issubdtype(df[column].dtype, np.number):
            st.error("Kolom yang dipilih harus berupa data numerik.")
            return

        # Menampilkan grafik harga saham historis
        st.subheader("Grafik Harga Saham Historis")
        plt.figure(figsize=(12, 6))
        plt.plot(df[column], label="Harga Saham Historis")
        plt.legend()
        st.pyplot(plt)

        # Memuat model dan scaler
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            return

        # Melatih scaler menggunakan data kolom yang dipilih
        scaler.fit(df[[column]])

        # Input jumlah hari untuk prediksi
        days_ahead = st.number_input("Jumlah hari ke depan untuk prediksi:", min_value=1, max_value=365, value=7)
        window_size = 10  # Ukuran jendela untuk sliding window

        # Tombol untuk memulai prediksi
        if st.button("Prediksi Harga Saham"):
            # Memastikan dataset cukup panjang untuk prediksi
            if days_ahead > len(df):
                st.error(f"Jumlah hari untuk prediksi terlalu besar. Dataset hanya memiliki {len(df)} data.")
                return

            # Melakukan prediksi harga saham
            predictions = predict_future_prices(model, scaler, df[[column]].values, window_size, days_ahead)
            if predictions is None or len(predictions) == 0:
                st.error("Prediksi gagal. Periksa kembali data atau konfigurasi model.")
                return

            # Membuat dataframe hasil prediksi
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
            prediction_df = pd.DataFrame({"Tanggal": future_dates, "Harga Prediksi": predictions})

            # Menampilkan hasil prediksi
            st.subheader("Hasil Prediksi")
            st.write(prediction_df)

            # Menampilkan grafik hasil prediksi
            st.subheader("Grafik Prediksi Harga Saham")
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[column], label="Harga Historis")
            plt.plot(future_dates, predictions, label="Harga Prediksi", linestyle="dashed", color="orange")
            plt.legend()
            st.pyplot(plt)

            # Menambahkan opsi untuk mengunduh hasil prediksi
            st.download_button(
                label="Download Prediksi sebagai CSV",
                data=prediction_df.to_csv(index=False),
                file_name='prediksi_harga_saham.csv',
                mime='text/csv',
            )

# Memulai aplikasi Streamlit
if __name__ == "__main__":
    main()
