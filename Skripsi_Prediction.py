# Import library yang diperlukan
import streamlit as st  # Framework untuk membangun aplikasi berbasis web
import pandas as pd  # Library untuk manipulasi data tabular
import numpy as np  # Library untuk perhitungan numerik
import matplotlib.pyplot as plt  # Library untuk visualisasi data
from sklearn.preprocessing import MinMaxScaler  # Digunakan untuk normalisasi data
from tensorflow.keras.models import load_model  # Untuk memuat model deep learning
from datetime import datetime, timedelta  # Untuk manipulasi tanggal
import logging  # Logging untuk debugging
import os  # Untuk operasi file
import yfinance as yf

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
    
    # Fungsi untuk memuat data dari Yahoo Finance
def load_data(ticker):
    START = "2015-01-01"
    TODAY = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # Memastikan kolom Date tersedia
    return data

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Prediksi Harga Saham PT Telkom Indonesia")
    st.write("Aplikasi ini memprediksi harga saham menggunakan model LSTM.")

    # Pilihan saham
    stocks = ("AAPL", "GOOG", "MSFT", "GME", "BBCA.JK", "TLKM.JK", "GOTO.JK")
    selected_stock = st.selectbox("Pilih dataset untuk prediksi", stocks)

    # Load data
    data_load_state = st.text("Loading data...")
    data = load_data(selected_stock)
    data_load_state.text("Loading data... done!")

    # Pilih kolom untuk prediksi
    if "Adj Close" in data.columns:
        column = "Adj Close"
    elif "Close" in data.columns:
        column = "Close"
    else:
        st.error("Kolom 'Adj Close' atau 'Close' tidak tersedia dalam dataset.")
        return

    st.subheader(f"Data {selected_stock} (menggunakan kolom '{column}')")
    st.write(data.head())

    # Konversi kolom Date ke datetime dan set sebagai indeks
    if 'Date' not in data.columns:
        st.error("Dataset tidak memiliki kolom 'Date'.")
        return
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Plot data historis
    st.subheader("Grafik Harga Saham Historis")
    plt.figure(figsize=(12, 6))
    plt.plot(data[column], label="Harga Saham Historis")
    plt.legend()
    st.pyplot(plt)

    # Memuat model dan scaler
    model, scaler = load_model_and_scaler()
    scaler.fit(data[[column]])

    # Input parameter prediksi
    days_ahead = st.number_input("Jumlah hari ke depan untuk prediksi:", min_value=1, max_value=365, value=7)
    window_size = 10
    if len(data) < window_size:
        st.error(f"Data tidak cukup untuk prediksi. Minimal {window_size} data diperlukan.")
        return

    # Tombol untuk memulai prediksi
    if st.button("Prediksi Harga Saham"):
        # Memastikan dataset cukup panjang untuk prediksi
        if days_ahead > len(data):
            st.error(f"Jumlah hari untuk prediksi terlalu besar. Dataset hanya memiliki {len(data)} data.")
            return

        # Melakukan prediksi harga saham
        predictions = predict_future_prices(model, scaler, data[[column]].values, window_size, days_ahead)
        if predictions is None or len(predictions) == 0:
            st.error("Prediksi gagal. Periksa kembali data atau konfigurasi model.")
            return

        # Membuat dataframe hasil prediksi
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
        prediction_df = pd.DataFrame({"Tanggal": future_dates, "Harga Prediksi": predictions})

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        st.write(prediction_df)

        # Menampilkan grafik hasil prediksi
        st.subheader("Grafik Prediksi Harga Saham")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data[column], label="Harga Historis")
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
